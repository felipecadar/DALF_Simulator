from torch.utils.data import Dataset
import torch
import re, os, json, glob
import cv2
import numpy as np
from functools import reduce, lru_cache
from scipy.spatial import KDTree
from tqdm import tqdm
import kornia as K

from typing import Union

filetypes = {
    'rgb': r'rgba_\d+\.png$',
    'depth': r'depth_\d+\.png$',
    'normal': r'normal_\d+\.png$',
    'forward_flow': r'forward_flow_\d+\.png$',
    'backward_flow': r'backward_flow_\d+\.png$',
    'segmentation': r'segmentation_\d+\.png$',
    'object_coordinates': r"object_coordinates_\d+\.png$",
    'uv': r"uv_\d+\.png$",
}

def filterByType(files, filetype):
    # match the file type
    pattern = filetypes[filetype]
    matches = []
    for filename in files:
        if re.search(pattern, filename):
            matches.append(filename)
    matches = sorted(matches)
    return matches

def NOCS2Color(NOCS):
    return ((NOCS.astype(float) / 65536 ) * 255).astype(np.uint8)

@lru_cache(1000)
def load_sample(rgb_path):
    image = cv2.imread(rgb_path)
    segmentation = cv2.imread(rgb_path.replace('rgba', 'segmentation'), cv2.IMREAD_UNCHANGED)
    uv_coords = cv2.imread(rgb_path.replace('rgba', 'uv'), cv2.IMREAD_UNCHANGED)
    bgmask = cv2.imread(rgb_path.replace('rgba', 'bgmask'), cv2.IMREAD_UNCHANGED)

    return {
        'image': image,
        'segmentation': segmentation,
        'uv_coords': uv_coords,
        'bgmask': bgmask,
    }

def get_GT_kps(
        coords1,
        coords2,
        segmentation1,
        segmentation2,
        kps1,
        threshold = 300,
):
    '''
    coords1: (H, W, 3) # Object coordinates
    coords2: (H, W, 3) # Object coordinates
    segmentation1: (H, W) # instance segmentation
    segmentation2: (H, W) # instance segmentation
    kps1: (N, 2) # keypoints coordinates in the first image

    return:
        kps_gt: (N, 2) # GT keypoint coordinates.
        gt_dist: (N,) # to measure how reliable is the GT. Lower is better
        colors: (N, 3) # RGB Colors of the keypoints based on the coordinates.
    '''
    H, W, _ = coords1.shape
    instances1 = np.unique(segmentation1.reshape(-1), axis=0)
    instances2 = np.unique(segmentation2.reshape(-1), axis=0)
    shared_instances = list(set(instances1).intersection(set(instances2)))
    # shared_instances = list(filter(lambda x: x > 1, shared_instances))

    # from int to float
    coords1 = coords1.astype(np.float32)
    coords2 = coords2.astype(np.float32)

    kps_gt = np.zeros_like(kps1) - 1
    gt_dist = np.zeros(len(kps1)) + np.inf

    for instance_id in shared_instances:
        # print(f"Instance {instance_id}")
        seg_mask1 = (segmentation1 == instance_id).astype(np.uint8)
        seg_mask2 = (segmentation2 == instance_id).astype(np.uint8)
        coords1_masked = coords1 * seg_mask1[:, :, None]
        coords2_masked = coords2 * seg_mask2[:, :, None]

        tree2 = KDTree(coords2_masked.reshape(-1, 3))

        # if seg_mask1.sum() == 0 or seg_mask2.sum() == 0:
        #     continue
        
        kps_count = 0
        for kp_idx in range(len(kps1)):
            # eval just the keypoints that belong to the same instance
            # to avoid wrong correspondences
            source_kps = kps1[kp_idx]

            if not seg_mask1[source_kps[1], source_kps[0]]:
                continue

            kps_count += 1
            NOCS1 = coords1_masked[source_kps[1], source_kps[0]]
            dists2, indexes2 = tree2.query(NOCS1, k=1)

            if dists2 > threshold:
                continue
            else:
                # find coordinates of the NOCS2 in the image
                y2, x2 = np.unravel_index(indexes2, (H, W))
                assert np.allclose(coords2_masked[y2, x2], coords2_masked.reshape(-1, 3)[indexes2])


            kps_gt[kp_idx] = [x2, y2]
            gt_dist[kp_idx] = dists2
        # print(f"Instance {instance_id}: {kps_count} keypoints")
    kps_gt = kps_gt.astype(float)

    return {
        'kps_gt': kps_gt,
        'gt_dist': gt_dist,
    }



LOCAL_DATA = '/work/cadar/Datasets/simulation_v2/train_single_obj/'

class KubrickInstances(Dataset):
    default_config = {
        "data_dir": LOCAL_DATA,
        "max_pairs": -1,
        "return_tensors": True,
        "splits": ['illumination-viewpoint', 'deformation_2', 'deformation_2-illumination-viewpoint', 'deformation_1', 'deformation_1-illumination-viewpoint', 'deformation_3', 'deformation_3-illumination-viewpoint']
    }

    def __init__(self, config={}) -> None:
        super().__init__()

        self.config = {**self.default_config}
        self.config.update(config)
        
        dataset_path = self.config['data_dir']

        with open(dataset_path + 'selected_pairs_v2.json') as f:
            self.experiments_definition = json.load(f)
            
        self.reload_pairs()
            
        # self.all_samples = {}
        # for image_path in tqdm(self.all_images, desc="Loading all images"):
        #     self.all_samples[image_path] = load_sample(image_path)
            
        self.sample_image = cv2.imread(self.all_images[0])
        
    def reload_pairs(self):
        global_pairs = [ self.experiments_definition[key][subset] for key in self.config['splits'] for subset in self.experiments_definition[key] ]
        global_pairs = reduce(lambda x, y: x + y, global_pairs)
        
        if self.config['max_pairs'] > 0:
            global_pairs = global_pairs[:self.config['max_pairs']]

        # Get all images recursively
        self.all_images = []

        self.all_pairs = []
        for pair in global_pairs:
            if pair[0].startswith('/'):
                pair[0] = pair[0][1:]
            if pair[1].startswith('/'):
                pair[1] = pair[1][1:]
            
            self.all_pairs.append({
                "image0_path": os.path.join(self.config['data_dir'], pair[0]),
                "image1_path": os.path.join(self.config['data_dir'], pair[1]),
            })

            self.all_images.append(self.all_pairs[-1]['image0_path'])
            self.all_images.append(self.all_pairs[-1]['image1_path'])
            
        self.all_images = list(set(self.all_images))
    
    def load_sample(self, rgb_path):
        # if rgb_path not in self.all_samples:
        #     self.all_samples[rgb_path] = load_sample(rgb_path)
        # return self.all_samples[rgb_path]
        return load_sample(rgb_path)

    def countExps(self):
        print("Found {} images".format(len(self.all_images)))
        for key in self.config['experiments']:
            print(f"Exp: {key}: {len(self.experiments_definition[key])} pairs.")

    def __len__(self) -> int:
        return len(self.all_pairs)
    
    def __getitem__(self, index: int):
        item_dict = self.all_pairs[index].copy()

        sample0 = self.load_sample(item_dict['image0_path'])
        sample1 = self.load_sample(item_dict['image1_path'])

        return_dict = {}

        for key in sample0:
            return_dict[key+'0'] = sample0[key]

        for key in sample1:
            return_dict[key+'1'] = sample1[key]

        if self.config['return_tensors']:
            for key in return_dict:
                if 'image' in key:
                    img = return_dict[key]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img.astype(float)) / 255
                    img = img.permute([2,0,1])
                    return_dict[key] = img
                else:
                    if isinstance(return_dict[key], np.ndarray):
                        return_dict[key] = torch.from_numpy(return_dict[key].astype(float))
                    else:
                        return_dict[key] = torch.tensor(return_dict[key], dtype=float)

        return return_dict
    
    def sample_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self[np.random.randint(len(self))])
        return batch
        
    def _warp(self, samples, keypoints0, inverse=False):
        if isinstance(keypoints0[0], cv2.KeyPoint):
            keypoints0 = np.array([kp.pt for kp in keypoints0])

        coords0 = samples['uv_coords0']
        coords1 = samples['uv_coords1']

        segmentation0 = samples['segmentation0']
        segmentation1 = samples['segmentation1']

        if isinstance(coords0, torch.Tensor):
            coords0 = coords0.cpu().detach().numpy()
            coords1 = coords1.cpu().detach().numpy()
            segmentation0 = segmentation0.cpu().detach().numpy()
            segmentation1 = segmentation1.cpu().detach().numpy()

        if inverse:
            out = get_GT_kps(coords1, coords0, segmentation1, segmentation0, keypoints0)
            warped_kps = out['kps_gt']
            warped_dists = out['gt_dist']
        else:
            out = get_GT_kps(coords0, coords1, segmentation0, segmentation1, keypoints0)
            warped_kps = out['kps_gt']
            warped_dists = out['gt_dist']

        return warped_kps, warped_dists
    
    def _warp_batch(self, samples, keypoints0, inverse=False):
        warps = []
        for i in range(len(keypoints0)):
            new_samples = {k:v[i] for k, v in samples.items()}
            new_kps = keypoints0[i]
            warps.append(self._warp(new_samples, new_kps, inverse))
        warps = np.array(warps)
        return warps

    def warp(self, samples, keypoints0, inverse=False):
        if isinstance(keypoints0[0], cv2.KeyPoint):
            keypoints0 = np.array([kp.pt for kp in keypoints0])

        keypoints0 = keypoints0.astype(int)

        # check if is batched
        if len(keypoints0.shape) == 3:
            return self._warp_batch(samples, keypoints0, inverse)
        elif len(keypoints0.shape) == 2:
            return self._warp(samples, keypoints0, inverse)
        else:
            raise Exception("Invalid shape for keypoints0. Expected BxNx2 or Nx2")
        
    def warp_torch(self, samples, keypoints0: torch.tensor, inverse=False):
        dev = keypoints0.device
        np_keypoints0 = keypoints0.cpu().detach().numpy()
        np_samples = {k:v.cpu().detach().numpy() for k, v in samples.items()}
        out = self.warp(np_samples, np_keypoints0, inverse)
        warped_kps = torch.tensor(out[0]).to(dev).float()
        warped_dists = torch.tensor(out[1]).to(dev).float()

        return warped_kps, warped_dists

# @torch.compile
def extract_patches(
        image:Union[torch.Tensor, np.ndarray],
        keypoints: Union[torch.Tensor, np.ndarray],
        patch_size:int = 64
    ):

    if isinstance(image, np.ndarray):
        image = K.image_to_tensor(image).float().unsqueeze(0)

    if isinstance(keypoints, np.ndarray):
        keypoints = torch.from_numpy(keypoints).float()

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    grid = torch.zeros((len(keypoints), patch_size, patch_size, 2))

    # grid limits
    x0 = keypoints[:, 0] - patch_size / 2
    x1 = keypoints[:, 0] + patch_size / 2
    y0 = keypoints[:, 1] - patch_size / 2
    y1 = keypoints[:, 1] + patch_size / 2

    # normalize to [-1, 1]
    x0 = x0 / image.shape[2] * 2 - 1
    x1 = x1 / image.shape[2] * 2 - 1
    y0 = y0 / image.shape[3] * 2 - 1
    y1 = y1 / image.shape[3] * 2 - 1

    for i in range(len(keypoints)):
        mesh_i, mesh_j = torch.meshgrid(
            torch.linspace(x0[i], x1[i], patch_size),
            torch.linspace(y0[i], y1[i], patch_size),
            indexing="ij"
        )
        grid[i, :, :, 1] = mesh_i
        grid[i, :, :, 0] = mesh_j

    image_batched = image.repeat(len(keypoints), 1, 1, 1)
    patches = torch.nn.functional.grid_sample(image_batched, grid, align_corners=False )

    return patches

class KubricTriplets(Dataset):
    default_config = {
        "data_dir": LOCAL_DATA,
    }

    def __init__(self, config={}) -> None:
        super().__init__()

        self.config = {**self.default_config, **config}
        dataset_path = self.config['data_dir']

        with open(dataset_path + '/selected_pairs_v2.json') as f:
            self.experiments_definition = json.load(f)
            
        # self.experiments_definition['exp']['subset'] = [p1, p2, covisibility]
        pairs = [ self.experiments_definition[key][subset] for key in self.experiments_definition for subset in self.experiments_definition[key] ]

        # Get all images recursively
        self.all_images = glob.glob(dataset_path + '/**/rgba*.png', recursive=True)
        self.all_pairs = []

        self.sample_image = cv2.imread(self.all_images[0])

        for pair in pairs:
            self.all_pairs.append({
                "image0_path": pair[0],
                "image1_path": pair[1],
            })

        # load all samples
        self.samples = {}
        for image_path in tqdm(self.all_images, desc="Loading all images"):
            self.samples[image_path] = load_sample(image_path)

    def load_sample(self, rgb_path):
        if rgb_path not in self.samples:
            self.samples[rgb_path] = load_sample(rgb_path)
        return self.samples[rgb_path]

    def make_triplet(self, index, n_patches=128, ignore_background=True):
        item_dict = self.all_pairs[index].copy()

        sample0 = self.load_sample(os.path.join(self.config['data_dir'], item_dict['image0_path']))
        sample1 = self.load_sample(os.path.join(self.config['data_dir'], item_dict['image1_path']))

        if ignore_background:
            mask0 = sample0['bgmask'] == 1
            mask1 = sample1['bgmask'] == 1
        else:
            mask0 = np.ones_like(sample0['bgmask'])
            mask1 = np.ones_like(sample1['bgmask'])

        prob_map0 = sample0['segmentation'] * mask0
        prob_map1 = sample1['segmentation'] * mask1

        # sample keypoints based on the probability map
        keypoints_idx0 = np.random.choice(np.arange(prob_map0.size), size=n_patches, p=prob_map0.reshape(-1) / prob_map0.sum())
        keypoints0 = np.unravel_index(keypoints_idx0, prob_map0.shape)

        keypoints1 = get_GT_kps(
            sample0['uv_coords'],
            sample1['uv_coords'],
            sample0['segmentation'],
            sample1['segmentation'],
            keypoints0,
        )['kps_gt']

        valid_idx = keypoints1[:, 0] >= 0

        keypoints0 = keypoints0[valid_idx]
        keypoints1 = keypoints1[valid_idx]

        # extract positive torch patches
        patches_anchor = extract_patches(sample0['image'], keypoints0)
        patches_positive = extract_patches(sample1['image'], keypoints1)
        patches_negative = patches_positive[torch.randperm(len(patches_positive))]

        return patches_anchor, patches_positive, patches_negative

    def __len__(self) -> int:
        return len(self.all_pairs)
    
    def __getitem__(self, index: int):
        return self.make_triplet(index)
    
    def sample_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self[np.random.randint(len(self))])
        return batch


if __name__ == "__main__":

    import sys
    sys.path.append('/work/cadar/Github/DALF_Simulator/')
    from modules.utils import plot_pair, plot_matches, plot_keypoints, show, plot_overlay
    
    sift = cv2.SIFT_create(30)
    dataset = KubrickInstances({
        "data_dir": LOCAL_DATA,
        "return_tensors": False
    })

    torch_dataset = KubrickInstances({
        "data_dir": LOCAL_DATA,
        "return_tensors": True
    })

    item = dataset[0]
    torch_item = torch_dataset[0]
    
    kp0, desc0 = sift.detectAndCompute(item['image0'], None)
    kp1, desc1 = sift.detectAndCompute(item['image1'], None)

    np_kps0 = np.array([kp.pt for kp in kp0])
    np_kps1 = np.array([kp.pt for kp in kp1])

    torch_kps0 = torch.from_numpy(np_kps0)
    torch_kps1 = torch.from_numpy(np_kps1)

    ### TEST PYTORCH 
    warp01, dist01 = dataset.warp_torch(torch_item, torch_kps0)
    # warp10, dist10 = dataset.warp_torch(torch_item, torch_kps1, inverse=True)

    warp01 = warp01.cpu().detach().numpy()
    # valid = warp01[:, 0] > 0
    # warp01 = warp01[valid]
    # dist01 = dist01.cpu().detach().numpy()[valid]

    uv_coords0 = ((torch_item['uv_coords0']/ 2**16) * 255).cpu().numpy().astype(np.uint8)
    uv_coords1 = ((torch_item['uv_coords1']/ 2**16) * 255).cpu().numpy().astype(np.uint8)

    plot_pair(torch_item['image0'], torch_item['image1'])
    plot_overlay(uv_coords0, uv_coords1)
    plot_keypoints(torch_kps0, color='r')
    plot_matches(torch_kps0, warp01, color='g')
    show()

    # joint = np.concatenate([item['image0'], item['image1']], axis=1)

    # for i in range(len(kp0)):
    #     kp = kp0[i].pt
    #     cv2.circle(joint, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
    # for i in range(len(kp1)):
    #     kp = kp1[i].pt
    #     cv2.circle(joint, (int(kp[0]) + 512, int(kp[1])), 3, (0, 255, 0), -1)

    # for i in range(len(kp0)):
    #     kp_src = kp0[i].pt
    #     kp_tgt = warp01[i]

    #     if kp_tgt[0] > 0:
    #         cv2.line(joint, (int(kp_src[0]), int(kp_src[1])), (int(kp_tgt[0]) + 512, int(kp_tgt[1])), (0, 255, 0), 1)

    # # save
    # cv2.imshow("joint.png", joint)
    # cv2.waitKey(0)

    ### TEST OPENCV
    # joint = np.concatenate([item['image0'], item['image1']], axis=1)

    # warp01, dist01 = dataset.warp(item, kp0)
    # warp10, dist10 = dataset.warp(item, kp1, inverse=True)
    # # warp10, dist10 = dataset.warp(item, kp1, inverse=True)

    # # extract image shape
    # H, W, _ = item['image0'].shape

    # for i in range(len(kp0)):
    #     kp = kp0[i].pt
    #     cv2.circle(joint, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
    # for i in range(len(kp1)):
    #     kp = kp1[i].pt
    #     cv2.circle(joint, (int(kp[0]) + 512, int(kp[1])), 3, (0, 255, 0), -1)

    # for i in range(len(kp0)):
    #     kp_src = kp0[i].pt
    #     kp_tgt = warp01[i]

    #     if kp_tgt[0] > 0:
    #         cv2.line(joint, (int(kp_src[0]), int(kp_src[1])), (int(kp_tgt[0]) + 512, int(kp_tgt[1])), (0, 255, 0), 1)

    # # save
    # cv2.imshow("joint_cv2.png", joint)
    
    # cv2.waitKey(0)