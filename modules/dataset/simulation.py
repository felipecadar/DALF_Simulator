from torch.utils.data import Dataset
import torch
import re, os, json, glob
import cv2
import numpy as np
from functools import reduce, lru_cache
from scipy.spatial import KDTree

filetypes = {
    'rgb': r'rgba_\d+\.png$',
    'depth': r'depth_\d+\.png$',
    'normal': r'normal_\d+\.png$',
    'forward_flow': r'forward_flow_\d+\.png$',
    'backward_flow': r'backward_flow_\d+\.png$',
    'segmentation': r'segmentation_\d+\.png$',
    'object_coordinates': r"object_coordinates_\d+\.png$",
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
    return ((NOCS / 65536 ) * 255).astype(int).tolist()


def get_GT_kps(
        coords1: np.ndarray,
        coords2: np.ndarray,
        segmentation1: np.ndarray,
        segmentation2: np.ndarray,
        kps1: np.ndarray,
        threshold: float = 1700,
) -> dict[str, np.ndarray] :
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
    # from torch to numpy
    if isinstance(coords1, torch.Tensor):
        coords1 = coords1.cpu().detach().numpy()
    if isinstance(coords2, torch.Tensor):
        coords2 = coords2.cpu().detach().numpy()
    if isinstance(segmentation1, torch.Tensor):
        segmentation1 = segmentation1.cpu().detach().numpy()
    if isinstance(segmentation2, torch.Tensor):
        segmentation2 = segmentation2.cpu().detach().numpy()
    if isinstance(kps1, torch.Tensor):
        kps1 = kps1.cpu().detach().numpy()
 
    H, W, _ = coords1.shape
    instances = np.unique(segmentation1.reshape(-1), axis=0)

    # from int to float
    if coords1.dtype != np.float32:
        coords1 = coords1.astype(np.float32)
    if coords2.dtype != np.float32:
        coords2 = coords2.astype(np.float32)

    kps_gt = np.zeros_like(kps1)
    gt_dist = np.zeros(len(kps1)) + np.inf
    colors = np.zeros((len(kps1), 3))

    for instance_id in instances:
        seg_mask1 = (segmentation1 == instance_id)
        seg_mask2 = (segmentation2 == instance_id)
        coords2_masked = coords2 * seg_mask2[:, :, None]

        values = coords2_masked.reshape(-1, 3)
        tree = KDTree(values)

        if len(values) == 0:
            print("No keypoints found for this instance" + str(instance_id))
            continue

        for kp_idx in range(len(kps1)):
            # eval just the keypoints that belong to the same instance
            # to avoid wrong correspondences
            source_kps = kps1[kp_idx]
            if not seg_mask1[source_kps[1], source_kps[0]]:
                continue

            NOCS1 = coords1[source_kps[1], source_kps[0]]

            dists, indexes = tree.query(NOCS1, k=1)
            # NOCS2 = values[indexes]

            if dists > threshold:
                # this keypoint is not reliable. Probably its occluded.
                # In previus tests, the threshold was of 1700 can vary up to 2.5 pixels
                # from the GT. Seems to be a good threshold to evaluate the covisibility
                # of the keypoints.
                kps_gt[kp_idx] = [-1, -1]
                gt_dist[kp_idx] = np.inf
                colors[kp_idx] = [0, 0, 0]
            else:

                # find coordinates of the NOCS2 in the image
                y, x = np.unravel_index(indexes, (H, W))

                kps_gt[kp_idx] = [x, y]
                gt_dist[kp_idx] = dists
                colors[kp_idx] = NOCS2Color(NOCS1)
            
    kps_gt = kps_gt.astype(float)

    return {
        'kps_gt': kps_gt,
        'gt_dist': gt_dist,
        'colors': colors,
    }


LOCAL_DATA = '/Users/cadar/Documents/dataset/_dataset_test2/'

class KubrickInstances(Dataset):
    default_config = {
        "data_dir": LOCAL_DATA,
        "max_pairs": -1,
        "return_tensors": True,
    }

    def __init__(self, config={}) -> None:
        super().__init__()

        self.config = {**self.default_config, **config}
        dataset_path = self.config['data_dir']

        # with open(dataset_path + '/selected_pairs.json') as f:
        #     self.experiments_definition = json.load(f)

        pairs = []

        for folder in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, folder)):
                # read the pairs.txt
                pairs_path = os.path.join(dataset_path, folder, 'pairs.txt')
                if os.path.exists(pairs_path):
                    with open(pairs_path) as f:
                        pairs = f.readlines()
                        pairs = [pair.strip().split(' ') for pair in pairs]
                        pairs = [[ pair[0], pair[1], float(pair[2])] for pair in pairs]
                        pairs.extend(pairs)

        # Get all images recursively
        self.all_images = glob.glob(dataset_path + '/**/rgba*.png', recursive=True)
        self.all_pairs = []

        self.sample_image = cv2.imread(self.all_images[0])

        for pair in pairs:
            self.all_pairs.append({
                "image0_path": pair[0],
                "image1_path": pair[1],
                "covisibility": pair[2],
            })

    @lru_cache(1000)
    def load_sample(self, rgb_path):
        image = cv2.imread(rgb_path)
        segmentation = cv2.imread(rgb_path.replace('rgba', 'segmentation'), cv2.IMREAD_UNCHANGED)
        object_coords = cv2.imread(rgb_path.replace('rgba', 'object_coordinates'), cv2.IMREAD_UNCHANGED)

        folder = os.path.dirname(rgb_path)
        parent = os.path.dirname(folder)

        simulation_config = json.load(open(os.path.join(parent, 'simulation_config.json')))
        objects_info = json.load(open(os.path.join(parent, 'objects_info.json')))

        instances = []
        for key, asset in simulation_config['scene_assets'].items():
            ftype, uid = key.split(":")
            if ftype == 'file':
                instances.append(asset['segmentation_id'])

        for object_info in objects_info:
            instances.append(object_info['segmentation_id'])

        return {
            'image': image,
            'segmentation': segmentation,
            'object_coords': object_coords,
            'instances': instances,
        }
    
    def countExps(self):
        print("Found {} images".format(len(self.all_images)))
        for key in self.config['experiments']:
            print(f"Exp: {key}: {len(self.experiments_definition[key])} pairs.")

    def __len__(self) -> int:
        return len(self.all_pairs)
    
    def __getitem__(self, index: int):
        item_dict = self.all_pairs[index].copy()

        sample0 = self.load_sample(os.path.join(self.config['data_dir'], item_dict['image0_path']))
        sample1 = self.load_sample(os.path.join(self.config['data_dir'], item_dict['image1_path']))


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
        
    def _warp(self, samples, keypoints0, inverse=False, return_colors=False):
        if isinstance(keypoints0[0], cv2.KeyPoint):
            keypoints0 = np.array([kp.pt for kp in keypoints0])

        coords0 = samples['object_coords0']
        coords1 = samples['object_coords1']

        segmentation0 = samples['segmentation0']
        segmentation1 = samples['segmentation1']

        if inverse:
            out = get_GT_kps(coords1, coords0, segmentation1, segmentation0, keypoints0)
            warped_kps = out['kps_gt']
            warped_dists = out['gt_dist']
            warped_colors = out['colors']
        else:
            out = get_GT_kps(coords0, coords1, segmentation0, segmentation1, keypoints0)
            warped_kps = out['kps_gt']
            warped_dists = out['gt_dist']
            warped_colors = out['colors']

        if return_colors:
            return warped_kps, warped_dists, warped_colors 

        return warped_kps, warped_dists
    
    def _warp_batch(self, samples, keypoints0, inverse=False, return_colors=False):
        warps = []
        for i in range(len(keypoints0)):
            new_samples = {k:v[i] for k, v in samples.items()}
            new_kps = keypoints0[i]
            warps.append(self._warp(new_samples, new_kps, inverse, return_colors))
        warps = np.array(warps)
        return warps

    def warp(self, samples, keypoints0, inverse=False, return_colors=False):
        if isinstance(keypoints0[0], cv2.KeyPoint):
            keypoints0 = np.array([kp.pt for kp in keypoints0])

        keypoints0 = keypoints0.astype(int)

        # check if is batched
        if len(keypoints0.shape) == 3:
            return self._warp_batch(samples, keypoints0, inverse, return_colors)
        elif len(keypoints0.shape) == 2:
            return self._warp(samples, keypoints0, inverse, return_colors)
        else:
            raise Exception("Invalid shape for keypoints0. Expected BxNx2 or Nx2")
        
    def warp_torch(self, samples, keypoints0: torch.tensor, inverse=False, return_colors=False):
        dev = keypoints0.device
        np_keypoints0 = keypoints0.cpu().detach().numpy()
        np_samples = {k:v.cpu().detach().numpy() for k, v in samples.items()}
        out = self.warp(np_samples, np_keypoints0, inverse, return_colors)
        warped_kps = torch.tensor(out[0]).to(dev)
        warped_dists = torch.tensor(out[1]).to(dev)
        if return_colors:
            warped_colors = torch.tensor(out[2]).to(dev)
            return warped_kps, warped_dists, warped_colors

        return warped_kps, warped_dists


if __name__ == "__main__":
    
    sift = cv2.SIFT_create(10)
    dataset = KubrickInstances({
        "data_dir": "/srv/storage/datasets/cadar/simulator/_dataset_train/",
        "return_tensors": False
    })

    torch_dataset = KubrickInstances({
        "data_dir": "/srv/storage/datasets/cadar/simulator/_dataset_train/",
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

    warp01, dist01, colors01 = dataset.warp_torch(torch_item, torch_kps0, return_colors=True)
    warp10, dist10, colors10 = dataset.warp_torch(torch_item, torch_kps1, inverse=True, return_colors=True)

    warp01 = warp01.cpu().detach().numpy()
    # valid = warp01[:, 0] > 0
    # warp01 = warp01[valid]
    # dist01 = dist01.cpu().detach().numpy()[valid]

    joint = np.concatenate([item['image0'], item['image1']], axis=1)

    for i in range(len(kp0)):
        kp_src = kp0[i].pt
        kp_tgt = warp01[i]

        if kp_tgt[0] > 0:
            cv2.circle(joint, (int(kp_src[0]), int(kp_src[1])), 3, (0, 255, 0), -1)
            cv2.circle(joint, (int(kp_tgt[0]) + 640, int(kp_tgt[1])), 3, (0, 255, 0), -1)
            cv2.line(joint, (int(kp_src[0]), int(kp_src[1])), (int(kp_tgt[0]) + 640, int(kp_tgt[1])), (0, 255, 0), 1)
        else:
            cv2.circle(joint, (int(kp_src[0]), int(kp_src[1])), 3, (0, 0, 255), -1)

    # save
    cv2.imwrite("joint.png", joint)



    ### TEST OPENCV
    joint = np.concatenate([item['image0'], item['image1']], axis=1)

    warp01, dist01 = dataset.warp(item, kp0)
    warp10, dist10 = dataset.warp(item, kp1, inverse=True)
    # warp10, dist10 = dataset.warp(item, kp1, inverse=True)

    # extract image shape
    H, W, _ = item['image0'].shape

    for i in range(len(kp0)):
        # kp_src = kp0[i].pt
        # kp_tgt = warp01[i]

        kp_src = warp10[i]
        kp_tgt = kp1[i].pt

        if kp_tgt[0] > 0:
            cv2.circle(joint, (int(kp_src[0]), int(kp_src[1])), 3, (0, 255, 0), -1)
            cv2.circle(joint, (int(kp_tgt[0]) + W, int(kp_tgt[1])), 3, (0, 255, 0), -1)
            cv2.line(joint, (int(kp_src[0]), int(kp_src[1])), (int(kp_tgt[0]) + W, int(kp_tgt[1])), (0, 255, 0), 1)
        else:
            cv2.circle(joint, (int(kp_src[0]), int(kp_src[1])), 3, (0, 0, 255), -1)

    # save
    cv2.imwrite("joint_cv2.png", joint)
    
