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

from easy_local_features.utils import io, vis, ops
from pathlib import Path

def tps(theta, ctrl, grid):
    N, H, W, _ = grid.size()

    if ctrl.dim() == 2:
        ctrl = ctrl.expand(N, *ctrl.size())
    
    T = ctrl.shape[1]
    diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
    D = torch.sqrt((diff**2).sum(-1))
    U = (D**2) * torch.log(D + 1e-6)
    w, a = theta[:, :-3, :], theta[:, -3:, :]
    reduced = T + 2  == theta.shape[1]
    if reduced:
        w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1) 
    b = torch.bmm(U.view(N, -1, T), w).view(N,H,W,2)
    z = torch.bmm(grid.view(N,-1,3), a).view(N,H,W,2) + b
    return z

def tps_sparse(theta, ctrl, xy):
    if xy.dim() == 2:
        xy = xy.expand(theta.shape[0], *xy.size())
    N, M = xy.shape[:2]
    grid = xy.new(N, M, 3)
    grid[..., 0] = 1.
    grid[..., 1:] = xy
    z = tps(theta, ctrl, grid.view(N,M,1,3))
    return xy + z.view(N, M, 2)

class DeSurT(Dataset):
    def __init__(self, eval_bench='/draft-nvme/cadar/eval_bench', use_cache=False, load_all=False) -> None:
        super().__init__()

        self.all_png = os.path.join(eval_bench, 'All_PNG/DeSurTSampled/')
        self.gt_tps = os.path.join(eval_bench, 'gt_tps/DeSurTSampled/')
        
        self.all_pairs = glob.glob(os.path.join(self.all_png, '*/*-rgb.png'))
        # remove the ones with "cloud_master" in the name
        self.all_pairs = [x for x in self.all_pairs if 'cloud_master' not in x]
        
        self.use_cache = use_cache
        self.cache = {}
        
        if load_all:
            for path in tqdm(self.all_pairs):
                self.cache[path] = io.fromPath(path, batch=False)
                
            # load masters
            for obj_folder in glob.glob(os.path.join(self.all_png, '*/')):
                master = glob.glob(os.path.join(obj_folder, 'cloud_master-rgb.png'))[0]
                self.cache[master] = io.fromPath(master, batch=False)
                
        
    def __len__(self):
        return len(self.all_pairs)

    def fetch_image(self, path):
        if not self.use_cache:
            return io.fromPath(path, batch=False)
        
        if path in self.cache:
            return self.cache[path]
        else:
            img = io.fromPath(path, batch=False)
            self.cache[path] = img
            return img
    
    def __getitem__(self, idx):
        
        img1_path = self.all_pairs[idx]
        # the master is cloud_master-rgb.png
        img0_path = img1_path.replace(img1_path.split('/')[-1], 'cloud_master-rgb.png')
        
        return {
            'image0': io.fromPath(img0_path, batch=False),
            'image1': io.fromPath(img1_path, batch=False),
            'image0_path': img0_path,
            'image1_path': img1_path
        }
        
    def sample_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self[np.random.randint(len(self))])
        return batch
        
    def find_correspondences(self, sample, kps0, kps1):
        img0 = sample['image0']
        img1 = sample['image1']
        
        img0_path = sample['image0_path']
        img1_path = sample['image1_path']
        
        mask0_path = img0_path.replace('-rgb.png', '_objmask.png').replace('All_PNG', 'gt_tps')
        mask1_path = img1_path.replace('-rgb.png', '_objmask.png').replace('All_PNG', 'gt_tps')
        
        dev = kps0.device
        
        mask0 = io.fromPath(mask0_path, batch=False)[0].to(dev)
        mask1 = io.fromPath(mask1_path, batch=False)[0].to(dev)
        
        valid0 = torch.ones(kps0.shape[0], dtype=torch.bool).to(dev)
        valid1 = torch.ones(kps1.shape[0], dtype=torch.bool).to(dev)
        
        # remove keypoints that are in the mask
        for i, kp in enumerate(kps0):
            if mask0[int(kp[1]), int(kp[0])] == 0:
                valid0[i] = False
                
        for i, kp in enumerate(kps1):
            if mask1[int(kp[1]), int(kp[0])] == 0:
                valid1[i] = False
        

        # loading_file = img1_path.replace('-rgb.png', '').replace('All_PNG', 'gt_tps')
        # theta = torch.tensor(np.load(loading_file + '_theta.npy').astype(np.float32))
        # ctrl_pts = torch.tensor(np.load(loading_file + '_ctrlpts.npy').astype(np.float32))
        # score = cv2.imread(loading_file + '_SSIM.png', 0) / 255.0

        # norm_factor = np.array(img1.shape[1:3][::-1], dtype = np.float32)
        # ctrl_pts = torch.tensor(ctrl_pts)
        # warped_coords = tps_sparse(theta, ctrl_pts, (kps1 / norm_factor)).squeeze(0).cpu().numpy() * norm_factor # kp1 projected to image0
        
        warped_coords = self.warp10(kps1, sample)[0].cpu().numpy()
        
        tree = KDTree(kps0.cpu().numpy())
        dists, idxs_ref = tree.query(warped_coords)
        px_thresh = 3.0
        gt_tgt  = np.arange(len(kps1.cpu().numpy()))[ dists < px_thresh] # Groundtruth indexes -- threshold is in pixels 
        gt_ref = idxs_ref[dists < px_thresh] 

        #filter repeated matches
        _, uidxs = np.unique(gt_ref, return_index = True)
        gt_ref = gt_ref[uidxs]
        gt_tgt = gt_tgt[uidxs]
        
        # take out the ones that are not valid
        valid_matches = np.zeros_like(gt_ref, dtype=bool)
        for i, idx in enumerate(gt_ref):
            if valid0[idx] and valid1[gt_tgt[i]]:
                valid_matches[i] = True
                
        gt_ref = gt_ref[valid_matches]
        gt_tgt = gt_tgt[valid_matches]
        
        idxs = torch.tensor(np.stack([gt_ref, gt_tgt], axis=1))
        
        # vis.plot_pair(sample['image0'], sample['image1'])
        # vis.plot_matches(kps0[gt_ref], kps1[gt_tgt])
        # vis.show()
        
        # return indexes of the keypoints
        return idxs
    
    def warp10(self, kps1, sample):
        dev = kps1.device
        img1 = sample['image1']
        img1_path = sample['image1_path']
        mask1_path = img1_path.replace('-rgb.png', '_objmask.png').replace('All_PNG', 'gt_tps')
        mask1 = io.fromPath(mask1_path, batch=False)[0].to(dev)
        valid1 = torch.ones(kps1.shape[0], dtype=torch.bool).to(dev)
        
        # remove keypoints that are in the mask
        for i, kp in enumerate(kps1):
            if mask1[int(kp[1]), int(kp[0])] == 0:
                valid1[i] = False

        loading_file = img1_path.replace('-rgb.png', '').replace('All_PNG', 'gt_tps')
        theta = torch.tensor(np.load(loading_file + '_theta.npy').astype(np.float32), device=dev)
        ctrl_pts = torch.tensor(np.load(loading_file + '_ctrlpts.npy').astype(np.float32), device=dev)

        norm_factor = torch.tensor(np.array(img1.shape[1:3][::-1], dtype = np.float32), device=dev)
        warped_coords = tps_sparse(theta, ctrl_pts, (kps1 / norm_factor)).squeeze(0) * norm_factor

        return warped_coords, valid1


if __name__ == "__main__":
    
    ds = DeSurT(use_cache=False, load_all=False)

    from easy_local_features.feature.baseline_xfeat import XFeat_baseline
    xfeat = XFeat_baseline()
    
    batch = ds.sample_batch(4)
    
    for sample in batch:
        kp0, desc0 = xfeat.detectAndCompute(sample['image0'])
        kp1, desc1 = xfeat.detectAndCompute(sample['image1'])
        kp0 = kp0.squeeze()
        kp1 = kp1.squeeze()
        
        corr = ds.find_correspondences(sample, kp0, kp1)
    

    