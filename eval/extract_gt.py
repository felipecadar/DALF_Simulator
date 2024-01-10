import os
import sys

modules = os.path.dirname(os.path.realpath(__file__)) + '/..'         
sys.path.insert(0, modules)

import torch
import cv2
import glob
import tqdm
import argparse
import numpy as np
import pdb
from scipy.spatial import KDTree

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

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def write_sift(filepath, kps):
    with open(filepath + '.sift', 'w') as f:
        f.write('size, angle, x, y, octave\n')
        for kp in kps:
            f.write('%.2f, %.3f, %.2f, %.2f, %d\n'%(kp.size, kp.angle, kp.pt[0], kp.pt[1], kp.octave))

def write_matches(filepath, idx_ref, idx_tgt):

    with open(filepath + '.match', 'w') as f:
        f.write('idx_ref,idx_tgt\n')
        for i in range(len(idx_ref)):
            f.write('%d, %d\n'%(idx_ref[i], idx_tgt[i]))

def draw_cv_matches(src_img, tgt_img, src_kps, tgt_kps, gt_ref, gt_tgt):
    dmatches = [cv2.DMatch(gt_ref[i], gt_tgt[i], 0.) for i in np.arange(len(gt_ref))]
    img = cv2.drawMatches(src_img, src_kps ,tgt_img, tgt_kps, dmatches, None, flags = 0)
    return img


def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input directory for one or more datasets (use --dir for several)"
    , required=True) 
    parser.add_argument("--tps_dir", help="Input directory containing the optimized TPS params for one or more datasets (use --dir for several)"
    , required=True) 
    parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
    , action = 'store_true')
    parser.add_argument("--cpu", help="Force CPU mode"
    , action = 'store_true')
    parser.add_argument("-m", "--method", help="Method used to extract keypoints"
    , required=False, choices = ['sift', 'r2d2', 'aslfeat', 'pgnet', 'pgdeal', 'tfeat'], default = 'sift')
    parser.add_argument("-np", "--net_path", help="pretrained weights path for model if applicable"
    , required=False, default = '') 
    args = parser.parse_args()

    args = parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('GPU not available, try running with --cpu')
    
    if not args.cpu:
        print('Using GPU')
        force_cudnn_initialization()

    return args

args = parseArg()
args.input = os.path.abspath(args.input) 

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.method == 'pgdeal' or args.method == 'sift':
    try:
        from modules.models.DALF import DALF_extractor as PGNet
    except:
        raise ImportError('Unable to import DALF ')  

if args.dir:
    datasets = [d for d in glob.glob(args.input+'/*/*') if os.path.isdir(d)]
else:
    datasets = [args.input]

tps_path = args.tps_dir
datasets = list(filter(lambda x: 'DeSurTSampled' in x or  'Kinect1' in x or 'Kinect2Sampled' in x or 'SimulationICCV' in x, datasets))
if args.method == 'r2d2':
    SIFT = R2D2()
elif args.method == 'tfeat':
    from easy_local_features.feature.baseline_tfeat import TFeat_baseline
    detector = cv2.SIFT_create(nfeatures = 2048, contrastThreshold=0.004)
    descritor = TFeat_baseline(device=0)
    
    # Finetuned
    descritor.model.load_state_dict(torch.load('/home/cadar/Documents/Github/tfeat/best-tfeat-simulation.pth', map_location=torch.device('cuda')))
    
    # Trained from scratch
    # descritor.model.load_state_dict(torch.load('/home/cadar/Documents/Github/tfeat/best-tfeat-finetune-simulation.pth', map_location=torch.device('cuda')))

elif args.method == 'aslfeat':
    SIFT = ASLFeat()
elif args.method == 'pgnet':
    SIFT = PGNet(dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
elif args.method == 'pgdeal':
    SIFT =  PGNet(model = args.net_path, dev = dev, fixed_tps=False)
else:
    SIFT = cv2.SIFT_create(nfeatures = 2048, contrastThreshold=0.004)
    pgnet =  PGNet(model = args.net_path, dev = dev, fixed_tps=False)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset in datasets:
    if len(glob.glob(dataset + '/*.csv')) == 0: raise RuntimeError('Empty dataset with no .csv file')

    targets = [os.path.splitext(t)[0] for t in glob.glob(dataset + '/*[0-9].csv')]
    master = os.path.splitext(glob.glob(dataset + '/*master.csv')[0])[0]
    if True:#args.method != 'sift':
        ref_img = cv2.imread(master + '-rgb.png')
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    else:
        ref_img = cv2.imread(master + '-rgb.png',0)

    loading_path = os.path.join(tps_path, *dataset.split('/')[-2:])

    if not os.path.exists(loading_path):
        raise RuntimeError('There is no TPS directory in ' + loading_path)

    ref_mask = cv2.imread(loading_path + '/' + os.path.basename(master) + '_objmask.png', 0)

    ref_descs = None

    if args.method == 'pgnet' or args.method == 'pgdeal':
        ref_kps, ref_descs = SIFT.detectAndCompute(ref_img, None)
    elif args.method == 'tfeat':
        ref_kps = detector.detect(ref_img, None)
        ref_descs, _ = descritor.compute(ref_img, ref_kps)
    else:
        ref_kps = SIFT.detect(ref_img, None)
        pgnet_kps, _ = pgnet.detectAndCompute(ref_img, None) 
        maxKps = len(pgnet_kps)
        ref_kps = sorted(ref_kps, key = lambda x: x.response, reverse = True)[:maxKps]


    # print('Detected ref kps: ', len(ref_kps))
    for i, k in enumerate(ref_kps): k.class_id = i
    ref_kps = [kp for kp in ref_kps if ref_mask[int(kp.pt[1]), int(kp.pt[0])] > 0] #filter by object mask
    ref_idx = [k.class_id for k in ref_kps]
    if ref_descs is not None:
        ref_descs = ref_descs[ref_idx]
        np.save(loading_path + '/' + os.path.basename(master)+'.pgnet', ref_descs)

    write_sift(loading_path + '/' + os.path.basename(master), ref_kps)


    for target in tqdm.tqdm(targets, desc = 'image pairs'):
        loading_file = loading_path + '/' + os.path.basename(target)
        theta_np = np.load(loading_file + '_theta.npy').astype(np.float32)
        ctrl_pts = np.load(loading_file + '_ctrlpts.npy').astype(np.float32)
        score = cv2.imread(loading_file + '_SSIM.png', 0) / 255.0
        tgt_mask = cv2.imread(loading_file + '_objmask.png', 0)
        if True:#args.method != 'sift':
            target_img = cv2.imread(target + '-rgb.png')
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)  
        else:
            target_img = cv2.imread(target + '-rgb.png',0)      

        score_mask = score > 0.25
        target_descs = None

        if args.method == 'pgnet' or args.method == 'pgdeal':
            target_kps, target_descs = SIFT.detectAndCompute(target_img, None)
        elif args.method == 'tfeat':
            target_kps = detector.detect(target_img, None)
            target_descs, _ = descritor.compute(target_img, target_kps)
        else: 
            target_kps = SIFT.detect(target_img, None)
            pgnet_kps, _ = pgnet.detectAndCompute(target_img, None) 
            maxKps = len(pgnet_kps)
            target_kps = sorted(target_kps, key = lambda x: x.response, reverse = True)[:maxKps]

        # print('Detected target kps: ', len(target_kps))
        # if target == '/srv/storage/datasets/nonrigiddataset/IJCV2020/All_PNG/SimulationICCV/kanagawa_rot/cloud_17':
        #     pdb.set_trace()

        for i, k in enumerate(target_kps): k.class_id = i
        target_kps = [kp for kp in target_kps if tgt_mask[int(kp.pt[1]), int(kp.pt[0])] > 0] #filter by object mask
        target_kps = [kp for kp in target_kps if score_mask[int(kp.pt[1]), int(kp.pt[0])] == True] #filter by score map with very low confidences
        target_idx = [k.class_id for k in target_kps]
        if target_descs is not None:
            target_descs = target_descs[target_idx]
            np.save(loading_file + '.pgnet', target_descs)

        if len(target_kps) == 0:
            print('No keypoints detected in ', target)
            write_sift(loading_file, target_kps)
            write_matches(loading_file, [], [])
            continue

        norm_factor = np.array(target_img.shape[:2][::-1], dtype = np.float32)
        theta = torch.tensor(theta_np, device= device)
        tgt_coords = np.array([kp.pt for kp in target_kps], dtype = np.float32) 
        warped_coords = tps_sparse(theta, torch.tensor(ctrl_pts, device=device), torch.tensor(tgt_coords / norm_factor, 
                                                                        device=device)).squeeze(0).cpu().numpy() * norm_factor
        tree = KDTree([kp.pt for kp in ref_kps])
        dists, idxs_ref = tree.query(warped_coords)
        px_thresh = 3.0
        gt_tgt  = np.arange(len(target_kps))[ dists < px_thresh] # Groundtruth indexes -- threshold is in pixels 
        gt_ref = idxs_ref[dists < px_thresh] 

        #filter repeated matches
        _, uidxs = np.unique(gt_ref, return_index = True)
        gt_ref = gt_ref[uidxs]
        gt_tgt = gt_tgt[uidxs]

        img_match = draw_cv_matches(ref_img, target_img, ref_kps, target_kps, gt_ref, gt_tgt)
        # cv2.imwrite( + os.path.basename(target) + '_match.png', img_match)
        cv2.imwrite(loading_file + '_match.png', img_match)

        write_sift(loading_file, target_kps)
        write_matches(loading_file, gt_ref, gt_tgt)

