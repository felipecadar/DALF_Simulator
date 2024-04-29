SEED = 32000

import argparse
import os
import glob

def parseArg():
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Training mode scheme - two stage (ts) is the default"
    , required=False, choices = ['end2end-backbone', 'end2end-tps', 'end2end-full', 'ts1', 'ts2', 'ts-fl'], default = 'ts1')
    parser.add_argument("-dpath", "--datapath", help="Dataset path."
    , required=False, default = '/work/cadar/Datasets/sfm/*/images/*.jpg') 	
    parser.add_argument("-sdpath", "--sim_datapath", help="Simulation Dataset path."
    , required=False, default = '/work/cadar/Datasets/simulation_v2/train_single_obj/') 	
    parser.add_argument("-log", "--logdir", help="Output path where results will be saved."
    , required=False, default = './log_dir') 
    parser.add_argument("-s", "--save", help="Path for saving model"
    , required=False, default = './log_dir')
    parser.add_argument("--dry_run", help="Sanity check, overfit the training loop without saving anything"
    , action = 'store_true')
    parser.add_argument("-gpu", "--gpu", help="GPU number"
    , required=False, default = "0")
    parser.add_argument("-pre", "--pretrained", help="Pretrained network path for second stage"
    , required=False, default = None) 

    parser.add_argument("-sim", "--simulation", help="Use simulation dataset", action='store_true'
    , required=False, default = False) 

    parser.add_argument("--finetune", help="Use simulation dataset", action='store_true'
    , required=False, default = False) 

    args = parser.parse_args()

    if args.mode == 'ts2' or args.mode=='ts-fl':
        if not os.path.exists(args.pretrained):
            raise RuntimeError('invalid pretrained path -- it is required for second stage;')

    if args.logdir is not None and not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
        # raise RuntimeError(args.logdir + ' does not exist! Please create the logdir')

    if not args.dry_run and args.save is None:
        raise RuntimeError('choose a location to save the models')

    if not args.dry_run and not os.path.exists(args.save):
        raise RuntimeError(args.save + ' does not exist!')

    if not args.simulation:
        if len( glob.glob(args.datapath)) == 0:
            raise RuntimeError(args.datapath + ': no images found')

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu # "0,1" or "0" for example

    if args.mode=='ts2' or args.mode=='ts-fl':
        SEED=64000

    return args

args = parseArg()

print(SEED)
import os
import random
os.environ['PYTHONHASHSEED']=str(SEED)

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import math
import pdb, tqdm

import torchvision.transforms as transforms

import kornia
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from modules.models.DALF import *
from modules.dataset.augmentation import *
from modules.losses import *
from modules.utils import *
from modules.tensorboard_utils import *

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def check_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


def train(args):
    '''
    This function implements custom training loop and different training strategies
    for the DEAL detector & descriptor, alongside the custom losses for joint detection
    and description of deformation-aware keypoints.
    For detailed discussion please refer to the paper.
    All hyperparams defined here were used for the experiments in the paper
    '''

    ######   Prepare model, data and hyperparms #######

    # if not torch.cuda.is_available():
    #     raise RuntimeError('Do you really want to train without a GPU? Then comment out this if')

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    experiment_name = args.mode
    BATCH_SCALE = 2

    if args.mode == 'end2end-backbone' or args.mode == 'ts1':
        batch_size = 12 // BATCH_SCALE
        steps = (80_000 * BATCH_SCALE) if not args.dry_run else 1000
        lr = 1e-3

    else:
        #reduce batch size due to memory constraints 
        batch_size = 4 // BATCH_SCALE
        steps = (95_001 * BATCH_SCALE) if not args.dry_run else 1000
        lr = 2e-4

    if args.mode == 'end2end-backbone':
        backbone_nfeats = 128
    else:
        backbone_nfeats = 64      

    num_grad_accs = 4*BATCH_SCALE # this performs grad accumulation to simulate larger batch size, set to 1 to disable;

    if args.dry_run:
        batch_size = 2


    if args.simulation:
        from modules.dataset.simulation import KubrickInstances
        splits_0 = ['illumination-viewpoint']
        splits_1 = ['deformation_1', 'deformation_1-illumination-viewpoint']
        splits_2 = ['deformation_2', 'deformation_2-illumination-viewpoint']
        splits_3 = ['deformation_3', 'deformation_3-illumination-viewpoint']
        
        simulation_data = KubrickInstances({
            "data_dir": args.sim_datapath,
            "return_tensors": True,
            "splits": splits_0,
        })
        
        triggers = [0.05, 0.1, 0.2]
        splits_progress = [splits_1, splits_2, splits_3]

        print("Simulation dataset loaded. Size: ", len(simulation_data))

    else:
        augmentor = AugmentationPipe(device = dev,  
                                    img_dir = args.datapath,
                                    max_num_imgs = 3_000 if not args.dry_run else 32, #Limit number of images in training, original impl is 5000
                                    num_test_imgs = 100,
                                    out_resolution = (300, 200), #300,200 
                                    batch_size = batch_size
                                    )
    

    logger = TrainLogger(logdir = args.logdir, name = experiment_name)

    # img = augmentor.sample_img

    if args.mode == 'ts2' or args.mode == 'ts-fl':
        
        print('loading pretrained net...')

        extractor = DALF_extractor(args.pretrained)

        if args.mode == 'ts-fl' and not args.finetune:
          print('adding fusion layer...')
          extractor.net.fusion_layer = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.Sigmoid())

        extractor.net.mode = args.mode
        net = extractor.net.to(dev).train()
        
        #Freeze encoder layers
        if not args.finetune:
            for param in net.net.encoder.parameters():
                param.requires_grad = False
            for param in net.net.features.parameters():
                param.requires_grad = False


    else:
        net = DEAL(enc_channels = [1, 32, 64, backbone_nfeats], fixed_tps = False, mode = args.mode).to(dev).train()

    #print(net)
    get_nb_trainable_params(net)

    fp_penalty = 0. #-1e-7 #-0.25
    kp_penalty = -7e-5 #-7e-5 #-0.03
    T = 7. # inv of softmax temperature

    ######   Training Loop   #######
    if args.finetune:
        print('finetuning...')
        lr = 1e-5
        steps = 90_000

    opt = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()) , lr = lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.8)
    dense_matcher = DenseMatcher()
    matcher = Matcher()
    net.train()
    fig = plt.figure(figsize = (8, 6), dpi = 100)

    if args.simulation:
        batch = simulation_data.sample_batch(batch_size)
        p1 = [x['image0'] for x in batch]
        p2 = [x['image1'] for x in batch]

        p1 = torch.stack(p1).to(dev).float()
        p2 = torch.stack(p2).to(dev).float()
    else:
        p1, p2, Hs = make_batch_sfm(augmentor, 0.2)


    opt.zero_grad()
    last_change = 0
    with tqdm.tqdm(total=steps) as pbar:
        for i in range(steps):          
            if i < steps/3:
                alpha = (3*i/steps)

            #Increase augmentation difficulty during training using simple schedule rule
            if i < steps * 0.1:
                difficulty = 0.10
            elif i < steps * 0.2:
                difficulty = 0.15
            elif i < steps * 0.6:
                difficulty = 0.25
            else:
                difficulty = 0.30
            
            for j, t in enumerate(triggers):
                if last_change < t and i > t * steps:
                    if simulation_data.config['splits'] != splits_progress[j]:
                        print(f'updating splits in step {i} to {splits_progress[j]}...')
                        simulation_data.config['splits'] = splits_progress[j]
                        simulation_data.reload_pairs()
                        last_change = t
                           
            #Initialize vars for current step
            #We need to handle batching because the description can have arbitrary number of keypoints
            mean_correct = 0
            dense_correct = 0
            loss = None
            loss_kp = None
            hard_loss = None
            pdesc1 = None
            pdesc2 = None
            pdesc1r = None
            pdesc2r = None
            pdesc1nr = None
            pdesc2nr = None
            ssimloss = None
            l_ssimloss = None
            small_desc1 = None
            small_desc2 = None
            good_matches = torch.tensor([True])
            kpts1, kpts2 = None, None
            acc = []
            
            pos_count = 0
            pos_indexes = [0]        
                
            if not args.dry_run:
                if args.simulation:
                    batch = simulation_data.sample_batch(batch_size)
                    # p1, p2 = batch['img1'], batch['img2']
                    p1 = [x['image0'] for x in batch]
                    p2 = [x['image1'] for x in batch]

                    p1 = torch.stack(p1).to(dev).float()
                    p2 = torch.stack(p2).to(dev).float()

                    # # show batch 
                    # img1 = p1[0].detach().cpu().numpy().transpose(1,2,0)
                    # img2 = p2[0].detach().cpu().numpy().transpose(1,2,0)
                    # img1 = (img1 * 255 ).astype(np.uint8)
                    # img2 = (img2 * 255 ).astype(np.uint8)
                    # cv2.imshow('img1', img1)
                    # cv2.imshow('img2', img2)
                    # cv2.waitKey(1)

                else:
                    p1, p2, Hs = make_batch_sfm(augmentor, difficulty)

            
            kpts1, out1 = net(p1, return_tensors = True)
            kpts2, out2 = net(p2, return_tensors = True)



            for b in range(batch_size):
                #ignore samples that have too few keypoints to avoid singularities
                if kpts1[b]['patches'] is None or kpts2[b]['patches'] is None  \
                or len(kpts1[b]['xy']) < 16 or len(kpts2[b]['xy']) < 16:
                    print('skipping batch item...')
                    continue
                
                if args.simulation:
                    # get mask for keypoints in foreground
                    bgmask0 = batch[b]['bgmask0']
                    bgmask1 = batch[b]['bgmask1']
                    
                    isinside0 = torch.tensor([bgmask0[int(x[1]), int(x[0])] for x in kpts1[b]['xy']]).to(dev) > 0
                    isinside1 = torch.tensor([bgmask1[int(x[1]), int(x[0])] for x in kpts2[b]['xy']]).to(dev) > 0
                    
                    kpts1[b]['logprobs'] = kpts1[b]['logprobs'][isinside0]
                    kpts2[b]['logprobs'] = kpts2[b]['logprobs'][isinside1]
                    
                    kpts1[b]['xy'] = kpts1[b]['xy'][isinside0]
                    kpts2[b]['xy'] = kpts2[b]['xy'][isinside1]
                    
                    kpts1[b]['patches'] = kpts1[b]['patches'][isinside0]
                    kpts2[b]['patches'] = kpts2[b]['patches'][isinside1]
                    
                    if len(kpts1[b]['xy']) < 16 or len(kpts2[b]['xy']) < 16:
                        print('skipping batch item...')
                        continue

                    warp10, _ = simulation_data.warp_torch(batch[b], kpts2[b]['xy'], inverse=True)
                    idx, patches1, patches2 = get_positive_corrs_simulation(kpts1[b], kpts2[b], warp10, i)
                    # image0 = batch[b]['image0']
                    # image1 = batch[b]['image1']
                    # image0 = (image0.detach().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
                    # image1 = (image1.detach().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)

                else:
                    idx, patches1, patches2 = get_positive_corrs(kpts1[b], kpts2[b], Hs[b], augmentor, i)

                # plot keypoints and save to tensorboard
                if i % 100 == 0:
                    print('plotting keypoints...')
                    image0 = p1[b].detach().cpu().numpy().transpose(1,2,0)
                    image1 = p2[b].detach().cpu().numpy().transpose(1,2,0)
                    image0 = (image0 * 255 ).astype(np.uint8)
                    image1 = (image1 * 255 ).astype(np.uint8)

                    kpts1_cv2 = kpts1[b]['xy'].detach().cpu().numpy()
                    kpts2_cv2 = kpts2[b]['xy'].detach().cpu().numpy()
                    kpts1_cv2 = [cv2.KeyPoint(x[0], x[1], 1) for x in kpts1_cv2]
                    kpts2_cv2 = [cv2.KeyPoint(x[0], x[1], 1) for x in kpts2_cv2]

                    positive_cors = idx.detach().cpu().numpy()
                    positive_cors = [cv2.DMatch(x[0], x[1], 1) for x in positive_cors]

                    match_image = cv2.drawMatches(image0, kpts1_cv2, image1, kpts2_cv2, positive_cors, None, flags=0)

                    # kp_image0 = cv2.drawKeypoints(image0, kpts1_cv2, None)
                    # kp_image1 = cv2.drawKeypoints(image1, kpts2_cv2, None)
                    # kp_image = cv2.hconcat([kp_image0, kp_image1])

                    # logger.log_fig(i, kp_image, f'Keypoints/{b}')
                    logger.log_fig(i, match_image, f'Matches/{b}')

                if len(patches1) >=16:
                
                    #only distinct
                    if args.mode == 'end2end-backbone' or args.mode == 'ts1':
                        l_pdesc1 = F.normalize(net.sample_descs(out1['feat'][b], kpts1[b]['xy'][idx[:,0],:], H = p1.shape[2], W = p1.shape[3])) 
                        l_pdesc2 = F.normalize(net.sample_descs(out2['feat'][b], kpts2[b]['xy'][idx[:,1],:], H = p2.shape[2], W = p2.shape[3]))
                        pdesc1 = l_pdesc1 if pdesc1 is None else torch.vstack((pdesc1, l_pdesc1))
                        pdesc2 = l_pdesc2 if pdesc2 is None else torch.vstack((pdesc2, l_pdesc2))
                    else:
                        nrdesc1 = net.hardnet(patches1); nrdesc2 = net.hardnet(patches2)

                        #distinct & invariant
                        if args.mode != 'end2end-tps':
                            rdesc1 = net.sample_descs(out1['feat'][b], kpts1[b]['xy'][idx[:,0],:], H = p1.shape[2], W = p1.shape[3])
                            rdesc2 = net.sample_descs(out2['feat'][b], kpts2[b]['xy'][idx[:,1],:], H = p2.shape[2], W = p2.shape[3])

                            if args.mode == 'ts-fl':
                                l_pdesc1 = torch.cat((nrdesc1, rdesc1), dim=1)
                                l_pdesc2 = torch.cat((nrdesc2, rdesc2), dim=1)
                                l_pdesc1 = F.normalize( net.fusion_layer( l_pdesc1 ) * l_pdesc1)
                                l_pdesc2 = F.normalize( net.fusion_layer( l_pdesc2 ) * l_pdesc2)
                            else:
                                l_pdesc1 = F.normalize( torch.cat((nrdesc1, rdesc1), dim=1) )
                                l_pdesc2 = F.normalize( torch.cat((nrdesc2, rdesc2), dim=1) )

                            pdesc1 = l_pdesc1 if pdesc1 is None else torch.vstack((pdesc1, l_pdesc1))
                            pdesc2 = l_pdesc2 if pdesc2 is None else torch.vstack((pdesc2, l_pdesc2))                                

                            l_pdesc1_nrigid = F.normalize(nrdesc1)
                            l_pdesc2_nrigid = F.normalize(nrdesc2)   

                            pdesc1nr = l_pdesc1_nrigid if pdesc1nr is None else torch.vstack((pdesc1nr, l_pdesc1_nrigid))
                            pdesc2nr = l_pdesc2_nrigid if pdesc2nr is None else torch.vstack((pdesc2nr, l_pdesc2_nrigid))  

                        #full invariant
                        else:
                            l_pdesc1 = F.normalize(nrdesc1)
                            l_pdesc2 = F.normalize(nrdesc2)  
                            pdesc1 = l_pdesc1 if pdesc1 is None else torch.vstack((pdesc1, l_pdesc1))
                            pdesc2 = l_pdesc2 if pdesc2 is None else torch.vstack((pdesc2, l_pdesc2))

      
                    with torch.no_grad():
                        good_matches = torch.argmin(torch.cdist(l_pdesc1, l_pdesc2), dim=1) == torch.arange(len(l_pdesc1),
                                                                                                    device = l_pdesc1.device)
                    acc.append(good_matches.sum().item()/len(good_matches))
                    
                    l_ssimloss = SSIMLoss(patches1, patches2)
                    #l_ssimloss = regularized_SSIM_loss(patches1, patches2)
                
                dense_kp_logprobs = kpts1[b]['logprobs'].view(-1,1) + kpts2[b]['logprobs'].view(1,-1)
                dense_logprobs = dense_kp_logprobs

                if args.simulation:
                    dense_rewards, dense_rwd_sum = get_dense_rewards_simulation(kpts1[b]['xy'], kpts2[b]['xy'], warp10,
                                                                                        penalty = fp_penalty * alpha)
                else:
                    dense_rewards, dense_rwd_sum = get_dense_rewards(kpts1[b]['xy'], kpts2[b]['xy'], Hs[b], augmentor,
                                                                                        penalty = fp_penalty * alpha)
            
                
                if i > 0.75 * steps and not args.mode == 'ts1': #penalyze wrong matches
                    with torch.no_grad():
                        if len(good_matches) == len(idx):
                            idx = idx[~good_matches]
                            dense_rewards[idx[:, 0]] = kp_penalty*10. # 10. 25.
                                            
                dense_correct+= dense_rwd_sum
                pos_count += len(patches1)
                pos_indexes.append(pos_count)
                
                loss_vals = (dense_rewards * dense_logprobs).view(-1)
                
                ssimloss = l_ssimloss if ssimloss is None else torch.hstack((l_ssimloss, ssimloss))
                
                loss =  loss_vals if loss is None else torch.hstack((loss, loss_vals))
                current_loss_kp = (kpts1[b]['logprobs'] * torch.full_like(kpts1[b]['logprobs'], kp_penalty*alpha)).mean() +     \
                                (kpts2[b]['logprobs'] * torch.full_like(kpts2[b]['logprobs'], kp_penalty*alpha)).mean()
                loss_kp = current_loss_kp if loss_kp is None else  torch.hstack((loss_kp, current_loss_kp))
                det_kpts1 = len(kpts1[b]['xy'])
                det_kpts2 = len(kpts2[b]['xy'])

            #Plot every x steps
            if len(patches1) >=16 and i % 200 == 0:
                plt.draw() ;  #plt.show()
                np_fig = grab_mpl_fig(fig)
                logger.log_fig(i, np_fig, 'Gradient Flows')
                fig = plot_grid( (patches1[:16], patches2[:16]) )
                plt.draw(); np_fig = grab_mpl_fig(fig)
                logger.log_fig(i, np_fig, 'Warped Patches')

                fig = plt.figure(figsize = (8, 6), dpi = 100)
                print('difficulty %.3f'%(difficulty))
           
            if loss is None or loss_kp is None:
                print("Loss is None")
                print("Skipped all samples in batch")
                continue

            #loss = -loss.mean() #average across batch
            loss = -(loss.mean() + loss_kp.mean())
            #hard_loss = hard_loss.mean()
            
            hard_loss = hardnet_loss(pdesc1, pdesc2) if pdesc1 is not None else None
            hard_loss_nrigid = hardnet_loss(pdesc1nr, pdesc2nr) if pdesc1nr is not None else None

           
            if hard_loss is not None and hard_loss_nrigid is not None:
                hard_loss += hard_loss_nrigid
                hard_loss /= 2.
            elif hard_loss_nrigid is not None:
                hard_loss = hard_loss_nrigid
                
            if hard_loss is not None:
                loss += 0.005 * hard_loss

            #if ssimloss is not None:
            #    loss += 0.05 * ssimloss.mean()
                
            pbar.set_description('L: {:.4f} - Det: ({:d}, {:d}), #rwd: {:.0f}/{:d}, #dRwd: {:.1f} #HL: {:.3f} ssimL: {:.3f}'.format( loss.item(), 
                                    det_kpts1, det_kpts2, good_matches.sum(), len(good_matches), dense_correct/batch_size,
                                    hard_loss.item() if hard_loss is not None else 0.,
                                    ssimloss.mean().item()*2. if ssimloss is not None else 0.))
            pbar.update(1)

            #backward pass
            loss /= num_grad_accs
            loss.backward()
        
            if i%10 == 0:
                plot_grad_flow(net.named_parameters())
                #[print(i) for i in net.named_parameters()]

            if i%num_grad_accs == 0:
                opt.step()
                opt.zero_grad()
                
            logger.log_scalars(i, avg_det = (det_kpts1 + det_kpts2)/2., 
                            acc = np.array(acc).mean() if len(acc) > 0 else 0.,
                            inliers = good_matches.sum(),
                            kp_rewards = dense_correct/batch_size,
                            hard_loss = hard_loss.item() if i > 150 and hard_loss is not None else 0.,
                            ssim_loss = ssimloss.mean().item()*2. if i > 150 and ssimloss is not None else 0.)
            

            if i%5000 == 0:
                if not args.dry_run:
                    torch.save(net.state_dict(), args.save + '/model_' + args.mode + '_%06d'%i + '.pth')
        #     scheduler.step()

    #save the model
    #if not args.dry_run:
    torch.save(net.state_dict(), args.save + '/model_' + args.mode + '_' + str(i+1) + '_final' + '.pth')

if __name__ == '__main__':
    train(args)
    # extractor = DEAL_extractor(args.pretrained)
    # kps, descs = extractor.detectAndCompute('/home/guipotje/Sources/nonrigid-keypoint-detector-descriptor/assets/notredame.png',
    #                                       threshold = 5.0)
    # print(len(kps), ' ', descs.shape)
