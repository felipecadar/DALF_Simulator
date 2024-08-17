from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from kornia.geometry.transform import warp_points_tps, warp_image_tps
import io
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

grad_fig = None


def grab_mpl_fig(fig):
    '''
        Transform current drawn fig into a np array
    '''
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=100)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr
    #plt.imshow(img_arr) ; plt.show(); input()

def plot_grid(warped, title = 'Grid Vis', mpl = True):
    #visualize 
    g = None
    n = warped[0].shape[0]

    for i in range(0, n, 16):
        if i + 16 <= n:
            for w in warped:
                pad_val = 0.7 if i//16%2 == 1 else 0
                gw = make_grid(w[i:i+16].detach().clone().cpu(), padding=4, pad_value=pad_val, nrow=16)
                g = gw if g is None else torch.cat((g, gw), 1)

    if mpl:
        fig = plt.figure(figsize = (12, 3), dpi=100)
        plt.imshow(np.clip(g.permute(1,2,0).numpy()[...,::-1], 0, 1))
        return fig


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=60, ha = 'right')
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()


def make_batch_sfm(augmentor, difficulty = 0.3, train = True):
    Hs = []
    img_list = augmentor.train if train else augmentor.test
    dev = augmentor.device
    batch_images = []

    with torch.no_grad(): # we dont require grads in the augmentation
        for b in range(augmentor.batch_size):
            rdidx = np.random.randint(len(img_list))
            img = torch.tensor(img_list[rdidx], dtype=torch.float32).permute(2,0,1).to(augmentor.device).unsqueeze(0)
            batch_images.append(img)

        batch_images = torch.cat(batch_images)

        p1, H1 = augmentor(batch_images, difficulty)
        p2, H2 = augmentor(batch_images, difficulty, TPS = True, prob_deformation = 0.7)

        H2, src, W, A = H2

        for b in range(augmentor.batch_size):
            Hs.append((H2[b]@torch.inverse(H1[b]).to(dev), 
                        src[b].unsqueeze(0),
                        W[b].unsqueeze(0),
                        A[b].unsqueeze(0)))

    return p1, p2, Hs



def get_reward(kps1, kps2, H, augmentor, penalty = 0., px_thr = 2):
    with torch.no_grad():
        #perspective transform 2 -> 1
        if not augmentor.TPS:
            warped = augmentor.warp_points(torch.inverse(H), kps2)
        else:
        #first undeform TPS, then perspective 2 -> 1
            H, src, W, A = H
            undeformed  = augmentor.denorm_pts_grid(   \
                                          warp_points_tps(augmentor.norm_pts_grid(kps2),
                                          src, W, A) ).view(-1,2)
            warped = augmentor.warp_points(torch.inverse(H), undeformed)
            
        error = torch.linalg.norm(warped - kps1, dim = 1)
        rewards = (error <= px_thr).float()
        reward_sum = torch.sum(rewards)
        rewards[rewards == 0.] = penalty
    return rewards, reward_sum

def get_dense_rewards(kps1, kps2, H, augmentor, penalty = 0., px_thr = 1.5):
    with torch.no_grad():
        #perspective transform 2 -> 1
        if not augmentor.TPS:
            warped = augmentor.warp_points(torch.inverse(H), kps2)
        else:
        #first undeform TPS, then perspective 2 -> 1
            H, src, W, A = H
            undeformed = augmentor.denorm_pts_grid(   \
                                          warp_points_tps(augmentor.norm_pts_grid(kps2),
                                          src, W, A) ).view(-1,2)
            warped = augmentor.warp_points(torch.inverse(H), undeformed)
            
        d_mat = torch.cdist(kps1, warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins)).long()

        d_mat[y_mins, x_mins] *= -1.
        d_mat[d_mat >= 0.] = 0.
        d_mat[d_mat < -px_thr] = 0.
        d_mat[d_mat != 0.] = 1.

        reward_mat = d_mat
        reward_sum = reward_mat.sum() 
        reward_mat[reward_mat == 0.] = penalty
        # import pdb; pdb.set_trace()
    return reward_mat, reward_sum
    
# from .dataset.desurt import DeSurT
def get_dense_rewards_desurt(kps1, kps2, sample, dataset, penalty = 0., px_thr = 1.5):
    with torch.no_grad():
        warped, valid = dataset.warp10(kps2, sample)
        warped[~valid] = -1.

        d_mat = torch.cdist(kps1, warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins)).long()

        d_mat[y_mins, x_mins] *= -1.
        d_mat[d_mat >= 0.] = 0.
        d_mat[d_mat < -px_thr] = 0.
        d_mat[d_mat != 0.] = 1.

        reward_mat = d_mat
        reward_sum = reward_mat.sum() 
        reward_mat[reward_mat == 0.] = penalty
        # import pdb; pdb.set_trace()
    return reward_mat, reward_sum
    
    
def get_positive_corrs(kps1, kps2, H, augmentor, i=0, px_thr = 1.5):
    with torch.no_grad():
        #perspective transform 2 -> 1
        if not augmentor.TPS:
            warped = augmentor.warp_points(torch.inverse(H), kps2['xy'])
        else:
        #first undeform TPS, then perspective 2 -> 1
            H, src, W, A = H
            undeformed = augmentor.denorm_pts_grid(   \
                                          warp_points_tps(augmentor.norm_pts_grid(kps2['xy']),
                                          src, W, A) ).view(-1,2)
            warped = augmentor.warp_points(torch.inverse(H), undeformed)
              
        d_mat = torch.cdist(kps1['xy'], warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins), device= d_mat.device).long()

        #grab indices of positive correspodences & filter too close kps in the same image
        y_mins = y_mins[(x_vmins < px_thr)] #* (self_vmins > 2.)]
        x_mins = x_mins[(x_vmins < px_thr)] #* (self_vmins > 2.)]

    return torch.hstack((y_mins.unsqueeze(1),x_mins.unsqueeze(1))),  \
           kps1['patches'][y_mins], kps2['patches'][x_mins]
  

    
def get_positive_corrs_simulation(kps1, kps2, warped, i=0, px_thr = 1.5):
    with torch.no_grad():
        if len(warped) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        d_mat = torch.cdist(kps1['xy'], warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins), device= d_mat.device).long()

        #grab indices of positive correspodences & filter too close kps in the same image
        y_mins = y_mins[(x_vmins < px_thr)] #* (self_vmins > 2.)]
        x_mins = x_mins[(x_vmins < px_thr)] #* (self_vmins > 2.)]

        # return idx1 idx2 and patches1 patches2

    return torch.hstack((y_mins.unsqueeze(1),x_mins.unsqueeze(1))),  \
              kps1['patches'][y_mins], kps2['patches'][x_mins]


def get_dense_rewards_simulation(kps1, kps2, warped, penalty = 0., px_thr = 1.5):
    with torch.no_grad():   

        d_mat = torch.cdist(kps1, warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins)).long()

        d_mat[y_mins, x_mins] *= -1.
        d_mat[d_mat >= 0.] = 0.
        d_mat[d_mat < -px_thr] = 0.
        d_mat[d_mat != 0.] = 1.

        # not_valid = warped[:,0] < 0.
        # d_mat[not_valid] = 0.

        reward_mat = d_mat
        reward_sum = reward_mat.sum() 
        reward_mat[reward_mat == 0.] = penalty
    return reward_mat, reward_sum



default_colors = {
    'g': '#4ade80',
    'r': '#ef4444',
    'b': '#3b82f6',
}

def load_image(path):
    '''Loads an image from a file path and returns it as a torch tensor
    Output shape: (3, H, W) float32 tensor with values in the range [0, 1]
    '''
    image = torchvision.io.read_image(str(path)).float() / 255
    return image

def load_depth(path):
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    image = torch.tensor(image).unsqueeze(0)
    return image

def to_cv(torch_image, convert_color=True, batch_idx=0, to_gray=False):
    '''Converts a torch tensor image to a numpy array'''
    if isinstance(torch_image, torch.Tensor):
        if len(torch_image.shape) == 2:
            torch_image = torch_image.unsqueeze(0)
        if len(torch_image.shape) == 4 and torch_image.shape[0] == 1:
            torch_image = torch_image[0]
        if len(torch_image.shape) == 4 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx]
        if len(torch_image.shape) == 3 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx].unsqueeze(0)
            
        if torch_image.max() > 1:
            torch_image = torch_image / torch_image.max()
        
        img = (torch_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    else:
        img = torch_image

    if convert_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def plot_pair(img0, img1, figsize=(20, 10), fig=None, ax=None, title=None):
    if fig is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(to_cv(img0))
    ax[1].imshow(to_cv(img1))
    
    # remove border
    for a in ax:
        a.axis('off')
    # remove space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if title is not None:
        fig.suptitle(title)
    return fig, ax

def plot_overlay(img0, img1, figsize=(20, 10), fig=None, ax=None, title=None):
    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes

    ax[0].imshow(to_cv(img0), alpha=0.5)
    ax[1].imshow(to_cv(img1), alpha=0.5)
    
    if title is not None:
        fig.suptitle(title)
    return fig, ax

def plot_depth(img0, img1, figsize=(20, 10), fig=None, ax=None, title=None):
    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes

    ax[0].imshow(to_cv(img0, to_gray=True), cmap='jet', alpha=0.5)
    ax[1].imshow(to_cv(img1, to_gray=True), cmap='jet', alpha=0.5)
    
    if title is not None:
        fig.suptitle(title)
    return fig, ax

def plot_keypoints(keypoints0=None, keypoints1=None, fig=None, ax=None, color=None):
    rainbow = plt.get_cmap('hsv')
    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes
    
    if keypoints0 is not None:        
        if isinstance(keypoints0, torch.Tensor):
            keypoints0 = keypoints0.detach().cpu().numpy()

        if len(keypoints0.shape) == 3:
            keypoints0 = keypoints0.squeeze(0)

        if isinstance(color, str) and color in default_colors:
            all_colors = [default_colors[color]] * len(keypoints0)
        else:
            all_colors = [rainbow(i / len(keypoints0)) for i in range(len(keypoints0))]

        ax[0].scatter(keypoints0[:, 0], keypoints0[:, 1], s=5, c=all_colors)

    if keypoints1 is not None:
        if isinstance(keypoints1, torch.Tensor):
            keypoints1 = keypoints1.detach().cpu().numpy()

        if len(keypoints1.shape) == 3:
            keypoints1 = keypoints1.squeeze(0)

        if isinstance(color, str) and color in default_colors:
            all_colors = [default_colors[color]] * len(keypoints1)
        else:
            all_colors = [rainbow(i / len(keypoints1)) for i in range(len(keypoints1))]
            
        ax[1].scatter(keypoints1[:, 0], keypoints1[:, 1], s=5, c=all_colors)
    
    return fig, ax

def plot_matches(mkpts0, mkpts1, fig=None, ax=None, color='b'):
    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes
    
    if color in default_colors:
        color = default_colors[color]
    
    for i, (mkp0, mkp1) in enumerate(zip(mkpts0, mkpts1)):
        con = ConnectionPatch(
            xyA=mkp0, xyB=mkp1, 
            coordsA="data", coordsB="data",
            axesA=ax[0], axesB=ax[1], 
            color=color, linewidth=0.5)
        con.set_in_layout(False) # remove from layout calculations
        ax[0].add_artist(con)
    ax[1].set_zorder(-1)
    
    return fig, ax

def show():
    plt.show()