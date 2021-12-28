import cv2 as cv
import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
import numpy as np
from typing import Tuple
from tqdm import tqdm
import math
import pdb


# class Discriminator(nn.Sequential):
    
#     def __init__(
#             self, 
#             nc: int = 1,
#             ndf: int = 16,
#             nlayers: int = 4, 
#             bias: bool = True,
#             batchnorm: bool = False,
#         ):
#         submodules = []
#         cin = nc
#         cout = ndf
        
#         for _ in range(nlayers):
            
#             if batchnorm:
#                 submodules.extend([
#                     nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
#                     nn.BatchNorm2d(cout),
#                     nn.LeakyReLU(0.2, inplace=True),
#                 ])
#             else:
#                 submodules.extend([
#                     nn.Conv2d(cin, cout, 4, 2, 1, bias=bias),
#                     nn.LeakyReLU(0.2, inplace=True),
#                 ])
            
#             cin = cout
#             cout = 2 * cin
        
#         submodules.append(nn.Conv2d(cin, 1, 1, 1))
                    
#         super().__init__(
#             *submodules[:-1],
#             nn.AdaptiveAvgPool2d((1, 1)),
#             submodules[-1]
#         )  # => returns logits

class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.submodules = nn.Sequential(
            nn.Linear(32*32, 16*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16*16, 8*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8*8, 1),  # Bx1
        )
    
    def forward(self, x):  # H=W=64
        B = x.shape[0]  # x is Bx1xHxW
        x = F.interpolate(x, size=(32, 32))  # H=W=32
        x = x.view(B, -1)  # Bx(H*W)
        x = self.submodules(x)  # Bx1
        x = x[:, :, None, None]  # Bx1x1x1; to be consistent with the convolutional discriminator
        return x
    
    
def weights_init(m):
    # model.apply(weights_init)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

if __name__ == '__main__':
    
    # https://www.youtube.com/playlist?list=PL93aLKqThq4h7UpgeNhkOtEeCnX3DMseS
    
    canvas = np.ones((64, 64), dtype=np.float32)
    
    # z <=> center (x,y) position of rectangle
    
    # define source gaussian
    z_mu_source = Parameter(torch.tensor([10.0, 20.0]))  # 2,
    z_log_std_source = Parameter(torch.log(torch.tensor([5.0, 6.0])))  # 2,
    # std = exp( log_std ) thus log_std can be subject to unconstraint optimization
    
    def log_prob(z: Tensor):  # for source normal distribution
        if sample.ndim != 2:
            raise AttributeError
        
        N = sample.shape[0]
        lp = ( -N*z_log_std_source - N/2*math.log(2*math.pi) - 
              ((z - z_mu_source[None, :])**2).sum(0)/(2*torch.exp(z_log_std_source)**2) )
        return lp  # 2,
    
    optG = optim.Adam((z_mu_source, z_log_std_source), lr=0.05, betas=(0.5, 0.999))
    
    # define target gaussian
    z_mu_target = Parameter(torch.tensor([45.0, 49.0]))
    z_std_target = Parameter(torch.tensor([9.0, 7.0]))

    obj_wh = np.array([3, 3], dtype=np.float32)  # 1x2
    
    # update distribution parameters with reparametrization trick
    unorm = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))  # 2,
    
    D = Discriminator()
    D.apply(weights_init)
    nparams = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f'Discriminator parameters: {nparams/1e3:.3f}k')
    
    optD = optim.Adam(D.parameters(), lr=0.005, betas=(0.7, 0.999))
    
    def render(z: np.ndarray) -> np.ndarray:
        if z.ndim == 2:
            B = z.shape[0]
        else:
            B = 1
        
        x = np.empty((B, 1, *canvas.shape), dtype=np.float32)
        for b in range(B):
            img = canvas.copy()
            tl: Tuple[int] = tuple(map(int, z[b] - obj_wh))  
            br: Tuple[int] = tuple(map(int, z[b] + obj_wh))
            color: Tuple[int] = (0, 0, 0)
            cv.rectangle(img, tl, br, color, -1)
            x[b, 0] = img
            
        return x
    
    # typically G(z) where z ~ N(0, 1) but now
    # G(z) := render(z1, z2,...) where z_i some random simulation variable ~ N(mu_i, sig_i)
    G = render
    
    B = 32
    first = True
    alpha = 0.9
    u_ema = 0.0
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    zeros = torch.zeros((B,1,1,1), dtype=torch.float32)
    ones = torch.ones((B,1,1,1), dtype=torch.float32)
         
    num_steps = 100
    warmup_D_steps = 80
    def draw_gaussian(img, mu, std, color):
        mu: Tuple[int] = tuple(map(int, mu))  # 2,
        std: Tuple[int] = tuple(map(int, std))  # 2,
        cv.circle(img, mu, 3, color, thickness=-1)
        cv.ellipse(img, mu, std, 0, 0, 360, color)
        return img
    
    mus, stds = [], []
    for global_step in tqdm(range(num_steps + warmup_D_steps)):
        
        # sample from the source distribution
        with torch.no_grad():
            z = z_mu_source + torch.exp(z_log_std_source) * unorm.sample((B,))  # Bx2
            sample = z
            x_source = torch.from_numpy(G(z.detach().numpy()))  # Bx1xHxW
            
            # sample from the target distribution
            z = z_mu_target + z_std_target * unorm.sample((B,))  # Bx2
            x_target = torch.from_numpy(G(z.detach().numpy()))  # Bx1xHxW
        
        # standard GAN loss for the discriminator
        optD.zero_grad(set_to_none=True)
        lossD =  criterion(D(x_target), ones).mean() + criterion(D(x_source.detach()), zeros).mean()
        lossD.backward()
        optD.step()
        
        # skip "generator" training step, thus not update the simulation parameters
        # at the beginning to let the discriminator produce more usefull gradients
        if global_step < warmup_D_steps: 
            continue
        
        if first:
            # test how good the descriminator is
            with torch.no_grad():
                nll_source = -torch.log(torch.sigmoid(D(x_source))).sum()  # should be 
                nll_target = -torch.log(torch.sigmoid(D(x_target))).sum()  # should be 
                print(f'nll source after {warmup_D_steps} steps of warmup:', nll_source)
                print(f'nll target after {warmup_D_steps} steps of warmup:', nll_target)
                
        # the non-saturating GAN loss would be
        # lossG = criterion(D(x_source), ones).mean()
        # but we cannot backpropagate through the non-differentiable renderer!
        # =>
        
        optG.zero_grad(set_to_none=True)
        # here we have a single parameter to optimize thus we have a single distribution
        # where we have sampled simulation parameters from, generally the probability p(x)
        # that a rendered image x is from the simulation distribution p is the joint probability
        # p(x) = p(z1, z2,...zN) (x = R(z1, z2,... zN)) for the N samples of simulation 
        # parameters drawn (since the rendering function R is
        # itself a deterministic function, thus which images are rendered only depends on
        # the simulation parameter distributions)
        log_probs = log_prob(sample)  # Bx2, calculate joint distribution p(z1,z2,..zN) <=> p(x)
        # for independent simulation parameters we can sum the log_prob values
        # to get the joint distribution!
        
        with torch.no_grad():  # calculate utility U(x)
            u = criterion(D(x_source), ones)  # Bx1x1x1
            u = u[:, :, 0, 0]  # Bx1
        
        lossG = (log_probs * (u - u_ema)).mean()
        lossG.backward()
        
        # print(z_mu_source.grad)
        # pdb.set_trace()
        
        optG.step()
        
        # # track training trajectory in weight space
        # mus.append(z_mu_source.detach().numpy())
        # stds.append(torch.exp(z_log_std_source).detach().numpy())
        
        if first:  # reduce variance of estimator
            u_ema = u.mean(0, keepdims=True)
            first = False
        else:
            u_ema = alpha * u.mean(0, keepdims=True) + (1 - alpha) * u_ema
            
    
    # img = cv.cvtColor(canvas.copy(), cv.COLOR_GRAY2RGBA)  # HxWx4
    # for i, (mu, std) in enumerate(zip(mus, stds)):
    #     alpha = 0.4 + i/len(mus) * 0.6
    #     color = (1, 0, 0, alpha)
    #     img = draw_gaussian(img, mu, std, color)
    
    # mu = z_mu_target.detach().numpy()  # 2,
    # std = z_std_target.detach().numpy()  # 2,
    # img = draw_gaussian(img, mu, std, (0, 1, 0, 1))
    
    # cv.imshow('', img)
    # cv.waitKey(0) 
    # cv.destroyAllWindows()
    
    print('mu:', z_mu_source.data, 'goal:', z_mu_target.data)
    print('std:', torch.exp(z_log_std_source.data), 'goal:', z_std_target.data)
    