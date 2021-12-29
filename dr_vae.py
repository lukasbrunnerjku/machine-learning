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


class Encoder(nn.Module):
    
    def __init__(self, enc_out_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32*32, 16*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16*16, enc_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_var = nn.Linear(enc_out_dim, latent_dim)
    
    def forward(self, x):  # H=W=64
        B = x.shape[0]  # x is Bx1xHxW
        x = F.interpolate(x, size=(32, 32))  # H=W=32
        x = x.view(B, -1)  # Bx(H*W)
        
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_log_var(x_encoded)
        std = torch.exp(log_var / 2)
        
        return mu, std
    
    
class Decoder(nn.Module):
    
    def __init__(self, dec_out_dim: int = 256, latent_dim: int = 64, out_dim: int = 2):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, dec_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(dec_out_dim, out_dim)
        self.fc_log_var = nn.Linear(dec_out_dim, out_dim)
        
    def forward(self, z):
        
        x_decoded = self.decoder(z)
        mu, log_var = self.fc_mu(x_decoded), self.fc_log_var(x_decoded)
        std = torch.exp(log_var / 2)
        
        return mu, std


def gaussian_likelihood(x_source, log_std, x_target):
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(x_source, std)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x_target)
    return log_pxz.sum(dim=(1, 2, 3))
    

def kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl


def summary(net: nn.Module):
    nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{net.__class__.__name__} parameters: {nparams/1e6:.3f}M')
    

class Renderer:
    
    def __init__(self) -> None:
        self.canvas = np.ones((64, 64), dtype=np.float32)
        self.obj_wh = np.array([3, 3], dtype=np.float32)
        
    def render(self, s: np.ndarray) -> np.ndarray:
        if s.ndim == 2:
            B = s.shape[0]
        else:
            B = 1
        
        x = np.empty((B, 1, *self.canvas.shape), dtype=np.float32)
        for b in range(B):
            img = self.canvas.copy()
            tl: Tuple[int] = tuple(map(int, s[b] - self.obj_wh))  
            br: Tuple[int] = tuple(map(int, s[b] + self.obj_wh))
            color: Tuple[int] = (0, 0, 0)
            cv.rectangle(img, tl, br, color, -1)
            x[b, 0] = img
            
        return x
        

if __name__ == '__main__':
    # https://arxiv.org/pdf/1906.02691.pdf
    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    
    renderer = Renderer()
    encoder = Encoder()
    decoder = Decoder()
    
    torch.optim.Adam(encoder.parameters(), lr=1e-4)
    
    # for the gaussian likelihood
    log_scale = nn.Parameter(torch.Tensor([0.0]))
    
    # define target gaussian
    z_mu_target = Parameter(torch.tensor([45.0, 49.0]))
    z_std_target = Parameter(torch.tensor([9.0, 7.0]))

    
    B = 32
    alpha=0.9
    u_ema = 0.0
    num_steps = 100
    
    mus, stds = [], []
    for global_step in tqdm(range(num_steps)):
        
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
        
  
        
        optG.zero_grad(set_to_none=True)
        log_probs = log_prob(sample)  # Bx2, calculate joint distribution p(z1,z2,..zN) <=> p(x)
        # for independent simulation parameters we can sum the log_prob values
        # to get the joint distribution!
        
        with torch.no_grad():  # calculate utility U(x)
            u = criterion(D(x_source), ones)  # Bx1x1x1
            u = u[:, :, 0, 0]  # Bx1
        
        lossG = (log_probs * (u - u_ema)).mean()
        lossG.backward()
        
        optG.step()
       
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
    