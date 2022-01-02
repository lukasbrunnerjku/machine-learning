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
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    
    def __init__(self, enc_out_dim: int = 512, latent_dim: int = 64):
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
    
    def __init__(self, dec_out_dim: int = 512, latent_dim: int = 64, out_dim: int = 2):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, dec_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(dec_out_dim, out_dim)
        self.fc_log_var = nn.Linear(dec_out_dim, out_dim)
        # self.fc_mu.bias.data.fill_(40)
        
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
        self.obj_wh = np.array([6, 6], dtype=np.float32)
        
    def render(self, s: np.ndarray) -> np.ndarray:
        if s.ndim == 2:  # s ... samples of xy rectangle centers
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
    # https://www.datagen.tech/10-promising-synthesis-papers-from-cvpr-2021/

    
    renderer = Renderer()
    encoder = Encoder()
    decoder = Decoder()
    
    summary(encoder)
    summary(decoder)
    
    optE = optim.Adam(encoder.parameters(), lr=1e-4)
    optD = optim.Adam(decoder.parameters(), lr=1e-4)
    
    # for the gaussian likelihood
    log_scale = nn.Parameter(torch.Tensor([0.0]))
    
    # define target gaussian
    mu_target = torch.tensor([45.0, 49.0])
    std_target = torch.tensor([1.0, 2.0])
    dist_target = Normal(mu_target, std_target)
    
    B = 32
    num_steps = 100
    losses = []
    
    for global_step in tqdm(range(num_steps)):
        
        optE.zero_grad(set_to_none=True)
        optD.zero_grad(set_to_none=True)
        
        s = dist_target.sample((B,))  # Bx2
        x_target = torch.from_numpy(renderer.render(s))  # Bx1xHxW
        mu, std = encoder(x_target)  # Bxlatent_dim
        
        q = torch.distributions.Normal(mu, std)  # q(z|x) approx. p(z|x)
        z = q.rsample()  # Bxlatent_dim (r ... by reparametrization)

        mu_source, std_source = decoder(z)  # Bx2
        p = torch.distributions.Normal(mu_source, std_source)  # p(s|z)
        s = p.rsample()  # Bx2
        
        with torch.no_grad():
            x_source = torch.from_numpy(renderer.render(s.detach().numpy()))  # Bx1xHxW
            recon_loss = gaussian_likelihood(x_source, log_scale, x_target)
            
            # plt.imshow(x_source.numpy()[0,0])
            # plt.show()
            
        log_prob = p.log_prob(s).sum((1,))  # Bx2 -> joint B, 
        
        # kl on latent gaussian to be close to N(0, 1)
        kl = kl_divergence(z, mu, std)  # B,
        
        # elbo
        elbo = (kl - (log_prob + recon_loss))
        elbo = elbo.mean()
        
        loss = elbo
        loss.backward()
        
        # print(decoder.fc_mu.bias.grad)
        # print(decoder.fc_mu.weight.grad)
        # print(decoder.fc_log_var.weight.grad)
        # pdb.set_trace()
        
        losses.append(loss.item())
        
        optE.step()
        optD.step()
    
    z = Normal(torch.zeros_like(z), torch.ones_like(z)).sample()
    # pdb.set_trace()
    mu_source, std_source = decoder(z)
    print('mu:', mu_source.data, 'goal:', mu_target.data)
    print('std:', torch.exp(std_source.data), 'goal:', std_target.data)
    
    plt.plot(losses, c='cyan', label='elbo')
    plt.legend()
    plt.show()