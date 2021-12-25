import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
import numpy as np
from typing import Tuple
import pdb


class Discriminator(nn.Sequential):
    
    def __init__(
            self, 
            nc: int = 1,
            ndf: int = 16,
            nlayers: int = 4, 
            bias: bool = True,
            batchnorm: bool = False,
        ):
        submodules = []
        cin = nc
        cout = ndf
        
        for _ in range(nlayers):
            
            if batchnorm:
                submodules.extend([
                    nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
            else:
                submodules.extend([
                    nn.Conv2d(cin, cout, 4, 2, 1, bias=bias),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
            
            cin = cout
            cout = 2 * cin
        
        submodules.append(nn.Conv2d(cin, 1, 1, 1))
                    
        super().__init__(
            *submodules[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            submodules[-1]
        )  # => returns logits
        
        
def weights_init(m):
    # model.apply(weights_init)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

if __name__ == '__main__':
    
    canvas = np.ones((256, 256), dtype=np.float32)
    
    # z <=> center (x,y) position of rectangle
    
    # define source gaussian
    z_mu_source = Parameter(torch.tensor([50.0, 40.0]))  # 2,
    z_std_source = Parameter(torch.tensor([11.0, 12.0]))  # 2,
    
    optG = optim.Adam((z_mu_source, z_std_source), lr=0.0002, betas=(0.5, 0.999))
    
    # define target gaussian
    z_mu_target = Parameter(torch.tensor([150.0, 190.0]))
    z_std_target = Parameter(torch.tensor([25.0, 20.0]))

    obj_wh = np.array([[8, 8]], dtype=np.float32)  # 1x2
    
    # update distribution parameters with reparametrization trick
    unorm = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))  # 2,
    
    D = Discriminator()
    D.apply(weights_init)
    
    def render(z: np.ndarray) -> np.ndarray:
        if z.ndim == 2:
            B = z.shape[0]
        else:
            B = 1
        
        x = np.empty((B, 1, *canvas.shape), dtype=np.float32)
        for b in range(B):
            img = canvas.copy()
            tl: Tuple[int] = tuple(map(int, z - obj_wh))  
            br: Tuple[int] = tuple(map(int, z + obj_wh))
            color: Tuple[int] = (0, 0, 0)
            cv.rectangle(img, tl, br, color, -1)
            x[b, 0] = img
            
        return x
    
    # typically G(z) where z ~ N(0, 1) but now
    # G(z) := render(z1, z2,...) where z_i some random simulation variable ~ N(mu_i, sig_i)
    G = render
    
    B = 8
    
    criterion = nn.BCEWithLogitsLoss()
    zeros = torch.zeros((B,), dtype=torch.float32)
    ones = torch.ones((B,), dtype=torch.float32)
         
    # sample from the source distribution
    z = z_mu_source + z_std_source * unorm.sample((B,))  # Bx2
    samples = z
    x_source = torch.from_numpy(G(z.detach().numpy()))  # Bx1xHxW
    
    # sample from the target distribution
    z = z_mu_target + z_std_target * unorm.sample(B,)  # Bx2
    x_target = render(z.detach().numpy())  # Bx1xHxW
    
    # standard GAN loss for the discriminator
    lossD =  criterion(D(x_target), ones).mean() + criterion(D(x_source.detach()), zeros).mean()
    
    # the non-saturating GAN loss would be
    # lossG = criterion(D(x_source), ones).mean()
    # but we cannot backpropagate through the non-differentiable renderer!
    # =>
    samples
    N = Normal(z_mu_source, z_std_source)
    N.log_prob(samples)
    
    # while True:
    
    #     z = z_mu_source + z_std_source * unorm.sample()  # 2,
    #     x = G(z.detach().numpy())  # Bx1xHxW
    #     cv.imshow('source', x)
        
    #     z = z_mu_target + z_std_target * unorm.sample()  # 2,
    #     x = render(z.detach().numpy())
    #     cv.imshow('target', x)
                
    #     key = cv.waitKey(1000) & 0xFF
    #     if key == ord('q'):
    #         break
        
    # cv.destroyAllWindows()     