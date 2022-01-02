
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt





if __name__ == '__main__':
    B = 16
    
    mu = Parameter(torch.zeros((1,)))
    std = Parameter(torch.ones((1,)))
    opt = optim.Adam((mu,std), lr=1)
    
    num_steps = 200
    for _ in range(num_steps):
        opt.zero_grad()
        
        p = Normal(mu, std)
        #sp = p.rsample((B,))
        
        q = Normal(torch.tensor([18.0]), torch.tensor([4.0]))
        sq = q.sample((B,))
        
        nlll = -p.log_prob(sq).sum()
        nlll.backward()
        
        opt.step()
        
    print(mu.data, std.data)