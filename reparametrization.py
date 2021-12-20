import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional, Tuple, Generator
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


def _midrange(samples: Tensor):
    return (samples.max() + samples.min()) / 2
    

def _range(samples: Tensor):
    return samples.max() - samples.min()


class Uniform:
    
    def __init__(self, a: Optional[Tensor] = None, b: Optional[Tensor] = None) -> None:
        super().__init__()
        
        if a is None:
            a = torch.tensor([[0.0]])
        if b is None:
            b = torch.tensor([[1.0]])
            
        self.a = Parameter(a)
        self.b = Parameter(b)
        self.size: Tuple[int] = a.shape
        
    def sample(self):
        x = torch.rand(self.size)
        return self.a + x * (self.b - self.a)
    
    def parameters(self) -> Generator[Tensor, None, None]:
        return (p for p in (self.a, self.b))
    
    @property
    def mean(self):
        return (self.a + self.b) / 2
    
    @property
    def std(self):
        return torch.sqrt((self.b - self.a)**2 / 12)
    
    @staticmethod
    def maximum_likelihood_estimators(samples: Tensor):
        mr = _midrange(samples)
        r = _range(samples)
        
        a_hat = mr - 0.5 * r
        b_hat = mr + 0.5 * r 
        return a_hat, b_hat  # estimators of "a" and "b"
    
    def pdf(self, x):
        m = torch.logical_and(x >= self.a, x <= self.b).float()
        return m / (self.b - self.a)
    
    
class Normal:
    
    def __init__(self, mean: Optional[Tensor] = None, std: Optional[Tensor] = None) -> None:
        super().__init__()
        
        if mean is None:
            mean = torch.tensor([[0.0]])
        if std is None:
            std = torch.tensor([[1.0]])
            
        self.mean = Parameter(mean)
        self.std = Parameter(std)
        self.size: Tuple[int] = mean.shape
        
    def sample(self):
        x = torch.normal(0, 1, self.size)
        return self.mean + x * self.std
    
    def parameters(self) -> Generator[Tensor, None, None]:
        return (p for p in (self.mean, self.std))
    
    @staticmethod
    def maximum_likelihood_estimators(samples: Tensor):
        mean_hat = samples.mean()
        std_hat = samples.std()
        return mean_hat, std_hat
    
    def pdf(self, x):
        return 1/math.sqrt(2*math.pi)/self.std * torch.exp(-(x-self.mean)**2/2/self.std**2)
            

if __name__ == '__main__':
    
    # P = Uniform(torch.tensor([[0.0]]), torch.tensor([[70.0]]))
    # P_target = torch.distributions.Uniform(torch.tensor([[10.0]]), torch.tensor([[50.0]]))
    
    P = Normal(torch.tensor([[10.0]]), torch.tensor([[5.0]]))
    P_target = torch.distributions.Normal(torch.tensor([[20.0]]), torch.tensor([[3.0]]))
    
    num_steps = 1000
    opt = optim.SGD(P.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=1, 
        total_steps=num_steps,
        div_factor=1.0,
        final_div_factor=10.0,
    )
    
    pbar = tqdm(range(num_steps))
    losses = []
    
    for step in pbar:
        
        y_target = torch.cat([P_target.sample() for _ in range(128)])
        opt.zero_grad(set_to_none=True)
        
        # negative log likelihood loss calculation
        # p = P.pdf(y_target)
        # loss = -torch.log(torch.clamp(p, 1e-3, None)).mean()
        
        # a_hat, b_hat = P.maximum_likelihood_estimators(y_target)
        # loss = F.l1_loss(P.a, a_hat) + F.l1_loss(P.b, b_hat)
        
        # negative log likelihood loss calculation
        p = P.pdf(y_target)
        loss = -torch.log(torch.clamp(p, 1e-3, None)).mean()
        
        loss.backward()
        
        # print(P.a.grad, P.b.grad)
        # import pdb; pdb.set_trace()
        
        opt.step()
        scheduler.step()
        
        if step % 50 == 0:
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
        pbar.update()
    
    print()
    print("Final estimates:")
    for i, param in enumerate(P.parameters()):
        print(f"param{i} = {param}")

    plt.plot(losses)
    plt.show()
    