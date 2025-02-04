import torch as pt
import torch.nn as nn

Actor = lambda: nn.Sequential(nn.Linear(18, 9), nn.Softmax(dim=-1))
Critic = lambda: nn.Linear(18, 1)
rbot = lambda x: pt.randn(9)
