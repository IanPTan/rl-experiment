import torch as pt
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc = nn.Linear(18, 9)  # 18 inputs (game state), 9 outputs (one for each cell on tictactoe grid)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc(x))  # output probabilities for each move

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc = nn.Linear(18, 1)  # 18 inputs and 1 output (value of the state)

    def forward(self, x):
        return self.fc(x)  # output the value of the state