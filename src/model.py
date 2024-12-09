import torch as pt


class Actor(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pt.nn.Sequential(
                pt.nn.Linear(18, 9),
                pt.nn.Softmax(dim=-1)
                )

    def forward(self, x):
        return self.model(x)


class Critic(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = pt.nn.Linear(27, 1)
        self.sigmoid = pt.nn.Sigmoid()
                

    def forward(self, x):
        y = self.linear(x)
        return y, self.sigmoid(y)


if __name__ == "__main__":
    actor = Actor()
    critic = Critic()
