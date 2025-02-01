import torch as pt
from model import Actor, Critic
from game import Game, self_play, bot_play
from tqdm import tqdm


def mse(x, y):
    return (y - x) ** 2


def rbot(_):
    return pt.randn(9)


if __name__ == "__main__":
    actor = Actor()
    critic = Critic()
