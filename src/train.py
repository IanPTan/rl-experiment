import torch as pt
from model import Actor, Critic
from game import Game
from tqdm import tqdm

def ppoUpdate(actor, critic, actorOptimizer, criticOptimizer, states, actions, rewards, oldProbs, clip_param=0.2):
    states = pt.stack(states)
    actions = pt.tensor(actions, dtype=pt.long)
    rewards = pt.tensor(rewards, dtype=pt.float32)
    oldProbs = pt.stack(oldProbs)

    # calculate new probabilities and values
    newProbs = actor(states).gather(1, actions.unsqueeze(1))
    values = critic(states).squeeze()

    # calculate advantages
    advantages = rewards - values.detach()

    # calculate ratio (pi_theta / pi_theta_old)
    ratio = (newProbs / oldProbs).squeeze()

    # surrogate loss with clipping
    surr1 = ratio * advantages
    surr2 = pt.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
    actorLoss = -pt.min(surr1, surr2).mean()

    # critic loss (it's the MSE between predicted and actual rewards)
    criticLoss = pt.nn.functional.mse_loss(values, rewards)

    # update actor and critic
    

def train():
    pass


if __name__ == "__main__":
    train()
