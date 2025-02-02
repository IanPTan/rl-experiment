import torch as pt
from model import Actor, Critic
from game import Game

def ppoUpdate(actor, critic, actorOptimizer, criticOptimizer, states, actions, rewards, oldProbs, clip_param=0.2, c1=0.5, c2=0.01):
    states = pt.stack(states)
    actions = pt.tensor(actions, dtype=pt.long)
    rewards = pt.tensor(rewards, dtype=pt.float32)
    oldProbs = pt.stack(oldProbs)

    # new probabilities and values
    newProbs = actor(states).gather(1, actions.unsqueeze(1))
    values = critic(states).squeeze()

    advantages = rewards - values.detach()

    # ratio (pi_theta / pi_theta_old)
    ratio = (newProbs / oldProbs).squeeze()

    # 1. surrogate loss with clipping
    surr1 = ratio * advantages
    surr2 = pt.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
    actorLoss = -pt.min(surr1, surr2).mean()

    # 2. critic loss (mse between predicted and actual rewards)
    criticLoss = pt.nn.functional.mse_loss(values, rewards)

    # 3. entropy bonus, i defined the function below this one
    entropy = compute_entropy(newProbs).mean()

    # ppo entire equation
    loss = actorLoss - c1 * criticLoss + c2 * entropy

    # update actor and critic
    actorOptimizer.zero_grad()
    criticOptimizer.zero_grad()
    loss.backward()
    actorOptimizer.step()
    criticOptimizer.step()

def compute_entropy(probs):
    return -pt.sum(probs * pt.log(probs + 1e-10), dim=-1)  # the + 1e-10 helps avoid log(0)

def train(epochs=1000):
    actor = Actor()
    critic = Critic()
    actorOptimizer = pt.optim.Adam(actor.parameters(), lr=1e-4)
    criticOptimizer = pt.optim.Adam(critic.parameters(), lr=1e-3)

    game = Game()
    for epoch in range(epochs):
        states, actions, rewards, oldProbs = [], [], [], []
        game.reset()
        done = False

        while not done:
            state = game.get_state()
            actionProbs = actor(state)
            action = pt.multinomial(actionProbs, 1).item()  # sample an action
            oldProb = actionProbs[action]  # store the probability of the chosen action

            done = game.move(actionProbs)
            reward = 1 if game.winner == 0 else -1 if game.winner == 1 else 0  # reward based on winner

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            oldProbs.append(oldProb)

        # update the actor and critic using PPO
        ppoUpdate(actor, critic, actorOptimizer, criticOptimizer, states, actions, rewards, oldProbs)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Reward: {sum(rewards)}")

if __name__ == "__main__":
    train() 