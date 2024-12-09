import torch as pt
from model import Actor, Critic
from game import Game, self_play
from tqdm import tqdm


def mse(x, y):
    return (y - x) ** 2


def self_train(actor, critic, actor_optimizer, critic_optimizer, max_games=100, noise=0.2):
    game = Game()
    actor.train()
    critic.train()


    for i in tqdm(range(max_games), desc="Playing...", unit="games"):
        inputs, outputs, moves, winner = self_play(game, actor, noise)
        game.reset()

        results = pt.zeros(len(inputs))
        if winner == -1:
            results[:] = 0.5
        else:
            results[winner::2] = 1
        
        critic_optimizer.zero_grad()
        move_states = pt.cat((inputs, moves), dim=1)
        _, result_predictions = critic(move_states)
        critic_loss = mse(result_predictions, results)
        critic_loss.mean().backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        output_states = pt.cat((inputs, outputs), dim=1)
        actor_loss, _ = critic(output_states)
        actor_loss.mean().backward()
        actor_optimizer.step()


if __name__ == "__main__":
    actor = Actor()
    critic = Critic()
    actor_optimizer = pt.optim.Adam(actor.parameters(), lr=1e-3, weight_decay=1e-4)
    critic_optimizer = pt.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-4)
    self_train(actor, critic, actor_optimizer, critic_optimizer, 10**5, 0.2)
