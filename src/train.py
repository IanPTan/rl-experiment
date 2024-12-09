import torch as pt
from model import Actor, Critic
from game import Game, self_play, bot_play
from tqdm import tqdm


def mse(x, y):
    return (y - x) ** 2


def rbot(_):
    return pt.randn(9)


def bot_test(bot, actor, max_games=1000):
    game = Game()
    actor.eval()

    draws = 0
    wins = 0
    losses = 0
    turn = 0
    for i in tqdm(range(max_games), desc="Playing...", unit="games"):
        _, _, winner = bot_play(game, bot, actor, turn)
        game.reset()
        turn = 1 - turn
        if winner == -1:
            draws += 1
        elif winner == turn:
            wins += 1
        else:
            losses += 1

    return losses, draws, wins
        


def bot_train(bot, actor, critic, actor_optimizer, critic_optimizer, max_games=100):
    game = Game()
    actor.train()
    critic.train()


    turn = 0
    for i in tqdm(range(max_games), desc="Playing...", unit="games"):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        inputs, moves, winner = bot_play(game, bot, actor, turn)
        game.reset()
        turn = 1 - turn

        results = pt.zeros(len(inputs))
        if winner == -1:
            results[:] = 0.5
        else:
            results[1 - winner::2] = 1
        
        move_states = pt.cat((inputs, moves), dim=1)
        _, result_predictions = critic(move_states)
        critic_loss = mse(result_predictions, results)
        critic_loss.mean().backward()
        critic_optimizer.step()

        actor_inputs = inputs.detach()[turn::2]
        actor_outputs = moves.detach()[turn::2]
        output_states = pt.cat((actor_inputs, actor_outputs), dim=1)
        actor_loss, _ = critic(output_states)
        actor_loss.mean().backward()
        actor_optimizer.step()


def self_train(actor, critic, actor_optimizer, critic_optimizer, max_games=100, noise=0.2):
    game = Game()
    actor.train()
    critic.train()


    for i in tqdm(range(max_games), desc="Playing...", unit="games"):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        inputs, outputs, moves, winner = self_play(game, actor, noise)
        game.reset()

        results = pt.zeros(len(inputs))
        if winner == -1:
            results[:] = 0.5
        else:
            results[1 - winner::2] = 1
        
        move_states = pt.cat((inputs, moves), dim=1)
        _, result_predictions = critic(move_states)
        critic_loss = mse(result_predictions, results)
        critic_loss.mean().backward()
        critic_optimizer.step()

        actor_inputs = inputs.detach()
        actor_outputs = moves.detach()
        output_states = pt.cat((actor_inputs, actor_outputs), dim=1)
        actor_loss, _ = critic(output_states)
        actor_loss.mean().backward()
        actor_optimizer.step()


if __name__ == "__main__":
    actor = Actor()
    critic = Critic()
    actor_optimizer = pt.optim.Adam(actor.parameters(), lr=1e-3, weight_decay=1e-4)
    critic_optimizer = pt.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-4)
    #self_train(actor, critic, actor_optimizer, critic_optimizer, 10**5, 0.2)
    print(bot_test(rbot, actor, 10**4))
    bot_train(rbot, actor, critic, actor_optimizer, critic_optimizer, 10**5)
    print(bot_test(rbot, actor, 10**4))
    #_ = self_play(Game(), actor, 0.1, 1)
