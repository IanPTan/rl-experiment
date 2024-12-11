import torch as pt
from model import Actor, Critic
from game import Game, self_play, bot_play
from tqdm import tqdm


def mse(x, y):
    return (y - x) ** 2


def rbot(_):
    return pt.randn(9)


def bot_test(bot, actor, games=1000):
    games = int(games)
    game = Game()
    actor.eval()

    draws = 0
    wins = 0
    losses = 0
    turn = 0
    pbar = tqdm(range(games), desc="Playing...", unit="games")
    for i in pbar:
        _, _, winner = bot_play(game, bot, actor, turn)
        game.reset()
        turn = 1 - turn
        if winner == -1:
            draws += 1
        elif winner == turn:
            wins += 1
        else:
            losses += 1
        pbar.set_postfix(L=losses, D=draws, W=wins)

    return losses, draws, wins
        


def bot_train(bot, actor, critic, actor_optimizer, critic_optimizer, epochs=100):
    epochs = int(epochs)
    game = Game()
    actor.train()
    critic.train()


    turn = 0
    for i in tqdm(range(int(epochs)), desc="Playing...", unit="epochs"):
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


def self_train(actor, critic, actor_optimizer, critic_optimizer, epochs=100, batches=1, steps=1, noise=0.2, decay=0.8):
    epochs = int(epochs)
    batches = int(batches)
    steps = int(steps)
    game = Game()
    actor.train()
    critic.train()


    pbar = tqdm(range(epochs), desc="Playing...", unit="epochs")
    for i in pbar:
        inputs = pt.zeros(0, 18)
        outputs = pt.zeros(0, 9)
        moves = pt.zeros(0, 9)
        results = pt.zeros(len(inputs))

        for j in range(batches):
            game_inputs, game_outputs, game_moves, winner = self_play(game, actor, noise, decay)
            game.reset()

            game_results = pt.zeros(len(inputs))
            if winner == -1:
                game_results[:] = 0.5
            else:
                game_results[1 - winner::2] = 1


            inputs = pt.cat((inputs, game_inputs))
            outputs = pt.cat((outputs, game_outputs))
            moves = pt.cat((moves, game_moves))
            results = pt.cat((results, game_results))
        
        for k in range(steps):
            critic_optimizer.zero_grad()
            move_states = pt.cat((inputs.detach(), moves.detach()), dim=1)
            _, result_predictions = critic(move_states)
            critic_loss = mse(result_predictions, results)
            critic_loss.mean().backward()
            critic_optimizer.step()
            pbar.set_postfix(loss=f"{critic_loss.mean():.4f}")

        for l in range(steps):
            actor_optimizer.zero_grad()
            actor_inputs = inputs.detach()
            actor_outputs = moves.detach()
            output_states = pt.cat((actor_inputs, actor_outputs), dim=1)
            actor_loss, _ = critic(output_states)
            actor_loss.mean().backward()
            actor_optimizer.step()


if __name__ == "__main__":
    actor = Actor()
    critic = Critic()
    actor_optimizer = pt.optim.Adam(actor.parameters(), lr=1e-2, weight_decay=1e-3)
    critic_optimizer = pt.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-4)
    print(bot_test(rbot, actor, 1e4))
    #bot_train(rbot, actor, critic, actor_optimizer, critic_optimizer, 1e5)
    for i in range(10):
        self_train(actor, critic, actor_optimizer, critic_optimizer, 1e2, 1e2, 1e1, 0.2, 0.8)
        print(bot_test(rbot, actor, 1e4))
    #_ = self_play(Game(), actor, 0.1, 1)
