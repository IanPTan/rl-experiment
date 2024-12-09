import torch as pt


def get_state(grid, turn):
    if turn:
        grid = grid.flip((0,))

    state = grid.view(-1).to(pt.float)

    return state


def get_open(grid):
    open = 1 - grid.sum(dim=0).view(-1).to(pt.float)

    return open


def check_win(grid):
    vertical_scores = grid.sum(dim=-2)
    vertical_win = (vertical_scores == 3).sum(dim=1)
    horizontal_scores = grid.sum(dim=-1)
    horizontal_win = (horizontal_scores == 3).sum(dim=1)

    diagonal1_scores = (grid * pt.eye(3)).sum(dim=(1, 2))
    diagonal1_win = diagonal1_scores == 3
    diagonal2_scores = (grid * pt.eye(3).flip((0,))).sum(dim=(1, 2))
    diagonal2_win = diagonal2_scores == 3

    won = (vertical_win + horizontal_win + diagonal1_win + diagonal2_win).to(bool)

    return won


def show(grid):
    string = [""] * 3

    for i in range(3):
        for j in range(3):
            x, o = grid[:, i, j].tolist()
            string[i] += {(0, 0): " ", (1, 0): "X", (0, 1): "O"}[(x, o)]

    return "\n─┼─┼─\n".join(["│".join(row) for row in string])


def self_play(game, model, noise=0, show=False):
    inputs = pt.zeros(0, 18)
    outputs = pt.zeros(0, 9)
    moves = pt.zeros(0, 9)

    done = False
    while not done:
        state = game.get_state()
        output = model(state)
        move = output + pt.rand(9) * noise
        done = game.move(move) 

        inputs = pt.cat((inputs, state.view(1, 18)))
        outputs = pt.cat((outputs, output.view(1, 9)))
        moves = pt.cat((moves, move.view(1, 9)))

        if show:
            game.print()
            print()

    return inputs, outputs, moves, game.winner


class Game:
    def __init__(self):
        self.grid = pt.zeros((2, 3, 3), dtype=bool)
        self.turn = 0
        self.done = False
        self.winner = -1

    def set(self, x):
        self.grid[self.turn].view(-1)[x] = 1
        self.turn = 1 - self.turn

        win = check_win(self.grid)
        if win.any():
            self.done = True
            self.winner = pt.where(win)[0].item()
        else:
            self.done = not self.get_open().any()

    def move(self, y, x=None):
        if self.done:
            return self.done

        if x != None:
            y = y * 3 + x
        if type(y) == pt.Tensor and len(y.flatten()) == 1:
            y = y.item()
        if type(y) == pt.Tensor:
            y = pt.argmax(y * self.get_open()).item()
        self.set(y)

        return self.done
    
    def get_state(self):
        return get_state(self.grid, self.turn)

    def get_open(self):
        return get_open(self.grid)

    def print(self):
        print(show(self.grid))

    def reset(self):
        self.grid = pt.zeros((2, 3, 3), dtype=bool)
        self.turn = 0
        self.done = False
        self.winner = -1


if __name__ == "__main__":
    game = Game()
    won = False
    while not won:
        y, x = input("Move: ").split(" ")
        won = game.move(int(y), int(x))
        game.print()

