import torch as pt


def check_win(grid):
    vert = grid.sum(dim=-2)
    hori = grid.sum(dim=-1)
    dia1 = (grid * pt.eye(3)).sum(dim=(-2, -1))
    dia2 = (grid * pt.eye(3).flip((0,))).sum(dim=(-2, -1))
    won = (vert == 3).any() or (hori == 3).any() or (dia1 == 3).any() or (dia2 == 3).any()
    return won


def show(grid):
    string = [""] * 3
    for i in range(3):
        for j in range(3):
            x, o = grid[:, i, j].tolist()
            string[i] += {(0, 0): " ", (1, 0): "X", (0, 1): "O"}[(x, o)]
    return "\n─┼─┼─\n".join(["│".join(row) for row in string])


class Game:
    def __init__(self):
        self.grid = pt.zeros((2, 3, 3), dtype=bool)
        self.turn = 0
        self.state = pt.zeros(18, dtype=pt.float)
        self.open = pt.zeros(9, dtype=pt.float) + 1
        self.won = False

    def set(self, x):
        self.grid[self.turn].view(-1)[x] = 1
        for turn, half in enumerate((slice(0, 9), slice(9, 18))):
            self.state[half] = self.grid[int(self.turn == turn)].view(-1)
        self.open[:] = 1 - self.grid.sum(dim=0).view(-1)
        self.won = check_win(self.grid)
        self.turn = 1 - self.turn

    def move(self, y, x=None):
        if self.won:
            return self.state, self.open, self.won

        if x != None:
            y = y * 3 + x
        if type(y) == pt.Tensor and len(y.flatten()) == 1:
            y = y.item()
        if type(y) == pt.Tensor:
            y = pt.argmax(y * self.open).item()

        self.set(y)
        return self.state, self.open, self.won
    
    def show(self):
        print(show(self.grid))


if __name__ == "__main__":
    game = Game()
    won = 0
    while not won:
        y, x = input("Move: ").split(" ")
        _, _, won = game.move(int(y), int(x))
        game.show()

