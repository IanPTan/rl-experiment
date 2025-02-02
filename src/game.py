import torch as pt

class Game:
    def __init__(self):
        self.grid = pt.zeros((2, 3, 3), dtype=bool)  # i'm keeping what you had for the constructor
        self.turn = 0  
        self.done = False
        self.winner = -1  

    def get_state(self):
        return self.grid.view(-1).float()  # flatten grid into a 1D tensor

    def get_open(self):
        return 1 - self.grid.sum(dim=0).view(-1).float()  # 1 for open, 0 for taken to see which cells are open

    def move(self, action_probs):
        if self.done:
            return self.done

        # hide already taken cells
        open_spaces = self.get_open()
        valid_action_probs = action_probs * open_spaces

        # choose action with the highest probability among valid moves
        action = pt.argmax(valid_action_probs).item()

        # and then mark the move on the grid and switch turns
        self.grid[self.turn].view(-1)[action] = 1
        self.turn = 1 - self.turn

        # check for a winner
        win = self.check_win()
        if win.any():
            self.done = True
            self.winner = pt.where(win)[0].item()
        else:
            self.done = not self.get_open().any()  # check for a draw

        return self.done

    def check_win(self):
        # check rows, columns, and diagonals for a win
        vertical = self.grid.sum(dim=-2) == 3
        horizontal = self.grid.sum(dim=-1) == 3
        diagonal1 = (self.grid * pt.eye(3)).sum(dim=(1, 2)) == 3
        diagonal2 = (self.grid * pt.eye(3).flip((0,))).sum(dim=(1, 2)) == 3
        return vertical.any(dim=1) | horizontal.any(dim=1) | diagonal1 | diagonal2

    def reset(self):
        self.grid = pt.zeros((2, 3, 3), dtype=bool)
        self.turn = 0
        self.done = False
        self.winner = -1