import numpy as np


class Game:
    def __init__(self, n):
        self.n = n
        self.board = np.chararray(shape=(n, n), itemsize=3)
        self.board[:] = '_'

    def get_symbol_indices(self, symbol):

        indices = np.where(self.board.astype(str) == symbol)

        return indices

    def get_allowed_moves(self):

        return self.get_symbol_indices('_')

    def make_move(self, symbol, coord):

        if self.board[coord] == b'_':
            self.board[coord] = symbol
            return True
        else:
            return False

    def draw_board(self):
        print(self.board.astype(str))
        print("_____________")

    def check_winner(self, symbol):

        for row in range(self.n):
            if all(self.board[row, :].astype(str)== symbol):
                return True

        for col in range(self.n):
            if all(self.board[:, col].astype(str)== symbol):
                return True

        if all(self.board.diagonal().astype(str) == symbol):
                return True

        if all(np.rot90(self.board).diagonal().astype(str) == symbol):
                return True

        return False

    def enumerate_state(self):

        state = ''

        for ch in (self.board.flatten().astype(str)):
            state +=ch

        return state


'''
game = Game(3)

game.make_move('x', (0,0))
game.draw_board()

# game.make_move('x', (0,1))
game.make_move('x', (0,2))
game.draw_board()
print(game.check_winner('x'))
game.make_move('x', (1,1))
game.make_move('x', (2,0))

game.draw_board()
print(game.check_winner('x'))
print(game.enumerate_state())
print(game.get_allowed_moves())

'''