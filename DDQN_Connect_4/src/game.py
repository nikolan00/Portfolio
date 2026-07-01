import numpy as np
import utils

class Game:
    def __init__(self):
        self.board = np.zeros(shape=(6,7))
        self.player = 1
        self.illegal_move = False


    def check_win(self):
        return utils.check_win(self.board)


    def print_board(self):
        return utils.print_board(self.board)
    
    
    def make_move(self, col):
        for row in range(6):
            if self.board[row,col] == 0:
                self.board[row,col] = self.player
                self.player *= -1
                return
        self.illegal_move = True