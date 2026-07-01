import numpy as np
import time
import utils


class MiniMax:
    def __init__(self, depth=7):
        self.player = 1
        self.depth = depth
        self.verbose = False

    
    def make_move(self, game):
        legal_moves = utils.get_legal_moves(game.board)
        assert len(legal_moves) > 0
        evals = []
        self.evaluated_pos = {}
        start = time.time()
        for move in legal_moves:
            board = game.board.copy()
            board, last_move = make_move(board, move, self.player)
            evals.append(self.miniMax(board, self.depth, -999999, 999999, -self.player, last_move))
        if self.verbose:
            print(list(map(float, evals)), "Time spent:", time.time()-start)
        if self.player == 1:
            move = legal_moves[np.argmax(evals)]
        else:
            move = legal_moves[np.argmin(evals)]
        game.make_move(move)
        return move


    
    def miniMax(self, board, depth, alpha, beta, player, last_move):
        d_board = embed_depth(board, depth)
        if utils.board_to_sym_hash(d_board) in self.evaluated_pos:
            return self.evaluated_pos[utils.board_to_sym_hash(d_board)]
        gameOver =  0 not in board or check_win_at(board, last_move[0], last_move[1], -player) != 0

        if depth == 0 or gameOver:
            eval = self.evaluate(board, player, depth, last_move)
            self.evaluated_pos[utils.board_to_sym_hash(d_board)] = eval
            return eval

        maxEval = -999999.0
        minEval = 999999.0

        for i in utils.get_legal_moves(board):
            # clone stacks
            board_copy = board.copy()

            # simulate move
            board_copy, last_move = make_move(board_copy, i, player)

            eval_val = self.miniMax(
                board_copy,
                depth - 1,
                alpha,
                beta,
                -player,
                last_move
            )

            maxEval = max(maxEval, eval_val)
            minEval = min(minEval, eval_val)

            if player == 1:
                alpha = max(alpha, eval_val)
            else:
                beta = min(beta, eval_val)

            if beta <= alpha:
                break

        if player == 1:
            eval = maxEval
        else:
            eval = minEval
        self.evaluated_pos[utils.board_to_sym_hash(d_board)] = eval
        return eval

    

    def evaluate(self, board, player, depth, last_move):
        d_board = embed_depth(board, depth)
        if utils.board_to_sym_hash(d_board) in self.evaluated_pos:
            return self.evaluated_pos[utils.board_to_sym_hash(d_board)]
        # Check if game is already over, in that case the enemy player won
        p_winner = check_win_at(board, last_move[0], last_move[1], -player)
        if p_winner:
            return (1000 + depth) * (-player)

        # In case it isnt and we have more than 1 winning move, game is won since its our move
        winningMoveCount = len(get_winning_moves(board, player))
        if winningMoveCount > 1:
            eval = (900 + depth) * player
        else:
            eval = np.sum(board[:,3]) * (3 + depth / 1000) # center cols
            eval += 0.5 * (np.sum(board[:,2]) + np.sum(board[:,4])) * (2 + depth / 1000) # next to center cols
            eval += 10 * (win_setups(board, player) - win_setups(board, -player)) * player
        return eval


def check_win_at(board, row, col, player):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        # check positive direction
        r, c = row + dr, col + dc
        while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        # check negative direction
        r, c = row - dr, col - dc
        while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        if count >= 4:
            return True
    return False


def get_winning_moves(board, player):
    legal_moves = utils.get_legal_moves(board)
    winning_moves = []

    for col in legal_moves:
        row = np.argmax(board[:, col] == 0)
        if board[row, col] != 0:
            continue
        board[row, col] = player
        if check_win_at(board, row, col, player):
            winning_moves.append(col)
        board[row, col] = 0

    return winning_moves


def undo_move(board, col):
    for row in reversed(range(6)):
        if board[row,col] != 0:
            board[row,col] = 0
            return board
        

def win_setups(board, player, stop_after=2):
    wins = 0
    board = board.copy()
    for col in range(board.shape[1]):
        # Find the first empty row in this column
        empty_rows = np.where(board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            continue
        row = empty_rows[0]

        # Simulate opponent move
        board[row, col] = -player
        # Then simulate player move on top
        next_empty_rows = np.where(board[:, col] == 0)[0]
        if len(next_empty_rows) == 0:
            board[row, col] = 0
            continue
        next_row = next_empty_rows[0]
        board[next_row, col] = player

        # Check if board is full or player has a win
        if np.all(board != 0) or check_win_at(board, next_row, col, player):
            wins += 1
            if wins >= stop_after:
                board[next_row, col] = 0
                board[row, col] = 0
                return wins

        # Undo moves
        board[next_row, col] = 0
        board[row, col] = 0

    return wins


def make_move(board, col, player):
    for row in range(6):
        if board[row,col] == 0:
            board[row,col] = player
            return board, (row, col)
    raise Exception("Illegal move")


def embed_depth(board, depth):
    d_board = board.copy()
    d_board[3,0] = d_board[3,0] + 10 * depth # board with embedded depth
    return d_board
