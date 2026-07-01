import pickle
from threading import Lock
import os
import numpy as np
import torch


def check_win(board):
    ROWS, COLS = board.shape

    def winner(player):
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 4 + 1):
                if np.sum(board[r, c:c+4]) == player * 4:
                    return True

        # Vertical
        for r in range(ROWS - 4 + 1):
            for c in range(COLS):
                if np.sum(board[r:r+4, c]) == player * 4:
                    return True

        # Diagonal (down-right)
        for r in range(ROWS - 4 + 1):
            for c in range(COLS - 4 + 1):
                if sum(board[r+i, c+i] for i in range(4)) == player * 4:
                    return True

        # Diagonal (up-right)
        for r in range(4 - 1, ROWS):
            for c in range(COLS - 4 + 1):
                if sum(board[r-i, c+i] for i in range(4)) == player * 4:
                    return True

        return False

    if winner(1):
        return 1
    elif winner(-1):
        return -1
    return 0


def print_board(board):
    print('-----------')
    for row in reversed(board):
        row_str = ''
        for cell in row:
            if cell == 1:
                row_str += 'X '
            elif cell == -1:
                row_str += 'O '
            else:
                row_str += '  '

        print(f'{row_str}')
        row_str = ''
    print('-----------')


def get_legal_moves(board):
    legal_cols = []
    for col in range(7):
        if board[5,col] == 0:
            legal_cols.append(col)
    return legal_cols


def make_move(board, col, player):
    for row in range(6):
        if board[row,col] == 0:
            board[row,col] = player
            return board
    print_board(board)
    raise Exception("Illegal move")


def get_illegal_moves(board):
    illegal_cols = []
    for col in range(7):
        if board[5,col] != 0:
            illegal_cols.append(col)
    return illegal_cols


def pick_random_argmax(tensor:torch.Tensor):
    if tensor.ndim == 1:
        max_indices = torch.where(tensor == tensor.max())[0]
        return max_indices[torch.randint(0, max_indices.numel(), (1,))].item()
    else:
        noise = torch.rand_like(tensor)
        return (tensor + noise * 1e-6).argmax(dim=1)
    

def map_symmetrical(board):
    mirror = np.flip(board, axis=1).copy()
    if np_to_hash(mirror) < np_to_hash(board):
        return mirror, True
    return board, False


def np_to_hash(arr):
    return tuple(arr.flatten())



def board_to_sym_hash(board):
    board,_ = map_symmetrical(board)
    return np_to_hash(board)

def save_agent(agent):
    agent.memory_lock = ""
    agent.eps_lock = ""
    agent.train_lock = ""

    os.makedirs("trained_agents", exist_ok=True)
    save_location = f'../trained_agents/{agent.training_episodes // 1000}k_episodes.pkl'
    with open(save_location, 'wb') as f:
        pickle.dump(agent, f)
    print(f'Saved checkpoint under trained_agents/{agent.training_episodes // 1000}k_episodes.pkl!')

    agent.memory_lock = Lock()
    agent.eps_lock = Lock()
    agent.train_lock = Lock()
