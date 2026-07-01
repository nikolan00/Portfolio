import random
import numpy as np
import torch
import torch.nn as nn
from nn import CNN
from collections import deque
from threading import Lock
import utils


class Agent:
    def __init__(self, batch_size=64, learning_rate=0.1, verbose=False):
        self.batch_size = batch_size
        self.verbose = verbose

        self.player = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.eps = 1.0
        self.training_episodes = 0
        self.replay_memory = deque(maxlen=200000)
        self.policy_net = CNN().to(self.device)
        self.target_net = CNN().to(self.device)
        self.policy_net.train()
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.memory_lock = Lock()
        self.eps_lock = Lock()
        self.train_lock = Lock()


    def update_eps(self, eps_min, eps_decay):
        with self.eps_lock:
            self.eps = max(eps_min, eps_decay ** self.training_episodes)


    def get_target_Q(self, board_batch, player_batch, is_final_batch):
        boards_batch = []
        for i,board in enumerate(board_batch):
            board = board.copy()
            board *= player_batch[i] # keep 1 as the player to move
            boards_batch.append(board)

        input = torch.from_numpy(np.stack(boards_batch)).to(self.device).to(torch.float32)
        with torch.no_grad():
            q_vals = self.target_net.forward(input)

        for i,board in enumerate(boards_batch):
            if is_final_batch[i]:
                # No moves to predict, state was final
                q_vals = q_vals.clone()
                q_vals[i] = 0
        return q_vals


    def get_Q(self, board_batch, player_batch, grad=False):
        boards_batch = []
        for i,board in enumerate(board_batch):
            board = board.copy()
            board *= player_batch[i] # keep 1 as the player to move
            boards_batch.append(board)
        input = torch.from_numpy(np.stack(boards_batch)).to(self.device).to(torch.float32)
        if not grad:
            with torch.no_grad():
                q_vals = self.policy_net.forward(input)
                for i,board in enumerate(boards_batch):
                    illegal_moves = utils.get_illegal_moves(board)
                    q_vals[i,illegal_moves] = -9999
        else:
            q_vals = self.policy_net.forward(input)
        return q_vals      
    

    def make_move(self, game):
        legal_moves = utils.get_legal_moves(game.board)
        assert len(legal_moves) > 0
        if random.random() < self.eps:
            move = random.randint(0, 6)
            while move not in legal_moves:
                move = random.randint(0, 6)
        else:
            # Policy move
            sym_board, flipped = utils.map_symmetrical(game.board)
            q = self.get_Q([sym_board], [self.player])[0]
            move = utils.pick_random_argmax(q)
            if flipped:
                move = 6-move
            if self.verbose:
                print(q)

        game.make_move(move)
        return move
    
    
    def add_to_memory(self, state_actions):
        for board,action in state_actions:
            board, flipped = utils.map_symmetrical(board)
            if flipped:
                action = 6-action
            player = 1 if np.sum(board.flatten()) == 0 else -1
            next_board = board.copy()
            # Punish illegal moves
            if action not in utils.get_legal_moves(board):
                reward = -player
                is_final = True
            else:
                next_board = utils.make_move(next_board, action, player)
                reward = utils.check_win(next_board) * player
                is_final = utils.check_win(next_board) != 0 or 0 not in next_board
            memory_entry = {'state': board.copy(), 'next_state': next_board, 'reward': reward, 'is_final': is_final, 'player': player, 'action': action}
            with self.memory_lock:
                self.replay_memory.append(memory_entry)
        
    
    def training_step(self):
        with self.memory_lock:
            batch = random.sample(self.replay_memory, min(len(self.replay_memory),self.batch_size))
        player_batch, state_batch, reward_batch, is_final_batch, next_state_batch, action_batch = [],[],[],[],[],[]
        for sample in batch:
            reward_batch.append(sample['reward'])
            is_final_batch.append(sample['is_final'])
            state_batch.append(sample['state'])
            next_state_batch.append(sample['next_state'])
            player_batch.append(sample['player'])
            action_batch.append(sample['action'])

        # Choose best move based on policy and eval based on target
        enemy_player_batch = -np.array(player_batch)   
        reward_batch = torch.tensor(reward_batch).to(self.device) # from movers perspective

        q_next = self.get_Q(next_state_batch, enemy_player_batch)                
        targ_q_next = self.get_target_Q(next_state_batch, enemy_player_batch, is_final_batch)

        optim_answers = utils.pick_random_argmax(q_next)
        batch_idx = torch.arange(targ_q_next.shape[0], device=targ_q_next.device)
        optim_future_qs = -targ_q_next[batch_idx, optim_answers] # "minus" since these are Q-vals for the opponents move
        q_target = reward_batch + self.gamma * optim_future_qs
        with self.train_lock:
            q_pred = self.get_Q(state_batch, player_batch, True)[batch_idx, action_batch]
            loss = self.criterion(q_pred, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.training_episodes += 1

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())