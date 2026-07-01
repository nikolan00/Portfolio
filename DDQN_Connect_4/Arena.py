from Agent import Agent
from Game import Game
from Frontend import pva
from threading import Lock, Thread
import time
from MiniMax import MiniMax


def selfplay_thread(agent, eps_min, eps_decay, stats_lock, val_freq):
    global stats
    while True:
        game = Game()
        state_actions = []
        while True:
            agent.player = game.player
            
            board_pre = game.board.copy()
            move = agent.make_move(game)
            state_actions.append((board_pre, move))

            if game.check_win() == 1 or (game.illegal_move and game.player == -1):
                with stats_lock:
                    stats[0] += 1
                break
            elif game.check_win() == -1 or (game.illegal_move and game.player == 1):
                with stats_lock:
                    stats[2] += 1
                break
            elif 0 not in game.board:
                with stats_lock:
                    stats[1] += 1
                break

        agent.add_to_memory(state_actions)
        agent.training_step() 
        agent.update_eps(eps_min, eps_decay)

        if sum(stats) >= val_freq:
            break
        


def train_selfplay(agent, episodes=10000, eps_min=0.02, num_workers=8, val_freq=10000):
    eps_decay = eps_min ** (1/(episodes*(2/3)))
    global stats
    stats = [0,0,0]
    stats_lock = Lock()
    episode = agent.training_episodes
    agent_minimax = MiniMax(depth=5)
    start = time.time()
    while True:
        threads = []
        for _ in range(num_workers):
            t = Thread(
                target=selfplay_thread,
                args=(agent, eps_min, eps_decay, stats_lock, val_freq),
            )
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        episode += val_freq
        if episode % 5000 == 0:
            agent.update_target()
        eps = agent.eps
        print(f'Episode (total): {agent.training_episodes}, Episode (cur. run): {episode}, Epsilon: {agent.eps}, Stats: {stats}')
        agent.eps = 0
        print(f'Against random (as X)')
        agent_duel(agent, Agent(), episodes=40, print_game_sample=False)
        print(f'Against random (as O)')
        agent_duel(Agent(), agent, episodes=40, print_game_sample=False)
        print(f'Against minimax (as X)')
        agent_duel(agent, agent_minimax, episodes=1, print_game_sample=False)
        print(f'Against minimax (as O)')
        agent_duel(agent_minimax, agent, episodes=1, print_game_sample=False)
        stats = [0,0,0]
        agent.eps = eps
        print(f'Passed time: {time.time() - start:.2f}')

        if episode >= episodes:
            print('Finished training!')
            break



def agent_duel(agentX, agentO, episodes=100, print_game_sample=True):
    agentX.player = 1
    agentO.player = -1
    stats = [0,0,0]
    for episode in range(episodes):
        game = Game()
        while True:
            if game.player == 1:
                agentX.make_move(game)
            else:
                agentO.make_move(game)

            if episode == episodes-1 and print_game_sample:
                game.print_board()

            if game.check_win() == 1 or (game.illegal_move and game.player == -1):
                stats[0] += 1
                break
            elif game.check_win() == -1 or (game.illegal_move and game.player == 1):
                stats[2] += 1
                break
            elif 0 not in game.board:
                stats[1] += 1
                break
    print(f'Stats: {stats}')
    



def human_vs_agent(agent):
    agent.eps = 0
    pva(agent)