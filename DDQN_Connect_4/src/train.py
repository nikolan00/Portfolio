from agent import Agent
from arena import train_selfplay


if __name__ == "__main__":
    agent = Agent(learning_rate=1e-4) 
    agent.verbose = False

    train_selfplay(agent, eps_min=0.02, episodes=200, num_workers=16, val_freq=5, save_freq=10)