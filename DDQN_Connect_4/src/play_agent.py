import pickle
from threading import Lock
import glob
import re
import os
from arena import human_vs_agent


def extract_episodes(path):
    name = os.path.basename(path)
    match = re.search(r'(\d+)k_episodes', name)
    return int(match.group(1)) * 1000

def load_agent():
    best_path = max(glob.glob("../trained_agents/*.pkl"), key=extract_episodes)
    with open(best_path, "rb") as f:
        agent = pickle.load(f)

    agent.memory_lock = Lock()
    agent.eps_lock = Lock()
    agent.train_lock = Lock()
    print("Loaded agent with", agent.training_episodes, "episodes")
    return agent

if __name__ == "__main__":
    agent = load_agent()
    agent.verbose = False
    agent.player = 1
    agent.verbose = True
    human_vs_agent(agent)