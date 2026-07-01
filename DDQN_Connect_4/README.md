# DDQN-based Connect 4 AI

This repository is an implementation of the DDQN algorithm for Connect 4. 

Tested with:
- Python=3.10.18
- pytorch 2.9.1
- CUDA 13.3

## Environment

To be able to fully use this repository please setup your environment as follows:

1. Install dependencies:
```shell
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```


## Usage
Start model training by running the following commands:
```shell
cd src
python train.py
```

Play against the latest checkpoint by running the following command:
```shell
python play_agent.py
```