import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0', render_mode='human')

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DQN(4, 2).cuda()
policy_net.load_state_dict(torch.load(r'RL-model-tuned.pth.tar')['state'])
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

for _ in range(1000):
    env.render()
    action = policy_net(state.cuda()).max(1)[1].view(1, 1)
    print('action taken: ', action.item())
    observation, reward, terminated, truncated, info = env.step(action.item())
    if terminated:
        print(f'failed ta time {_}')
        observation, info = env.reset()
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

env.close()
