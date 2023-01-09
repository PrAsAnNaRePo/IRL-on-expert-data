import gym
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE = True
TRAIN = True

# initialize the environment.
env = gym.make('CartPole-v0', render_mode='human')

# creating the reward model.
class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_bins=1) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.numbins = num_bins
        self.block1 = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.Dropout(0.23),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.action_dim, 100),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.Linear(200, 8),
            nn.Dropout(0.44),
            nn.ReLU(True),
            nn.Linear(8, 1),
        )

    def forward(self, s, a):
        s = self.block1(s)
        a = self.block2(a)
        return self.block3(torch.cat([s, a]).reshape(-1, 200)).reshape(-1, 1)

def get_human_feedback():
    feedback = float(input('Enter the required rating: '))
    return [feedback]

n_actions = env.action_space.n
# Get the number of state observations
if gym.__version__[:4] == '0.26':
    state, _ = env.reset()
elif gym.__version__[:4] == '0.25':
    state, _ = env.reset(return_info=True)
n_observations = len(state)

RM = RewardModel(n_observations,1).cuda()


training_samples = []

for t in range(50):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # predicted_reward = RM(torch.tensor(observation, device=device, dtype=torch.float32).reshape(1, n_observations),
    #                     torch.tensor(action, device=device, dtype=torch.float32).reshape(1, 1))
    print('action_taken_', t, ' :',action)
    h_inp = get_human_feedback()
    training_samples.append([[observation, action], h_inp])
    if terminated:
        print(f'failed ta time {_}')
        observation, info = env.reset()

if SAVE:
  np.save('reward_training.npy', np.array(training_samples))
if TRAIN:
  # training_samples = np.load('reward_training.npy', allow_pickle=True)
  print('free your hands, Reward Model training started...')

  EPOCHS = 15
  BATCH_SIZE = 4

  RM_optimizer = optim.SGD(RM.parameters(), lr=1e-3)

  class reward_dataset(Dataset):
      def __init__(self) -> None:
          super().__init__()

      def __len__(self):
          return len(training_samples)

      def __getitem__(self, index):
          state, action = training_samples[index][0]
          state = torch.tensor(state, dtype=torch.float32).reshape(1, 4)
          action = torch.tensor(action, dtype=torch.float32).reshape(1, 1)
          r = torch.tensor(training_samples[index][1], dtype=torch.float32).reshape(1, 1)
          return state, action, r

  data = reward_dataset()
  loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

  RM.train()

  for e in range(EPOCHS):
      loop = tqdm(loader)
      for ix, (s, a, r) in enumerate(loop):
          s = s.cuda()
          a = a.cuda()
          r = r.cuda()

          pred = RM(s.reshape(-1, 4), a.reshape(-1, 1))
          l = F.mse_loss(pred, r)

          RM_optimizer.zero_grad()
          l.backward()
          torch.nn.utils.clip_grad_norm_(RM.parameters(), 0.5)
          RM_optimizer.step()
          loop.set_postfix(loss=l.item())

  torch.save(RM, r'pre-trained-reward.ckpt')
