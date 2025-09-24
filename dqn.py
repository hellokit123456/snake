# dqn.py
# Minimal DQN implementation (PyTorch)
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Experience = namedtuple('Experience', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Experience(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        # obs_shape: (H,W) grid. We'll flatten.
        in_dim = obs_shape[0]*obs_shape[1]
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        # x: batch x H x W
        batch = x.view(x.size(0), -1)
        return self.net(batch)

class DQNAgent:
    def __init__(self, obs_shape, n_actions, device='cpu', lr=1e-3, gamma=0.99):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.policy_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = ReplayBuffer(20000)
        self.steps = 0
        self.update_target_every = 1000

    def select_action(self, state, epsilon=0.1):
        # state is numpy array H x W -> convert to tensor
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t)
            return int(q.argmax().cpu().numpy())

    def push(self, *args):
        self.replay.push(*args)

    def train_step(self, batch_size=64):
        if len(self.replay) < batch_size:
            return 0.0
        batch = self.replay.sample(batch_size)
        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = nn.functional.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
