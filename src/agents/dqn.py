# src/agents/dqn.py
from dataclasses import dataclass
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int):
        # inital vlaue is empty
        # maxlen is capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        # return a list
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.array(s, dtype=np.int64),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.int64),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNNet(nn.Module):
    """
    Single Q-network: input = state_id (int), output = Q-values for each action.
    Use embedding for discrete states like Taxi-v3.
    """
    def __init__(self, n_states: int, n_actions: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_states, emb_dim)
        # two hidden layers
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        # state_ids: (B,) int64
        x = self.emb(state_ids)          # (B, emb_dim)
        # output of a feedforward neural network
        q = self.mlp(x)                  # (B, n_actions)
        return q


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3 # 0.01
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 2_000          # warm-up before training
    train_every: int = 1             # update frequency (steps)
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000    # linear decay


class DQNAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: DQNConfig = DQNConfig()):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        # self.qnet.forward(s)
        self.qnet = DQNNet(n_states, n_actions)
        self.optim = optim.Adam(self.qnet.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)

        self.total_steps = 0

    def epsilon(self) -> float:
        # linear decay
        t = min(self.total_steps, self.cfg.eps_decay_steps)
        frac = t / self.cfg.eps_decay_steps
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, state: int, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)

        s = torch.tensor([state], dtype=torch.long)
        with torch.no_grad():
            q = self.qnet(s)  # (1, n_actions)
        return int(torch.argmax(q, dim=1).item())

    def remember(self, s: int, a: int, r: float, s2: int, done: bool):
        self.buffer.push(s, a, r, s2, done)

    def learn_step(self):
        cfg = self.cfg
        # if expience is too little
        if len(self.buffer) < cfg.min_buffer:
            return None

        if self.total_steps % cfg.train_every != 0:
            return None

        # s.shape=s2.shape=(B,)
        s, a, r, s2, done = self.buffer.sample(cfg.batch_size)

        s = torch.tensor(s, dtype=torch.long)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.long)
        done = torch.tensor(done, dtype=torch.float32)

        # Q(s,a) compute Q for every s in batch
        q_all = self.qnet(s)  # (B64,A6)
        # method gather(dim, index)
        # unsqueeze(1),(B,1)
        # gather(B,1)
        # squeeze (B,)
        q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1) 

        # target: y = r + gamma * max_a' Q(s', a')
        # single NN: use same qnet, but NO grad through target
        with torch.no_grad():
            q_next = self.qnet(s2)                 
            max_q_next = q_next.max(dim=1).values  #（B,）
            # target
            y = r + cfg.gamma * (1.0 - done) * max_q_next #(B,)

        loss = nn.MSELoss()(q_sa, y)

        self.optim.zero_grad()
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.qnet.parameters(), cfg.grad_clip)
        # θ←θ−α⋅∇θ​J(θ)
        self.optim.step()
        return float(loss.item())

    def step(self):
        self.total_steps += 1
