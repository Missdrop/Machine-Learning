from collections import deque
import numpy as np
import random
import torch


class MemoryReplay(object):
    def __init__(self, capacity, device):
        self.pool = deque([], maxlen=capacity)
        self.device = device

    def push(self, transition):
        self.pool.extend(transition)

    def sample(self, batch_size):
        data = random.sample(self.pool, batch_size)

        state = (
            torch.FloatTensor(np.array([i[0] for i in data]))
            .reshape(-1, 3)
            .to(self.device)
        )
        action = (
            torch.FloatTensor(np.array([i[1] for i in data]))
            .reshape(-1, 1)
            .to(self.device)
        )
        reward = (
            torch.FloatTensor(np.array([i[2] for i in data]))
            .reshape(-1, 1)
            .to(self.device)
        )
        next_state = (
            torch.FloatTensor(np.array([i[3] for i in data]))
            .reshape(-1, 3)
            .to(self.device)
        )
        over = (
            torch.LongTensor(np.array([i[4] for i in data]))
            .reshape(-1, 1)
            .to(self.device)
        )

        return state, action, reward, next_state, over

    def __len__(self):
        return len(self.pool)
