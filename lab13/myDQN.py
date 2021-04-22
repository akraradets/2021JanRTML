import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import queue
import numpy as np

class Memory():
    def __init__(self, buffer_size):
        self.memory = queue.Queue()
        self.buffer_size = buffer_size
    def get_memory(self):
        return list(self.memory.queue)

    def get_memory_random(self):
        index = np.random.randint(self.memory.qsize(), size=1)
        return self.memory.queue[index[0]]

    def add_memory(self, s_t, a_t, r_t, s_t1):
        temp = (s_t, a_t, r_t, s_t1)
        if(self.memory.qsize() > self.buffer_size):
            self.memory.get()
        self.memory.put(temp)
        return True
    def reset(self):
        self.memory = queue.Queue()


class DQN(nn.Module):
    def __init__(self, number_action):
        super(DQN, self).__init__()
        # we would have just 27 inputs and 9 outputs. Two fully connected layers of 10 units each would give us 10x28+10x11+9x11=489 parameters
        self.fc = nn.Linear(in_features=27, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=number_action)

    def forward(self, state):
        # state = torch.tensor(state.reshape(-1).astype(float), requires_grad=True).float()
        # print(state)
        # # state = torch.from_numpy(state, requires_grad=True)
        # # state = state.reshape(-1)
        # state.requires_grad_(True)

        # print(state.shape, type(state.float()),state)
        out = self.fc(state)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

