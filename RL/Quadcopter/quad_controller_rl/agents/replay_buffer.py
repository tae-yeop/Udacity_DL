import random
from collections import namedtuple

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:

    def __init__(self, size=1000):
        self.size = size
        self.memory = []
        self.idx = 0

    def __len__(self):
        return len(self.memory)

        
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size = 64):
        return random.sample(self.memory, k = batch_size)


