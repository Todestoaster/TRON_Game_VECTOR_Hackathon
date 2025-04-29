import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Speichere eine neue Erfahrung."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # Platzhalter erweitern

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Ringpuffer

    def sample(self, batch_size):
        """Gebe ein zufälliges Batch zurück."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
