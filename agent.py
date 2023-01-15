import numpy as np
from collections import defaultdict


class QAgent(object):
    def __init__(self):
        self.action_space = [0, 1]     # 0: go left   1: go right
        self.V = []
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.gamma = 0.99    # discounted factor
        self.alpha = 0.5     # learning rate
        self.epsilon = 0.01

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[state])]
        return action

    def train(self, state, action, next_state, reward):
        next_action = self.action_space[np.argmax(self.Q[next_state])]
        td_target = reward + self.gamma * self.Q[next_state][next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
