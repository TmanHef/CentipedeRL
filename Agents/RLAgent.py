from Agent import Agent
import numpy as np


class RLAgent(Agent):

    Q = None
    num_rounds = None
    num_legs = None
    gamma = None
    alpha = None
    epsilon = None

    def __init__(self, rounds, legs, Q):
        Agent.__init__(self)
        self.num_rounds = rounds
        self.num_legs = legs
        self.Q = Q

        # discount factor
        self.gamma = 0.8

        # learning rate
        self.alpha = 0.95

        # exploration rate
        self.epsilon = 0.4

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # this is the pai function
    def decide(self, leg, round):
        state_entry_index = round * self.num_legs + leg
        action_index = np.where(self.Q[state_entry_index] == np.max(self.Q[state_entry_index]))[1]

        # choose according to policy
        if np.random.random() > self. epsilon:
            # check if there is more than one action possible
            if action_index.shape[0] > 1:
                max_index = int(np.random.choice(action_index, size = 1))
            else:
                max_index = int(action_index)
        # explore
        else:
            max_index = np.random.random_integers(0, 1)

        # decay epsilon
        self.epsilon = self.epsilon * self.alpha

        return max_index

    # updates the Q matrix
    def update(self, leg, round, action, next_leg, next_round, reward):

        # s
        state_entry_index = round * self.num_legs + leg

        # s'
        next_state_entry_index = next_round * self.num_legs + next_leg

        # will get a'
        max_index = np.where(self.Q[next_state_entry_index] == np.max(self.Q[next_state_entry_index]))[1]

        # check if there is more than one action possible
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)

        # Q'[s',a']
        max_value = self.Q[next_state_entry_index, max_index]

        # Q[s,a] <-- (1 - alpha) * Q[s,a] + alpha * (reward + gamma * Q'[s',a'])
        self.Q[state_entry_index, action] = (1 - self.alpha) * self.Q[state_entry_index, action] + self.alpha * (reward + self.gamma * max_value)

        # decay alpha
        self.alpha = self.alpha ** 2

    # override
    def print_state(self):
        print(self.__class__.__name__, ' got: ', self._total_points)
        print('Q matrix is:')
        print(self.Q)