from Agent import Agent
import numpy as np
from Constants import pass_action


class RLAgent(Agent):

    Q = None
    num_rounds = None
    num_legs = None
    num_of_pass = None


    def __init__(self, rounds, legs, Q):
        Agent.__init__(self)
        self.num_rounds = rounds
        self.num_legs = legs
        self.num_of_pass = 0
        self.Q = Q

        # discount factor
        self.gamma = 0.8

        # learning rate
        self.alpha = 0.95

        # exploration rate
        self.epsilon = 0.4



    # this is the pi function
    def decide(self, leg, round):
        state_entry_index = (round - 1) * self.num_legs + (leg - 1)
        action_index = np.where(self.Q[state_entry_index] == np.max(self.Q[state_entry_index]))[1]

        # choose according to policy
        if np.random.random() > self.epsilon:
            # check if there is more than one action possible
            if action_index.shape[0] > 1:
                max_index = int(np.random.choice(action_index, size = 1))
            else:
                max_index = int(action_index)
        # explore
        else:
            max_index = np.random.random_integers(0, 1)

        #print('RL chose: ', max_index)

        if max_index == pass_action:
            self.num_of_pass += 1

        return max_index

    # updates the Q matrix
    def update(self, leg, round, action, next_leg, next_round, reward):

        # s
        state_entry_index = (round - 1) * self.num_legs + (leg - 1)

        if (state_entry_index == 90):
            print('hiii')

        # s'
        next_state_entry_index = (next_round - 1) * self.num_legs + (next_leg - 1)

        # will get a'
        max_index = np.where(self.Q[next_state_entry_index] == np.max(self.Q[next_state_entry_index]))[1]

        # check if there is more than one action possible
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)

        # Q'[s',a']
        max_value = self.Q[next_state_entry_index, max_index]

        #print('before: Q[', state_entry_index, ',', action, '] = ', self.Q[state_entry_index, action], ' alpha = ', self.alpha)

        # Q[s,a] <-- (1 - alpha) * Q[s,a] + alpha * (reward + gamma * Q'[s',a'])
        self.Q[state_entry_index, action] = (1 - self.alpha) * self.Q[state_entry_index, action] + self.alpha * (reward + self.gamma * max_value)

        #print('after:  Q[', state_entry_index, ',', action, '] = ', self.Q[state_entry_index, action], ' alpha = ', self.alpha)

    # override
    def print_state(self):
        print(self.__class__.__name__, ' got: ', self._total_points)
        print('Q matrix is:')
        print(self.Q)

    # override
    def decay_alpha(self, factor):
        self.alpha = 0.95 ** factor

    # for comfort (RLAgent overrides this)
    def decay_epsilon(self, factor):
        self.epsilon = 0.95 ** factor

    def new_game_reset(self):
        self.num_of_pass = 0
