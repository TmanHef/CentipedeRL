import numpy as np
from Constants import take_action


# a class modelling the game
class Centipede:
    numOfLegs = None
    numOfRounds = None
    agent = None
    env_agent_starts = None

    # will store the last state the env agent was in - [leg, round, action]
    last_env_state = None

    last_leg = None
    legs_history = None

    # payoff is a 2x(NumOfLegs + 1)  matrix (+ 1 in case PASS was the action in the last leg),
    # row 0 represents the payoff of starting player
    # row 1 for the other player
    payoffs = None

    def __init__(self, legs, rounds, agent):
        self.numOfLegs = legs
        self.numOfRounds = rounds
        self.agent = agent
        self.env_agent_starts = False
        self.last_leg = 1
        self.legs_history = []

        for i in range(0, rounds):
            self.legs_history.append([])

        # legs + 1 to support pass action on last leg
        self.payoffs = np.matrix(np.zeros((2, legs + 1)))
        for i in range(0, legs + 1):
            if (i % 2) != 0:
                self.payoffs[0, i] = 2 ** (i)
                self.payoffs[1, i] = 2 ** (i + 2)
            else:
                self.payoffs[1, i] = 2 ** (i)
                self.payoffs[0, i] = 2 ** (i + 2)

    def get_first_state(self):
        # if self.env_agent_starts:
        #     env_action = self.agent.decide(1, 1)
        #     return self._next_state_helper(1, 1, env_action)
        return 1, 1


    def _update_env_Q(self, next_leg, next_round):
        last_env_leg = self.last_env_state[0]
        last_env_round = self.last_env_state[1]
        last_env_action = self.last_env_state[2]

        # calculate the reward for the last action
        reward = self._calculate_reward_internal(last_env_action, last_env_leg)

        self.agent.update(last_env_leg, last_env_round, last_env_action, next_leg, next_round, reward)

    def get_next_state(self, leg, round, action):

        # get the state for env agent
        new_leg, new_round = self._next_state_helper(leg, round, action)

        if new_round == round:
            # env agent makes a choice and learning agent gets next state
            env_action = self.agent.decide(new_leg, new_round)

            # if this is not the first decide the agent does - need to update the Q matrix
            if self.last_env_state is not None:
                self._update_env_Q(new_leg, new_round)

            # now it's safe to update the last env state
            self.last_env_state = [new_leg, new_round, env_action]

            round_before_env_action = new_round

            # get the next state according to env agent action
            new_leg, new_round = self._next_state_helper(new_leg, new_round, env_action)

            # the env agent might cause a new round to start - need to check if he starts in that round
            # if it is a new round AND the env agent needs to start - let him start
            if (new_round != round_before_env_action) and self.env_agent_starts:
                env_action = self.agent.decide(new_leg, new_round)

                # if this is not the first decide the agent does - need to update the Q matrix
                if self.last_env_state is not None:
                    self._update_env_Q(new_leg, new_round)

                # now it's safe to update the last env state
                self.last_env_state = [new_leg, new_round, env_action]

                return self._next_state_helper(new_leg, new_round, env_action)

            # else return the state to rl agent
            return new_leg, new_round


        # else it is a new round
        # new round starting and it's the env agent's turn to start
        if self.env_agent_starts:

            # env agent makes a choice and learning agent gets next state
            env_action = self.agent.decide(new_leg, new_round)

            # if this is not the first decide the agent does - need to update the Q matrix
            if self.last_env_state is not None:
                self._update_env_Q(new_leg, new_round)

            # now it's safe to update the last env state
            self.last_env_state = [new_leg, new_round, env_action]

            # return the state for learning agent
            return self._next_state_helper(new_leg, new_round, env_action)
        else:
            return new_leg, new_round

    def _next_state_helper(self, leg, round, action):
        # if TAKE was made or PASS was made in the last leg - return next round state
        if action == take_action or leg == self.numOfLegs:

            # a round ended
            self.legs_history[round - 1].append(leg)
            self.env_agent_starts = not self.env_agent_starts
            return 1, round + 1

        else:
            return leg + 1, round

    # assumes only external call (from experiment)
    def calculate_reward(self, action, leg):
        # reward for take is the payoff
        if action == take_action:
            reward_leg = leg

        # reward for pass is the payoff of the next leg
        else:
            reward_leg = leg + 1

        # if env agent didn't start, external did
        return self._get_payoff(reward_leg, not self.env_agent_starts)

    # assumes only internal call
    def _calculate_reward_internal(self, action, leg):
        # reward for take is the payoff
        if action == take_action:
            reward_leg = leg

        # reward for pass is the payoff of the next leg
        else:
            reward_leg = leg + 1

        return self._get_payoff(reward_leg, self.env_agent_starts)

    def _get_payoff(self, leg, did_start):
        if did_start:
            return self.payoffs[0, leg - 1]
        return self.payoffs[1, leg - 1]

    # assumes only external call (from experiment)
    def get_payoff(self, leg):
        if self.env_agent_starts:
            return self.payoffs[1, leg - 1]
        return self.payoffs[0, leg - 1]

    def reset_game(self):
        self.last_leg = 1
        self.env_agent_starts = False
        self.last_env_state = None

