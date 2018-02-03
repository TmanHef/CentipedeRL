from Centipede import Centipede
from Agents.PassAgent import PassAgent
from Agents.TakeAgent import TakeAgent
from Agents.RLAgent import RLAgent
from Constants import take_action
import numpy as np


# SETUP ---------------------------------------------------

legs = 10
rounds = 10
current_leg = 1
current_round = 1
history = []
Q = np.matrix(np.zeros([rounds * legs, 2]))
centipede = Centipede(legs, rounds)
agent0 = PassAgent()
agent1 = RLAgent(rounds, legs, Q)

# learning rate
alpha = 0.95

# exploration rate
gamma = 0.8

agent1.set_alpha(alpha)
agent1.set_gamma(gamma)

# FUNCTIONS -----------------------------------------------

def swap_starting_agent():
    global agent0
    global agent1
    temp = agent1
    agent1 = agent0
    agent0 = temp

# TRAINING ------------------------------------------------

while current_round <= rounds:
    payoff_agent0 = centipede.get_payoff_player0(current_leg)
    payoff_agent1 = centipede.get_payoff_player1(current_leg)
    last_round = current_round
    last_leg = current_leg

    if (current_leg % 2) == 0:
        action = agent1.decide(current_leg, current_round)

    else:
        action = agent0.decide(current_leg, current_round)

    # get the next state according to the current state and the action
    current_leg, current_round = centipede.get_next_state(current_leg, current_round, action)
    if last_round != current_round:

        # take into account the action in the last leg
        if action == take_action:
            history.append(last_leg)
        else:
            history.append(last_leg + 1)

        agent0.add_points(payoff_agent0)
        agent1.add_points(payoff_agent1)
        swap_starting_agent()

# RESULTS ------------------------------------------------

agent0.print_state()
agent1.print_state()
print('history: ', history)