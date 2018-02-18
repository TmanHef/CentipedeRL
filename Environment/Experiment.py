from CentipedeEnv import Centipede
from Agents.PassAgent import PassAgent
from Agents.TakeAgent import TakeAgent
from Agents.RLAgent import RLAgent
from Constants import pass_action
import numpy as np
import matplotlib.pyplot as plt


# SETUP ---------------------------------------------------

training_games = 5000
legs = 10
rounds = 10

# Amount of Pass Per Game
APPG = []
sum_off_pass = 0

# Q = np.random.random_sample([rounds * legs, 2]) * 10
# Q = np.matrix(Q)

Q = np.matrix(np.zeros([rounds * legs, 2]))
print('initial Q:')
print(Q)
pass_agent = PassAgent()
rl_agent = RLAgent(rounds, legs, Q)
centipede = Centipede(legs, rounds, pass_agent)
current_leg, current_round = centipede.get_first_state()

# learning rate
alpha = 0.999

# discount factor
gamma = 0.85

# exploration rate
epsilon = 0.09

rl_agent.set_alpha(alpha)
rl_agent.set_gamma(gamma)
rl_agent.set_epsilon(epsilon)

# FUNCTIONS -----------------------------------------------


def reset_game(game):
    global current_round
    global current_leg
    global pass_agent
    global rl_agent
    global sum_off_pass
    global APPG
    global alpha
    global epsilon
    pass_agent.reset_points()
    rl_agent.reset_points()

    APPG.append(sum_off_pass)
    sum_off_pass = 0

    rl_agent.set_alpha(alpha ** (1 + (0.2 * game)))
    #rl_agent.set_epsilon(epsilon ** (1 + (0.1 * game)))

    centipede.reset_game()
    current_leg, current_round = centipede.get_first_state()


def run_game():
    global current_round
    global current_leg
    global rounds
    global legs
    global rl_agent
    global sum_off_pass

    while current_round <= rounds:
        last_round = current_round
        last_leg = current_leg
        rl_payoff = centipede.get_payoff(current_leg)
        action = rl_agent.decide(current_leg, current_round)

        if action == pass_action:
            sum_off_pass += 1

        # has to be calculated before get_next_state - because of the starting agent boolean in reward calc
        reward = centipede.calculate_reward(action, last_leg)

        # get the next state according to the current state and the action
        current_leg, current_round = centipede.get_next_state(current_leg, current_round, action)

        if current_round <= rounds:
            if current_round == rounds:
                print('hi')

            # update the Q matrix of the agent
            rl_agent.update(last_leg, last_round, action, current_leg, current_round, reward)

        if current_round != last_round:
            rl_agent.add_points(rl_payoff)

# TRAINING ------------------------------------------------


for game in range(0, training_games):
    #print('game: ', game)
    run_game()
    if game <= training_games:
        reset_game(game)


# RESULTS ------------------------------------------------

rl_agent.print_state()
# plt.figure(None, (7, 2.5), 500)
plt.plot(APPG)
plt.show()
