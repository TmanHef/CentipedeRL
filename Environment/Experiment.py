from CentipedeEnv import Centipede
from Agents.PassAgent import PassAgent
from Agents.TakeAgent import TakeAgent
from Agents.AmbivalentAgent import AmbivalentAgent
from Agents.RLAgent import RLAgent
from Constants import pass_action
import numpy as np
import matplotlib.pyplot as plt


# SETUP ---------------------------------------------------

training_games = 10000
legs = 10
rounds = 10

Q1 = np.random.random_sample([rounds * legs, 2]) * 10
Q1 = np.matrix(Q1)

Q2 = np.random.random_sample([rounds * legs, 2]) * 10
Q2 = np.matrix(Q2)

print('initial Q:')
print(Q1)
agent = RLAgent(rounds, legs, Q1)
rl_agent = RLAgent(rounds, legs, Q2)

# learning rate
alpha = 0.999

# discount factor
gamma = 0.9

# exploration rate
epsilon = 0.8

rl_agent.set_alpha(alpha)
rl_agent.set_gamma(gamma)
rl_agent.set_epsilon(epsilon)

agent.set_alpha(alpha)
agent.set_gamma(gamma)
agent.set_epsilon(epsilon)

centipede = Centipede(legs, rounds, agent)
current_leg, current_round = centipede.get_first_state()


# FUNCTIONS -----------------------------------------------


def reset_game(game):
    global current_round
    global current_leg
    global agent
    global rl_agent
    global sum_off_pass
    global APPG
    global alpha
    global epsilon
    agent.reset_points()
    rl_agent.reset_points()

    # APPG.append(sum_off_pass)
    # sum_off_pass = 0

    rl_agent.set_alpha(alpha ** (1 + (0.2 * game)))
    rl_agent.set_epsilon(epsilon ** (1 + (0.01 * game)))

    centipede.agent.set_alpha(alpha ** (1 + (0.2 * game)))
    centipede.agent.set_epsilon(epsilon ** (1 + (0.01 * game)))

    rl_agent.new_game_reset()
    centipede.agent.new_game_reset()

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

        # has to be calculated before get_next_state - because of the starting agent boolean in reward calc
        reward = centipede.calculate_reward(action, last_leg)

        # get the next state according to the current state and the action
        current_leg, current_round = centipede.get_next_state(current_leg, current_round, action)

        if current_round <= rounds:

            # update the Q matrix of the agent
            rl_agent.update(last_leg, last_round, action, current_leg, current_round, reward)

        if current_round != last_round:
            rl_agent.add_points(rl_payoff)

# TRAINING ------------------------------------------------


for game in range(0, training_games):
    print('game: ', game)
    run_game()
    if game <= training_games:
        reset_game(game)


# RESULTS ------------------------------------------------
print('alpha = ', rl_agent.alpha)
print('epsilon = ', rl_agent.epsilon)
rl_agent.print_state()
plt.figure(1)
plt.ylabel('# of PASS actions taken per game')
plt.xlabel('# of training game')
plt.plot(rl_agent.pass_per_game)
plt.savefig('Output/APPG1.png')

plt.figure(2)
plt.ylabel('# of PASS actions taken per game')
plt.xlabel('# of training game')
plt.plot(centipede.agent.pass_per_game)
plt.savefig('Output/APPG2.png')

for i in range(0, rounds):
    plt.figure(i + 3)
    plt.ylabel('Round ' + str(i + 1) + ' Finished on Leg')
    plt.xlabel('# of training game')
    plt.plot(centipede.legs_history[i])
    plt.savefig('Output/RFL' + str(i + 1) + '.png')
