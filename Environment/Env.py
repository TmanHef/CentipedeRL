from Centipede import Centipede
from Agents.PassAgent import PassAgent
from Agents.TakeAgent import TakeAgent

# SETUP ---------------------------------------------------

legs = 10
rounds = 10
currentLeg = 1
currentRound = 1
history = []
centipede = Centipede(legs, rounds)
agent0 = PassAgent()
agent1 = TakeAgent()

# FUNCTIONS -----------------------------------------------

def swap_starting_agent():
    temp = agent1
    agent1 = agent0
    agent0 = temp

# TRAINING ------------------------------------------------

while currentRound <= rounds:
    payoff_agent0 = centipede.get_payoff_player0(currentLeg)
    payoff_agent1 = centipede.get_payoff_player1(currentLeg)
    last_round = currentRound
    last_leg = currentLeg

    if (currentLeg % 2) == 0:
        action = agent1.decide(currentLeg, currentRound)

    else:
        action = agent0.decide(currentLeg, currentRound)

    currentLeg, currentRound = centipede.get_next_state(currentLeg, currentRound, action)
    if last_round != currentRound:
        history.append(last_leg)
        agent0.add_points(payoff_agent0)
        agent1.add_points(payoff_agent1)
        swap_starting_agent()