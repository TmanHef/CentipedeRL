import numpy as np


# a class modelling the game
# player 0 is assumed to be the one who starts
class Centipede:
    numOfLegs = None
    numOfRounds = None

    # payoff is a 2x(NumOfLegs + 1)  matrix (+ 1 in case PASS was the action in the last leg),
    # row 0 represents the payoff of player 0 at each stage
    # symmetrically for row 1
    payoffs = None

    def __init__(self, legs, rounds):
        self.numOfLegs = legs
        self.numOfRounds = rounds

        self.payoffs = np.matrix(np.zeros((2, legs + 1)))
        for i in range(1, legs + 1):
            if (i % 2) == 0:
                self.payoffs[0, i] = 2 ^ (i - 1)
                self.payoffs[1, i] = 2 ^ (i + 1)
            else:
                self.payoffs[1, i] = 2 ^ (i - 1)
                self.payoffs[0, i] = 2 ^ (i + 1)

    def get_next_state(self, leg, round, action):

        # if TAKE was made or PASS was made in the last leg - return next round state
        if action == 'TAKE' | leg == self.numOfLegs:
            return 1, round + 1

        else:
            return leg + 1, round

    def get_payoff_player0(self, leg):
        return self.payoffs[0, leg]

    def get_payoff_player1(self, leg):
        return self.payoffs[1, leg]
