import numpy as np

# a class modelling the game
# player 0 is assumed to be the one who starts
class Centipede:

    numOfLegs = None
    numOfRounds = None

    # payoff is a 2xNumOfLegs matrix,
    # row 0 represents the payoff of player 0 at each stage
    # symmetrically for row 1
    payoffs = None
    player0 = None
    player1 = None

    def __init__(self, legs, rounds, playerA, playerB):
        self.numOfLegs = legs
        self.numOfRounds = rounds

        if np.random.random_sample() > 0.5:
            self.player0 = playerA
            self.player1 = playerB
        else:
            self.player0 = playerB
            self.player1 = playerA


        self.payoffs = np.matrix(np.zeroes((2, legs)))
        for i in range(1, legs + 1):
            if (i % 2) == 0:
                self.payoffs[0, i] = 2 ^ (i - 1)
                self.payoffs[1, i] = 2 ^ (i + 1)
            else:
                self.payoffs[1, i] = 2 ^ (i - 1)
                self.payoffs[0, i] = 2 ^ (i + 1)

