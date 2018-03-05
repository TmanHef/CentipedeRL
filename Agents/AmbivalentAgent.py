from Agent import Agent
from Constants import pass_action
from Constants import take_action
import numpy as np


class AmbivalentAgent(Agent):

    def decide(self, leg, round):
        if np.random.random() > 0.5:
            return pass_action
        return take_action