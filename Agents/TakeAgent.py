from Agent import Agent
from Constants import take_action


class TakeAgent(Agent):

    def decide(self, leg, round):
        return take_action
