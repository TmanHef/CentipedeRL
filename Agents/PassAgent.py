from Agent import Agent
from Constants import pass_action


class PassAgent(Agent):

    def decide(self, leg, round):
        return pass_action
