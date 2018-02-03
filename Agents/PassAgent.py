from Agent import Agent


class PassAgent(Agent):

    def decide(self, leg, round):
        return 'PASS'
