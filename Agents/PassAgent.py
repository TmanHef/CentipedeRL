class PassAgent:
    _total_points = None

    def __init__(self):
        self._total_points = 0

    def decide(self, leg, round):
        return 'PASS'

    def add_points(self, points):
        self._total_points += points

    def reset_points(self):
        self._total_points = 0

    def print_state(self):
        print(self.__class__, ' got: ', self._total_points)