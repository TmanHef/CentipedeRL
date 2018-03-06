class Agent:

    _total_points = None
    gamma = None
    alpha = None
    epsilon = None

    def __init__(self):
        self._total_points = 0

    def add_points(self, points):
        self._total_points += points

    def reset_points(self):
        self._total_points = 0

    def print_state(self):
        print(self.__class__.__name__, ' got: ', self._total_points)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # for comfort (RLAgent overrides this)
    def decay_alpha(self, factor):
        return

    # for comfort (RLAgent overrides this)
    def decay_epsilon(self, factor):
        return

    # for comfort (RLAgent overrides this)
    def update(self, leg, round, action, next_leg, next_round, reward):
        return