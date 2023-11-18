from dataclasses import dataclass

distribution = dataclass


@distribution
class DengueState:
    pass


@distribution
class DengueCountModel:
    dengue_state: DengueState


@distribution
class DengueStateModel:
    dengue_state: DengueState
    rainfall: float
    theta: float

    def log_prob(self, dengue_state: DengueState):
        return Normal(self.dengue_state + self.theta * self.rainfall, 0.2).log_prob(dengue_state)

    def sample(self, n):
        return Normal(self.dengue_state + self.theta * self.rainfall, 0.2).sample(n)


class SimpleDengueCountModel:
    dengue_state: DengueState

    def log_prob(self, dengue_count: int):
        return Poisson(self.dengue_state**2).log_prob(dengue_count)

    def sample(self, n):
        return Poisson(self.dengue_state**2).sample(n)