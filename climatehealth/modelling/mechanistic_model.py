import dataclasses

import numpy as np

state = dataclasses.dataclass
distribution=  dataclasses.dataclass


@state
class DengueState:
    accumulated_rain: float


@distribution
class StateDistribution:
    denge_state: DengueState
    dengue_counts: int
    rainfall: float
    run_off: float

    def sample(self, n):
        return DengueState(np.max(self.dengue_state.accumulated_rain + self.rainfall - self.run_off, 0))


@distribution
class DengueCasesDistribution:
    dengue_state: DengueState

    def sample(self, n):
        return Poisson()


