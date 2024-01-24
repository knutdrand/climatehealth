from ..modelling.compartementalized_model import state

@state
class HumanState(state):
    S: float
    E: float
    I: float
    R: float

@state
class MosquitoState(state):
    pass
