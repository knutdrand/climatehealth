import numpy as np
import particles
from particles import state_space_models as ssm, mcmc, distributions as dists
from climatehealth.modelling.compartementalized_model import SIRState, SIRModel, make_ssm_class, check_capasity, \
    check_model_capasity
from climatehealth.modelling.dengue_sir_model import make_ssm_class as make_ssm
import plotly.express as px


def test_make_ssm_class():
    ssm = make_ssm_class(SIRModel, 1000)(beta=0.5, gamma=0.01)
    assert ssm is not None
    x, y = ssm.simulate(20)
    px.line(y).show()

def test_bootstrap():
    model = make_ssm_class(SIRModel, 1000)(beta=0.5, gamma=0.01)
    x, y = model.simulate(20)
    alg = particles.SMC(fk=ssm.Bootstrap(ssm=model, data=y), N=200)
    print(alg.run())

def test_capasity():
    check_capasity(SIRModel, 1000)
    # ssm = make_ssm_class(SIRModel, 10000)(beta=0.5, gamma=0.01)


def test_dengue():
    cls = make_ssm((np.arange(100) % 12) + 20, 10000)
    priors = {'gamma': dists.Beta(0.5, 10), 'beta': dists.Normal(0.5, 10), 'beta_0': dists.Normal(0, 10)}
    check_model_capasity(cls, cls(beta_0=1), priors, T=100)


