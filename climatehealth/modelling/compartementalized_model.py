import dataclasses
from collections import OrderedDict
import arviz as az
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import scipy.stats
from particles import state_space_models as ssm, mcmc, distributions as dists
from dataclasses import dataclass
from npstructures import npdataclass

state = npdataclass

parameters = dataclass
statemodel = dataclass


@state
class SIRState:
    S: float
    I: float
    R: float

    @property
    def shape(self):
        return (len(self), )


@statemodel
class SIRModel:
    state: SIRState
    beta: float = 0.1
    gamma: float = 0.05
    epsilon: float = 0.00
    dim=1

    def dS(self):
        return -self.beta * self.state.S * self.state.I + self.epsilon * self.state.R

    def dI(self):
        return self.beta * self.state.S * self.state.I - self.gamma * self.state.I

    def dR(self):
        return self.gamma * self.state.I-self.epsilon * self.state.R

    def next(self):
        next_state = self.next_state()
        self.state = next_state

    def next_state(self):
        cls = self.state.__class__
        return cls(
            *(getattr(self.state, field.name) + getattr(self, f"d{field.name}")() for field in
              dataclasses.fields(self.state)))

    def rvs(self, size=None):
        beta = self.beta.rvs(size)
        return dataclasses.replace(self, beta=beta).next_state()

    def logpdf(self, state):
        dS = state.S-self.state.S
        beta = -dS/(self.state.S*self.state.I)
        return self.beta.logpdf(beta)

def get_state_distribution(state_class: type):
    class StateDist:
        def __init__(self, alpha):
            self._dist = scipy.stats.dirichlet(np.array(alpha))
            self.dim = 1

        def logpdf(self, state: state_class):
            return self._dist.logpdf(np.array([getattr(field.name) for field in dataclasses.fields(state)]))

        def rvs(self, size=None):
            rvs = self._dist.rvs(size)
            return state_class(*rvs.T)

    StateDist.__name__ = f"{state_class.__name__}Dist"
    StateDist.__qualname__ = f"{state_class.__qualname__}Dist"
    return StateDist


class StateDistribution:
    def __class_getitem__(cls, item):
        return get_state_distribution(item)


class DeltaDistribution:
    def __init__(self, value):
        self.value = value
        self.cls = value.__class__

    def logpdf(self, value):
        return value == self.value

    def rvs(self, size):
        if size == 1:
            return self.value
        return self.cls(*(np.full(size, getattr(self.value, field.name)) for field in dataclasses.fields(self.value)))
        # return np.full(size, self.value)


def make_ssm_class(statemodel, N):

    class SIRSSM(ssm.StateSpaceModel):
        default_params = {field.name: field.default for field in dataclasses.fields(statemodel) if field.name != 'state'}

        def PX0(self):
            return StateDistribution[SIRState](np.array([0.33, 0.33, 0.34]))

        def PX(self, t, xp):
            print(xp)
            params_ = {name: getattr(self, name) for name in self.default_params}
            return DeltaDistribution(statemodel(xp, **params_).next_state())

        def PY(self, t, xp, x):
            return dists.Poisson(x.I * N)

    return SIRSSM


def check_capasity(statemodel, N):
    cls = make_ssm_class(statemodel, N)
    priors = {'beta': dists.Beta(1, 10), 'gamma': dists.Beta(0.5, 10)}
    check_model_capasity(cls, cls(beta=0.5, gamma=0.01), priors)


def check_model_capasity(cls, my_model, priors,  T=24):
    xs, y = my_model.simulate(T)
    plt.plot([x.I*100000 for x in xs], '.')
    plt.plot(y, '-')
    plt.show()
    # px.plot(y).show()
    # px.line(x.I).show()
    prior = dists.StructDist(OrderedDict(priors))
    niter = 1000
    my_pmmh = mcmc.PMMH(ssm_cls=cls,
                        prior=prior, data=y, Nx=500,
                        niter=niter)
    my_pmmh.run();  # may take several seconds
    for name in my_model.default_params:
        plt.plot(my_pmmh.chain.theta[name][niter//3:], '-')
        plt.title(name)
        plt.hlines(getattr(my_model, name), 0, niter-niter//3)
        plt.show()
        # px.histogram(my_pmmh.chain.theta[name][3000:], title=name).show()
    new_model = plot_posteriors(T, cls, my_model, my_pmmh, niter, y)
    #data = az.convert_to_inference_data(posterior_samples)
    # az.plot_posterior(data, kind="timeseries", figsize=(10, 6))
    # plt.show()

    #x, new_y = new_model.simulate(T)
    #plt.plot(new_y, '-')
    # plt.plot(y, '-')
    # plt.show()
    # px.line(new_y).show()


def plot_posteriors(T, cls, my_pmmh, niter, y):
    series = []
    for i in range(niter // 3, niter):
        new_model = cls(**{name: my_pmmh.chain.theta[name][i] for name in cls.default_params})
        time_series = new_model.simulate(T)[1]
        series.append(time_series)
    posterior_samples = np.array(series).reshape(len(series), -1)
    lines = np.quantile(posterior_samples, [0.25, 0.5, 0.75], axis=0)
    plt.plot(lines[1], '-')
    plt.fill_between(np.arange(T), lines[0], lines[2], alpha=0.5)
    plt.plot(y, '-')
    plt.show()
    return new_model

