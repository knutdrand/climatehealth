from collections import OrderedDict
from numbers import Number

import arviz as az
import plotly.express as px
import numpy as np
import scipy.special
from matplotlib import pyplot as plt
from particles import state_space_models as ssm, mcmc, distributions as dists
from .compartementalized_model import SIRState, SIRModel, make_ssm_class, check_capasity, StateDistribution, \
    check_model_capasity, plot_posteriors, plot_trace
from .plotting import plot_forecast

logit = scipy.special.logit

class SimpleNormal:
    def __init__(self, mu, sigma=1):
        self.mu = np.asanyarray(mu)
        assert self.mu.ndim == 1
        self.sigma = sigma
        self._dist = scipy.stats.norm(loc=self.mu, scale=self.sigma)
        self.dim = self.mu.size
        self.dtype=np.float64

    def rvs(self, size=None):
        if size is not None:
            size = (size, self.dim)
        return np.random.normal(self.mu, self.sigma, size=size)

    def logpdf(self, x):
        return scipy.special.logsumexp(self._dist.logpdf(x), axis=-1)


def make_ssm_class(weather_data, N=100000):

    class DengueSIRModel(ssm.StateSpaceModel):
        default_params = {'gamma': 0.01, 'beta': 0.05, 'epsilon': 0.0, 'beta_0': 0.0, 'i0': 0.05}

        def PX0(self):
            i0 = np.minimum(np.maximum(self.i0, 1/N), 0.4)
            s0 = 1-2*i0
            r0 = i0
            return StateDistribution[SIRState](1000*np.array([s0, i0, r0]))

        def PX(self, t, xp):
            beta = scipy.special.expit(self.eta(t))
            beta = np.clip(beta, 0.001, 0.999)
            k = 2
            a, b = beta*k, (1-beta)*k
            model = dists.Beta(a=a, b=b)
            return SIRModel(xp, beta=model, gamma=self.gamma, epsilon=self.epsilon)
            # return dists.Normal(loc=xp + self.theta * weather_data[t] + self.beta_0, scale=self.sigma ** 2)

        def eta(self, t):
            return self.beta + self.beta_0 * weather_data[t]

        def PY(self, t, xp, x):
            return dists.Poisson(x.I * N)


    class SeasonalDengueModel(DengueSIRModel):
        default_params = {'gamma': 0.01, 'beta': 0.05, 'epsilon': 0.0, 'beta_0': 0.0, 'i0': 0.05, 'seasons': np.ones(11)}
        # default_params = DengueSIRModel.default_params | {'seasons': np.ones(11)}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert self.seasons.ndim == 1, self.seasons.shape

        def eta(self, t):
            seasons = self.seasons.ravel()
            assert len(seasons) == 11, (seasons, seasons.shape)
            month = t % 12
            season = seasons[month] if month < 11 else 0
            return self.beta_0 + self.beta * weather_data[t] + season

    class LogParameterizedDengueModel(DengueSIRModel):
        default_params = {'logit_gamma':0, 'beta':0, 'logit_epsilon':0, 'beta_0':0, 'logit_i0':0, 'seasons':np.zeros(11)}
        priors = {'logit_gamma': dists.Normal(),
                  'beta': dists.Normal(),
                  'logit_epsilon': dists.Normal(),
                  'beta_0': dists.Normal(),
                  'logit_i0': dists.Normal(),
                  'seasons': SimpleNormal(np.zeros(11), 1)}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.gamma = scipy.special.expit(kwargs['logit_gamma'])
            self.epsilon = scipy.special.expit(kwargs['logit_epsilon'])
            self.i0 = scipy.special.expit(kwargs['logit_i0'])

    class PureSeasonalDengueModel(DengueSIRModel):
        default_params = {'logit_i0': 0.05, 'seasons': np.ones(12),
                          'logit_gamma':0, 'logit_epsilon':0}

        priors = {'logit_i0': dists.Normal(loc=logit(0.05), scale=3),
                  'seasons': SimpleNormal(np.zeros(12), 1),
                  'logit_gamma': dists.Normal(logit(0.99)),
                  'logit_epsilon': dists.Normal()
                  }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.i0 = scipy.special.expit(kwargs['logit_i0'])
            self.gamma = scipy.special.expit(kwargs['logit_gamma'])
            self.epsilon = scipy.special.expit(kwargs['logit_epsilon'])
            assert self.seasons.ndim == 1, self.seasons.shape

        def eta(self, t):
            seasons = self.seasons.ravel()
            assert len(seasons) == 12, (seasons, seasons.shape)
            month = t % 12
            return seasons[month]
            # season = seasons[month] if month < 11 else 0
            # return season

    return PureSeasonalDengueModel
    return LogParameterizedDengueModel

def analyze_data(df, exog_names = ['Rainfall', 'Temperature']):
    df = df.iloc[30:]
    pop_size = 1000000
    temperature = df['Temperature'].to_numpy()
    temperature = temperature - temperature.mean()
    temperature = temperature / temperature.std()
    plt.plot(temperature, '-')
    plt.show()
    cls = make_ssm_class(temperature, N=pop_size)
    priors = cls.priors
    #px.line(df, x='Date', y='DengueCases').show()
    y = df['DengueCases'].to_numpy()
    # ratio0 = (y[0]+1) / (pop_size+1)
    # priors['logit_i0'] = dists.Normal(loc=scipy.special.logit(ratio0), scale=1)
    # priors['logit_gamma'] = dists.Normal(loc=scipy.special.logit(0.99), scale=1)
    # priors['logit_epsilon'] = dists.Normal(loc=scipy.special.logit(1/12), scale=1)
    # priors['beta_0'] = dists.Normal(loc = scipy.special.logit(0.01), scale=1)
    # priors['beta'] = dists.Normal(loc=0.05, scale=1)
    # priors = {'gamma': dists.Beta(0.5, 10),
    #           'beta': dists.Normal(0.5, 10),
    #           'epsilon': dists.Beta(0.1, 1),
    #           'beta_0': dists.Normal(0, 10),
    #           'i0': dists.Beta(2*ratio0, 2*(1-ratio0)),
    #           'seasons': SimpleNormal(np.zeros(11), 10)}
    params = {name: dist.rvs() for name, dist in priors.items()}
    print(params)
    # check_model_capasity(cls, cls(**params), priors, T=len(y), pop_size=pop_size)
    # return

    prior = dists.StructDist(OrderedDict(priors))
    niter = 200
    my_pmmh = mcmc.PMMH(ssm_cls=cls,
                        prior=prior, data=y, Nx=600,
                        niter=niter)


    print('Estimating parameters')
    my_pmmh.run()  # may take several se
    plot_trace(my_pmmh, cls)
    plot_seasons(my_pmmh, niter, y)
    plot_forecast(y, cls, my_pmmh.chain.theta)
    return
    I = np.maximum(y[0], 1)/pop_size
    start_state = SIRState([1-2*I], [I], [I])
    plot_posteriors(len(y), cls, my_pmmh, niter, y, start_state)
    #plt.plot(my_pmmh.chain.theta['seasons'][-1], '-')


    #az.plot_posterior(az.convert_to_inference_data(posterior))# , round_to=2, credible_interval=0.95)
    # plt.show()
    # theta = my_pmmh.chain.theta['theta'][-1]
    # sigma = my_pmmh.chain.theta['sigma'][-1]
    # beta_0 = my_pmmh.chain.theta['beta_0'][-1]
    # px.density_heatmap(my_pmmh.chain.theta['theta'][100:], my_pmmh.chain.theta['beta_0'][100:]).show()
    # for name in cls.default_params:
    #     px.line(my_pmmh.chain.theta[name][100:], title=name).show()
    #     px.histogram(my_pmmh.chain.theta[name][100:], title=name).show()
    # my_model = cls(**{name: my_pmmh.chain.theta[name][-1] for name in cls.default_params})
    # x, new_y = my_model.simulate(len(y))
    # plt.plot(y)
    # plt.plot(new_y)
    # plt.show()


def plot_seasons(my_pmmh, niter, y):
    n_params= len(my_pmmh.chain.theta['seasons'][0])
    posterior = my_pmmh.chain.theta['seasons'][niter // 3:]
    quantiles = np.quantile(posterior, [0.25, 0.5, 0.75], axis=0)
    plt.plot(quantiles[1], '-')
    plt.fill_between(np.arange(n_params), quantiles[0], quantiles[2], alpha=0.5)
    n_years = len(y) // 12
    mean_seasons = y[:n_years * 12].reshape(n_years, 12).mean(axis=0)
    plt.plot(mean_seasons * np.max(quantiles[1]) / np.max(mean_seasons), '.')
    plt.show()
