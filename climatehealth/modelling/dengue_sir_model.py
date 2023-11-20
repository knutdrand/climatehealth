from collections import OrderedDict
import arviz as az
import plotly.express as px
import numpy as np
import scipy.special
from matplotlib import pyplot as plt
from particles import state_space_models as ssm, mcmc, distributions as dists
from .compartementalized_model import SIRState, SIRModel, make_ssm_class, check_capasity, StateDistribution, \
    check_model_capasity, plot_posteriors


def make_ssm_class(weather_data, N=100000):

    class DengueSIRModel(ssm.StateSpaceModel):
        default_params = {'gamma': 0.01, 'beta': 0.05, 'epsilon': 0.0, 'beta_0': 0.0, 'i0': 0.05}

        def PX0(self):
            print(self.seasons)
            i0 = np.minimum(np.maximum(self.i0, 1/N), 0.4)
            #i0 = self.i0
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

        def eta(self, t):
            month = t % 12
            season = self.seasons[month] if month < 11 else 0
            return self.beta + self.beta_0 * weather_data[t] + season

    return SeasonalDengueModel


def analyze_data(df, exog_names = ['Rainfall', 'Temperature']):
    df = df.iloc[30:]
    pop_size = 100000
    cls = make_ssm_class(df['Temperature'].to_numpy(), N=pop_size)
    #px.line(df, x='Date', y='DengueCases').show()
    y = df['DengueCases'].to_numpy()
    ratio0 = (y[0]+1) / (pop_size+1)
    # check_model_capasity(cls, cls(**{name: dist.rvs() for name, dist in priors.items()}), priors, T=len(y))
    priors = {'gamma': dists.Beta(0.5, 10), 'beta': dists.Normal(0.5, 100), 'epsilon': dists.Beta(0.1, 1), 'beta_0': dists.Normal(0, 100), 'i0': dists.Beta(2*ratio0, 2*(1-ratio0)),
              'seasons': dists.MvNormal(np.zeros(11), 1000)}


    prior = dists.StructDist(OrderedDict(priors))
    niter = 1000
    my_pmmh = mcmc.PMMH(ssm_cls=cls,
                        prior=prior, data=y, Nx=600,
                        niter=niter)
    print('Estimating parameters')
    my_pmmh.run()  # may take several se
    plot_posteriors(len(y), cls, my_pmmh, niter, y)
    #plt.plot(my_pmmh.chain.theta['seasons'][-1], '-')

    posterior = my_pmmh.chain.theta['seasons'][niter // 3:]
    quantiles = np.quantile(posterior, [0.25, 0.5, 0.75], axis=0)
    plt.plot(quantiles[1], '-')
    plt.fill_between(np.arange(11), quantiles[0], quantiles[2], alpha=0.5)
    plt.show()
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
