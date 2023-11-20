from collections import OrderedDict
import plotly.express as px
import numpy as np
import scipy.special
from matplotlib import pyplot as plt
from particles import state_space_models as ssm, mcmc, distributions as dists
from .compartementalized_model import SIRState, SIRModel, make_ssm_class, check_capasity, StateDistribution, \
    check_model_capasity


def make_ssm_class(weather_data, N=100000):

    class DengueSIRModel(ssm.StateSpaceModel):
        default_params = {'gamma': 0.01, 'beta': 0.05, 'beta_0': 0.0, 's0': 0.9, 'i0': 0.05}

        def PX0(self):
            s0= self.s0
            i0 = self.i0*(1-s0)
            r0 = 1-s0-i0
            return StateDistribution[SIRState](np.array([s0, r0, i0]))

        def PX(self, t, xp):
            beta = scipy.special.expit(self.beta+self.beta_0*weather_data[t])
            beta = np.clip(beta, 0.001, 0.999)
            k = 2
            a, b = beta*k, (1-beta)*k
            model = dists.Beta(a=a, b=b)
            return SIRModel(xp, beta=model, gamma=self.gamma)
            # return dists.Normal(loc=xp + self.theta * weather_data[t] + self.beta_0, scale=self.sigma ** 2)

        def PY(self, t, xp, x):
            return dists.Poisson(x.I * N)

    return DengueSIRModel


def analyze_data(df, exog_names = ['Rainfall', 'Temperature']):
    cls = make_ssm_class(df['Rainfall'].to_numpy(), N=100000)
    #px.line(df, x='Date', y='DengueCases').show()
    y = df['DengueCases'].to_numpy()
    priors = {'gamma': dists.Beta(0.5, 10), 'beta': dists.Normal(0.5, 10), 'beta_0': dists.Normal(0, 10),
              's0': dists.Beta(1, 1), 'i0': dists.Beta(1, 1)}

    check_model_capasity(cls, cls(**{name: dist.rvs() for name, dist in priors.items()}), priors, T=len(y))
    return
    prior = dists.StructDist(OrderedDict(priors))
    my_pmmh = mcmc.PMMH(ssm_cls=cls,
                        prior=prior, data=y, Nx=600,
                        niter=10000)
    print('Estimating parameters')
    my_pmmh.run()  # may take several se
    # theta = my_pmmh.chain.theta['theta'][-1]
    # sigma = my_pmmh.chain.theta['sigma'][-1]
    # beta_0 = my_pmmh.chain.theta['beta_0'][-1]
    # px.density_heatmap(my_pmmh.chain.theta['theta'][100:], my_pmmh.chain.theta['beta_0'][100:]).show()
    for name in cls.default_params:
        px.line(my_pmmh.chain.theta[name][100:], title=name).show()
        px.histogram(my_pmmh.chain.theta[name][100:], title=name).show()
    my_model = cls(**{name: my_pmmh.chain.theta[name][-1] for name in cls.default_params})
    x, new_y = my_model.simulate(len(y))
    plt.plot(y)
    plt.plot(new_y)
    plt.show()
