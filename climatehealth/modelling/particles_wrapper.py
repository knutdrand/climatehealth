from collections import OrderedDict

import numpy as np
import particles
from matplotlib import pyplot as plt
from particles import state_space_models as ssm, mcmc
from particles import distributions as dists
from particles.collectors import Moments
from particles.state_space_models import StochVol


def make_ssm_class(weather_data):
    class DengueModel(ssm.StateSpaceModel):
        default_params = {'theta': 1.0, 'sigma': 1.0, 'beta_0': 0.0}

        def PX0(self):
            return dists.Normal(loc=0, scale=self.sigma**2)

        def PX(self, t, xp):
            return dists.Normal(loc=xp + self.theta * weather_data[t]+self.beta_0, scale=self.sigma**2)

        def PY(self, t, xp, x):
            return dists.Poisson(x**2)

    return DengueModel

def analyze_data(df, exog_names = ['Rainfall', 'Temperature']):
    cls = make_ssm_class(df['Rainfall'].to_numpy())
    y = df['DengueCases'].to_numpy()
    prior = dists.StructDist(OrderedDict({'beta_0': dists.Normal(0, 10), 'theta': dists.Normal(0, 10), 'sigma': dists.Gamma()}))
    my_pmmh = mcmc.PMMH(ssm_cls=cls,
                        prior=prior, data=y, Nx=600,
                        niter=500)
    print('Estimating parameters')
    my_pmmh.run();  # may take several se
    theta = np.mean(my_pmmh.chain.theta['theta'][100:])
    sigma = np.mean(my_pmmh.chain.theta['sigma'][100:])
    my_model = cls(theta=theta, sigma=sigma)
    x, new_y = my_model.simulate(len(y))
    plt.plot(y)
    plt.plot(new_y)
    plt.show()
    # plt.plot(my_pmmh.chain.theta['theta'][100:])
    # plt.plot(my_pmmh.chain.theta['sigma'][100:])
    # plt.show()
    # plt.hist(my_pmmh.chain.theta['theta'][100:], bins=50)
    # plt.show()


def make_particles_estimator(weather_data, case_count_distribution, state_distribution, initial_state_distribution):
    '''
    case_count_distribution are with priors maybe?
    '''

    class Model(ssm.StateSpaceModel):

        def PX0(self):
            return initial_state_distribution

        def PX(self, t, xp):
            return state_distribution(xp, weather_data[t])

        def PY(self, t, xp, x):
            return case_count_distribution(x)

    cls = Model
    x, y = my_model.simulate(100)

    prior = dists.StructDist(OrderedDict({'theta': dists.Normal(0, 10)}))
    my_pmmh = mcmc.PMMH(ssm_cls=cls,
                        prior=prior, data=y, Nx=300,
                        niter=1000)
    my_pmmh.run();  # may take several se
    plt.plot(my_pmmh.chain.theta['theta'])
    plt.show()
    plt.hist(my_pmmh.chain.theta['theta'][100], bins=50)
    plt.show()