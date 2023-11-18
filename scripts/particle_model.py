from collections import OrderedDict

import particles.distributions as dists
import numpy as np
from matplotlib import pyplot as plt
from particles import mcmc

from climatehealth.modelling.particles_wrapper import make_ssm_class

if __name__ == '__main__':
    cls = make_ssm_class(np.random.randn(100))
    my_model = cls(theta=2.0, sigma=0.5)

    x, y = my_model.simulate(100)

    prior = dists.StructDist(OrderedDict({'theta': dists.Normal(0, 10), 'sigma': dists.Gamma()}))
    my_pmmh = mcmc.PMMH(ssm_cls=cls,
                        prior=prior, data=y, Nx=500,
                        niter=1000)
    my_pmmh.run();  # may take several se
    plt.plot(my_pmmh.chain.theta['theta'][100:])
    plt.plot(my_pmmh.chain.theta['sigma'][100:])
    plt.show()
    plt.hist(my_pmmh.chain.theta['theta'][100:], bins=50)
    plt.show()
