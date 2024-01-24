from probabilistic_machine_learning.adaptors.jax_nuts import sample
import jax.numpy as jnp
from jax.scipy import stats as stats
import numpy.random as random
np = jnp
'''
Temperature and rainfall play crucial roles in influencing the life cycle of Aedes aegypti mosquitoes, which are known vectors for diseases such as dengue, Zika, and chikungunya. The life cycle of these mosquitoes involves four stages: egg, larva, pupa, and adult. Here's how temperature and rainfall can impact each stage:

Egg Stage:

Aedes aegypti mosquitoes typically lay their eggs in containers holding water, such as flower pots, discarded tires, or water storage containers.
Rainfall contributes to the availability of breeding sites by filling containers with water.
The combination of warmer temperatures and water availability promotes the hatching of mosquito eggs.
Larval Stage:

Larvae hatch from eggs and live in the water. They feed on microorganisms present in the water.
Higher temperatures generally accelerate the larval development process. Warmer environments speed up the metabolism of the larvae, leading to faster growth.
Increased rainfall provides more breeding sites for larvae, promoting their survival and growth.
Pupal Stage:

After the larval stage, mosquitoes enter the pupal stage. Pupae are immobile and non-feeding but undergo important transformations.
Warmer temperatures speed up the pupal development, leading to a shorter time between the larval and adult stages.
Adult Stage:

Once the pupal stage is complete, adult mosquitoes emerge from the water.
Adult mosquitoes are highly sensitive to temperature, and their activity, feeding habits, and lifespan are influenced by temperature variations.
Adequate rainfall ensures the persistence of breeding sites, allowing for continuous reproduction and sustained mosquito populations.
Overall, warmer temperatures generally accelerate the entire life cycle of Aedes aegypti mosquitoes. This can lead to faster population growth and increased transmission of diseases they may carry. Additionally, rainfall contributes to the availability of suitable breeding sites, influencing the distribution and abundance of these mosquitoes in a given area. Effective mosquito control strategies often take these environmental factors into account to mitigate the risk of disease transmission.
'''
import jax

logit = lambda x: jax.scipy.special.logit(x)
expit = lambda x: jax.scipy.special.expit(x)


class SpecificMosquitoModel:
    intial_state = np.array([100., 100., 100., 100.])
    T = 100
    death_rate = np.array([0.05, 0.05, 0.05, 0.1])
    maturation = np.array([0.01, 0.2, 0.33, 0.7])
    alpha = 1/10000
    beta = -1
    sigma = 0.1
    K = 100
    initial_position = {#'log_populations': np.array([np.log(intial_state)]*T),
                        'lo_death_rate': np.full(4, -1.),
                        'lo_maturation_rate': np.full(4, -1.),
                        'l_alpha': -1.0}

    @staticmethod
    def d(state, death_rate, maturation_rate, alpha, beta):
        #maturation = maturation_rate * state
        deaths = -death_rate * state
        maturation = (state+deaths)*maturation_rate
        # d = -death_rate*state
        d = jnp.array([deaths[..., 0] - maturation[..., 0] + maturation[..., -1],
                       deaths[..., 1] - maturation[..., 1] + maturation[..., 0] - expit(beta+alpha * state[..., 1]) * (
                                   state[..., 1] + deaths[..., 1] - maturation[..., 1]),
                       # (maturation_rate[..., 0]*state[..., 0]-maturation[..., 1]+maturation[..., 0]-alpha*state[..., 1]**2),
                       deaths[..., 2] - maturation[..., 2] + maturation[..., 1],
                       deaths[..., 3] + maturation[..., 2]]).T
        return d


    def sample(self):
        states = [self.intial_state.copy()]
        for i in range(self.T):
            state = states[-1]
            d = self.d(state, self.death_rate, self.maturation, self.alpha, self.beta)
            mu = state + d
            assert np.all(mu > 0)
            #mu = np.maximum(1, mu)
            states.append(np.exp(random.normal(np.log(mu), self.sigma)))
        return np.array(states)

    def logdensity_func(self, log_populations, lo_death_rate, lo_maturation_rate, l_alpha):
        death_rate = jax.scipy.special.expit(lo_death_rate)
        maturation_rate = jax.scipy.special.expit(lo_maturation_rate)
        alpha = np.exp(l_alpha)
        populations = np.exp(log_populations)
        d = self.d(populations, death_rate, maturation_rate, alpha, self.beta)
        mu = populations + d
        # mu = np.maximum(1, mu)
        logpdf = np.sum(stats.norm.pdf(log_populations[1:], np.log(mu)[:-1], self.sigma))
        return logpdf

    def estimate(self, observed):
        logdensity = lambda x: self.logdensity_func(log_populations=np.log(observed), **x)
        l = logdensity(self.initial_position)
        d = jax.grad(logdensity)(self.initial_position)
        rng_key = jax.random.key(1000)

        return sample(logdensity, rng_key,
                      self.initial_position,
                      num_warmup=500,
                      num_samples=500)


if __name__ == '__main__':
    model = SpecificMosquitoModel()
    states = model.sample()
    import matplotlib.pyplot as plt
    for i, col in enumerate(states.T):
        plt.plot(col, '-', label=f'{i}')
    plt.legend()
    plt.show()
    result = model.estimate(states)
    #print(result)
    print(expit(result['lo_death_rate']).mean(axis=0))
    print(expit(result['lo_maturation_rate']).mean(axis=0))
    print(np.exp(result['l_alpha']).mean(axis=0))
    for i in range(4):
        plt.plot(result['lo_death_rate'][..., i], label=f'{i}')
    plt.legend()
    plt.show()
    for i in range(4):
        plt.plot(result['lo_maturation_rate'][..., i], label=f'{i}')
    plt.legend()
    plt.show()