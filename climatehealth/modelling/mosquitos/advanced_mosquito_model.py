import numpy as np
from jax.scipy.special import logit, expit


class FixedMosquitoModel:
    T = 1000
    n_infected_mosquitos = (np.arange(T) % 365)*1000+1000
    # gamma, mu, a
    seir_params = [0.2, 0.1, 1/10]
    alpha, beta = logit(0.1), 0.0001
    def d(self, human_state, seir_params):
        beta, gamma, mu, a = seir_params
        n_mu = (1-mu)
        S, E, I, R = human_state
        d = np.array(
            [mu - mu * S - n_mu * beta * S * I,
             beta * S * I - (mu + a * n_mu) * E,
             a * E - (mu + gamma * n_mu) * I,
             gamma * I - mu * R])
        return d

    def sample(self):
        observed = []
        state = np.array([0.25, 0.25, 0.25, 0.25])
        for t in range(self.T):
            beta = expit(self.alpha + self.beta * self.n_infected_mosquitos[t])
            state = state + self.d(
                state, [beta] + self.seir_params)
            observed.append(state[-2])
        return np.array(observed)



if __name__ == '__main__':
    model = FixedMosquitoModel()
    states = model.sample()

    import matplotlib.pyplot as plt
    plt.plot(states)
    plt.show()

