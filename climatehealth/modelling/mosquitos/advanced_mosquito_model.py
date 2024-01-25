import numpy as np
from jax.scipy.special import logit, expit
import jax.numpy as jnp
mosquito_state_names = ['E', 'L', 'P', 'A', 'L', 'I']
human_state_names = ['S', 'E', 'I', 'R']

class FixedMosquitoModel:
    T = 2000
    n_infected_mosquitos = (np.arange(T) % 365) * 1000 + 1000
    temperature = (np.arange(T) % 365) * 24/365 + 10
    # gamma, mu, a
    seir_params = [1/14, 0.1, 1/10]
    death_rate = np.array([0.05, 0.05, 0.05, 0.1,0.1,0.1])
    maturation = np.array([0.00001, 0.2, 0.33, 0.1, 1/7, 0.1])
    m_alpha, m_beta = 1.0/250, -11.
    t_beta = 0.5
    h_beta = 1.
    alpha, beta = logit(0.0002), 0.005

    @staticmethod
    def d_human(human_state, seir_params):
        beta, gamma, mu, a = seir_params
        n_mu = (1-mu)
        S, E, I, R = human_state
        d = np.array(
            [mu - mu * S - beta * S,
             beta * S - (mu + a * n_mu) * E,
             a * E - (mu + gamma * n_mu) * I,
             gamma * I - mu * R])
        return d

    @staticmethod
    def d_mosquito(state, death_rate, maturation_rate, beta, alpha, egglaying_rate):
        deaths = -death_rate * state
        maturation = (state + deaths) * maturation_rate
        carry_deaths = expit(beta + alpha * state[..., 1])*0.7
        d = jnp.array([deaths[..., 0] - maturation[..., 0] + egglaying_rate *state[..., 3:].sum(),
                       deaths[..., 1] + maturation[..., 0] - carry_deaths * (state[..., 1] + deaths[..., 1])-maturation[..., 1]*(1-carry_deaths),
                       deaths[..., 2] - maturation[..., 2] + maturation[..., 1]*(1-carry_deaths),
                       deaths[..., 3] + maturation[..., 2],
                       deaths[..., 4] + maturation[..., 3] - maturation[..., 4],
                       deaths[..., 5] + maturation[..., 4]]).T
        return d

    def sample(self):
        observed = []
        state = np.array([0.25, 0.25, 0.25, 0.25])
        mosquito_state = np.full(6, 2000.0)
        for t in range(self.T):
            beta = expit(self.alpha + self.beta * mosquito_state[-1])
            maturation_rate = self.maturation.copy()
            temp_dependent = expit(logit(maturation_rate[0]) + self.t_beta * self.temperature[t])
            maturation_rate[0] = temp_dependent / 3
            maturation_rate[1] = temp_dependent / 5
            maturation_rate[3] = expit(logit(maturation_rate[3])+self.h_beta*state[-2])
            d_human = self.d_human(state, [beta] + self.seir_params)
            d_mosquito = self.d_mosquito(mosquito_state, self.death_rate, maturation_rate, self.m_beta, self.m_alpha, 0.7)
            state = state + d_human
            mosquito_state = mosquito_state + d_mosquito
            fullstate= np.concatenate([state, mosquito_state])
            observed.append(fullstate)
        return np.array(observed)



if __name__ == '__main__':
    T = 1000
    model = FixedMosquitoModel()
    t = np.load(
        '/home/knut/Sources/climatehealth/tests/NCLE: 7. Dengue cases (any) 01 Vientiane Capital_temperature_daily.npy') - 273.15
    model.temperature = t[:2000]
    model.T = len(model.temperature)
    states = model.sample()
    import matplotlib.pyplot as plt
    for i in range(4):
        plt.plot(states[:, i], label=human_state_names[i])
    plt.legend()
    plt.show()
    for i in range(4, 10):
        plt.plot(states[:, i], label=mosquito_state_names[i-4])
    plt.legend()
    plt.show()
    adult_mosquitos = states[:, -3:].sum(axis=-1)
    plt.plot(states[:, 1] + states[:, 2], label='Sick')
    plt.plot(adult_mosquitos/np.max(adult_mosquitos), label='Adult Mosquitos')
    plt.plot(model.temperature/np.max(model.temperature), label='Temperature')
    plt.legend()
    plt.show()
    plt.plot(adult_mosquitos, label='Adult Mosquitos')
    plt.legend()
    plt.show()
