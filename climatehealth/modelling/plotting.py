import numpy as np
import particles
from matplotlib import pyplot as plt
from particles import state_space_models as ssm, mcmc, distributions as dists


def simulate_from_state(model, state, T, t0=0):
    x = [state]
    for t in range(t0, T-1):
        law_x = model.PX(t, x[-1])
        x.append(law_x.rvs(size=1))
    y = model.simulate_given_x(x)
    return x, y


def plot_forecast(y, model_cls, theta):
    param_names = list(model_cls.default_params)
    n_iter = len(theta[param_names[0]])
    # mcmc_i = t
    ys = []
    T = len(y)
    cutoff = T*2//3
    fixed_y = y[:cutoff]
    for mcmc_i in range(n_iter // 2, n_iter, n_iter // 200+1):
        params = {name: theta[name][mcmc_i] for name in param_names}
        model = model_cls(**params)
        alg = particles.SMC(fk=ssm.Bootstrap(ssm=model, data=fixed_y), N=200)
        alg.run()
        i = np.random.choice(np.arange(len(alg.W)),
                             p=alg.W)
        state = alg.X[i]
        print(state)
        _, new_y = simulate_from_state(model, state, len(y), t0=cutoff)
        ys.append(new_y)

    posterior_samples = np.array(ys).reshape(len(ys), -1)
    lines = np.quantile(posterior_samples, np.array([0.25, 0.5, 0.75]), axis=0)
    x = np.arange(cutoff, T)
    plt.plot(x, lines[1], '-')
    plt.fill_between(x, lines[0], lines[2], alpha=0.5)
    plt.plot(y, '-')
    plt.show()
