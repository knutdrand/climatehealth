import numpy as np
import scipy
from matplotlib import pyplot as plt


def plot_beta(a, b):
    x = np.linspace(0, 1, 1000)
    plt.plot(x, scipy.stats.beta.pdf(x, a, b), label=f'Beta({a}, {b})')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    k = 200
    mu = 0.01
    plot_beta(k*mu, (1-mu)*k)