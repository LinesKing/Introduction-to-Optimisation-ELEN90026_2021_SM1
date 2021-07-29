import numpy as np
import matplotlib.pyplot as plt


def p(x1, x2):
    return x - x1 * (np.log(x2) + np.log(1 - x2))


## Problem 5
mu = np.linspace(0.1, 5, 10)
delta = 0.005
x = np.arange(1e-6, 1 - 1e-6, delta)

for mui in mu:
    plt.plot(x, p(mui, x), label='mu = %.2f' % mui)

plt.legend()
plt.show()
