import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def Q(x1, x2):
    return -5*x1**2 + x2**2 + 1/2*(x1-1)**2

## Problem 4
delta = 0.05
x = np.arange(-5.0, 5.0, delta)
y = np.arange(-5.0, 5.0, delta)
X, Y = np.meshgrid(x, y)


fig, ax = plt.subplots()
CS = ax.contour(X, Y, Q(X, Y))
fig.colorbar(CS)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Contour lines of the quadratic penalty function Q')
plt.show()
