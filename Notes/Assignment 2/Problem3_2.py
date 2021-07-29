from numpy import array, dot
from qpsolvers import solve_qp

## Problem 3.(2)
# Define parameters
delta = (-0.1, 0., 0.1)
P = array([[2., -1.], [-1., 4.]])
q = array([[-1., 0.]]).reshape((2,))
G = array([[1., 2.], [1., -4.], [5., 76.]])

for i in range(len(delta)):
    for j in range(len(delta)):
        h = array([-2.+delta[i], -3.+delta[j], 1.]).reshape((3,))

        # Solve QP
        x = solve_qp(P, q, G, h)
        p = 1 / 2 * dot(dot(x, P), x) + dot(q, x)

        print("")
        print("delta1 = {}, delta2 = {}".format(delta[i], delta[j]))
        print("QP solution: x* = {}".format(x))
        print("QP solution: p* = {}".format(p))

for i in range(len(delta)):
    for j in range(len(delta)):
        pp = 8.2222 - 1.916*delta[i] - 3.4571*delta[j]

        print("")
        print("delta1 = {}, delta2 = {}".format(delta[i], delta[j]))
        print("QP solution: p^ = {}".format(pp))