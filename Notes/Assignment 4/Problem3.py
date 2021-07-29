import numpy as np
from time import process_time
from scipy.sparse import diags
from scipy.linalg import solveh_banded
from cvxopt import matrix, solvers
from qpsolvers import solve_qp

## Problem 3
k = 100
n = 2000
delta = 1
eta = 1

Delta = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
Delta[0, 0] = 1
Delta[n - 1, n - 1] = 1
Ia = np.eye(n, n)
Ib = np.eye(k, k)

N = 10


# (a) a generic method
def genericmethod(A, b, Delta, delta, eta, ta):
    taInitial = process_time()
    Ahat = A.T.dot(A) + delta * Delta + eta * Ia
    bhat = A.T.dot(b)
    xaStar = np.linalg.solve(Ahat, bhat)
    taEnd = process_time()
    ta = ta + taEnd - taInitial
    return xaStar, ta


# (b) an efficient method
def efficientmethod(A, b, n, tb):
    # Ahat = delta * Delta + eta * Ia
    Bhat = A.T
    Chat = A
    bhat = A.T.dot(b)
    Atild = np.append(-np.ones((1, n)), 3 * np.ones((1, n)), axis=0)
    Atild[1, 0] = 2

    tbInitial = process_time()
    AhatInv = solveh_banded(Atild, Ia)
    F = AhatInv.dot(Bhat)
    g = AhatInv.dot(bhat)
    H = Ib + Chat.dot(F)
    k = Chat.dot(g)
    ybStar = np.linalg.solve(H, k)
    xbStar = g - F.dot(ybStar)
    tbEnd = process_time()
    tb = tb + tbEnd - tbInitial
    return xbStar, tb


# (c) an optimisation package method
# Define model and constrains
def optimisationpackage(A, b, Delta, delta, eta, tc):
    Q = 2*matrix(A.T.dot(A) + delta*Delta + eta*Ia)
    p = matrix(-2*A.T.dot(b))
    tcInitial = process_time()
    xcStar = solvers.qp(Q, p)['x']
    # P = A.T.dot(A) + delta * Delta + eta * Ia
    # q = -2 * A.T.dot(b)
    # xcStar = solve_qp(P, q)
    tcEnd = process_time()
    tc = tc + tcEnd - tcInitial
    return xcStar, tc


np.random.seed(15643789)
ta = 0
xaAll = 0
tb = 0
xbAll = 0
tc = 0
xcAll = 0

for i in range(N):
    A = np.random.rand(k, n)
    b = np.random.rand(k, 1)

    # (a)
    xaStar, ta = genericmethod(A, b, Delta, delta, eta, ta)
    xaAll = xaAll + xaStar

    # (b)
    xbStar, tb = efficientmethod(A, b, n, tb)
    xbAll = xbAll + xbStar

    # (c)
    xcStar, tc = optimisationpackage(A, b, Delta, delta, eta, tc)
    xcAll = xcAll + xcStar

taMean = ta / N
xaStarMean = xaAll / N

tbMean = tb / N
xbStarMean = xbAll / N

tcMean = tc / N
xcStarMean = xcAll / N

# MSE = mean_squared_error(xaStar, xbStar)
errorab = np.linalg.norm(xbStarMean - xaStarMean) / np.linalg.norm(xaStarMean)
errorac = np.linalg.norm(xcStarMean - xaStarMean) / np.linalg.norm(xaStarMean)
