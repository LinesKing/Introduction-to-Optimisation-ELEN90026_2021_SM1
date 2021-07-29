from scipy.sparse import random
import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import cvxpy as cp

# Define parameters
n = 100  # nodes
m = 10  # edges

np.random.seed(8941962)
s = random(n, 1, density=10 / n).A  # sparse matrix s with 10 nonzero data
for i in range(n):
    s[i, 0] = 10 * s[i, 0] * (int((s[i, 0] > 0.5)) - int((s[i, 0] < 0.5) & (s[i, 0] > 0)))
s_sum = np.sum(s)  # check if sum of s is 0

np.random.seed(2648157)
t = random(n, 1, density=10 / n).A  # sparse matrix t with 10 nonzero data
for i in range(n):
    t[i, 0] = 10 * t[i, 0] * (int((t[i, 0] > 0.5)) - int((t[i, 0] < 0.5) & (s[i, 0] > 0)))
t_sum = np.sum(t)  # check if sum of s is 0

# Make the problem feasible by sign(s) = sign(t)
flag = np.sign(s) - np.sign(t)
for i in range(n):
    if np.abs(flag[i, 0]) >= 1:
        t[i, 0] = -t[i, 0]
flagNew = np.sign(s) - np.sign(t)

np.random.seed(5615234)
ASumOfRow = np.random.rand(n, 1)
for i in range(n):
    ASumOfRow[i, 0] = round(5 * ASumOfRow[i, 0] * np.sign(s[i, 0]))
# A.sum(axis=1)
A = random(n, m, density=0.005).A  # sparse matrix A
for i in range(n):
    sumA = A.sum(axis=1)[i]
    if sumA != ASumOfRow[i, 0]:
        A[i, np.random.randint(0, m-1)] = A[i, np.random.randint(0, m-1)] + ASumOfRow[i, 0] - sumA
