from numba import jit, njit
import numpy as np

# Regularization parameters
beta_x = 0.001
beta_y = 0.001
# Number of nodes
N = 100

@njit
def logloss(w, X, y):
    summ = 0
    for i in range(X.shape[0]):
        summ += np.log(1 + np.exp(-y[i] * (w.T @ X[i])))
    return summ / X.shape[0]

# def loss()

@njit
def f_m(w, X, y):
    summ = 0
    for i in range(X.shape[0]):
        summ += np.log(1 + np.exp(-y[i] * (w.T @ X[i])))
    return summ / X.shape[0]

# Loss based on f_i
def P(weights, noise, X_all, y_all):
    n = len(X_all)
    for i in range(n):


    return 1/N *

# Regularization
def Q(weights, noise, X, y):
    return beta_x * np.linalg.norm(weights)**2 - beta_y * np.linalg.norm(noise)**2

# Objective
def R():
    return P()+Q()

@njit
def grad(w, X, y):
    g = np.zeros(w.shape)
    for i in range(X.shape[0]):
        g += y[i] * X[i] / (1 + np.exp(y[i] * w.T @ X[i]))
    return -g / X.shape[0]



def extragrad_sliding(x_0, nu, theta, K):
    history = []
    x = x_0
    for k in range(K):
        # Use EAG-V to solve auxiliary problem
        u = EAG_V(x, theta)
        x = x - nu * R(u)
        #history.append(full_grad(x, X_train, y_train))
    return history, x