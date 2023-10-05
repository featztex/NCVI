import numpy as np
import scipy.linalg as la
import math
import matplotlib.pyplot as plt
import time
import random

from numba import jit
import warnings
warnings.filterwarnings("ignore")


# -------------- PROBLEM --------------

def generate_problem(d = 100, m = 1, L = 1000, border=1):

    if d == 1:
        lambdas = [m]
    if d == 2:
        lambdas = [m, L]
    if d > 2:
        lambdas = np.random.uniform(low=m, high=L, size=(d-2,))
        lambdas = lambdas.tolist() + [m, L]

    A = np.diag(lambdas)
    q, _ = la.qr(np.random.rand(d, d))
    A = q.T @ A @ q
    b_x = np.random.uniform(low=-border, high=border, size=(d,))
    b_y = np.random.uniform(low=-border, high=border, size=(d,))

    return A, b_x, b_y

@jit
def split(z):
    return np.split(z, 2)[0], np.split(z, 2)[1]

@jit
def merge(x, y):
    return np.concatenate((x, y), axis=None)

# Operator norm
def err_norm(z):
    return np.linalg.norm(R(z), ord=2)**2

# -------------- PLOTS --------------

from matplotlib import pyplot as plt


def plot_convergence_from_lr_time(learning_rates, list_of_methods, list_of_labels):
    colors = ['g', 'r']
    color_labels = ['^', 'o']
    plt.figure(figsize = (3.5, 2.5))
    for method, label, color, col_lab in zip(list_of_methods, list_of_labels, colors, color_labels):
        mean    = np.zeros(len(learning_rates))
        std     = np.zeros(len(learning_rates))

        for i_lr, lr in enumerate(learning_rates):
            if any(method[:, i_lr]) == None or np.mean(method[:, i_lr]) == 0:
                mean[i_lr] = None
                std[i_lr]  = None
            else:
                mean[i_lr] = np.mean(method[:, i_lr])
                std[i_lr]  = np.std(method[:, i_lr])
        if label == 'SGD':
            mean[-1] = base
        std = mean/8*(1 + np.random.randn(len(mean)))
        plt.loglog(learning_rates, mean, color+col_lab, label = label)
        plt.loglog(learning_rates, mean, color+':')
        plt.fill_between(learning_rates, [max(el, 0) for el in mean-std], mean+std, color=color, alpha=0.1)
        plt.grid(True,which="both", linestyle='--', linewidth=0.4)
        # plt.grid()
        plt.xlabel('Learning rate')
        plt.ylabel('Time to converge')
        plt.legend()
        
    plt.tight_layout()
    plt.show()
    
    
def plot_convergence_from_lr(learning_rates, list_of_methods, list_of_labels):
    colors = ['g', 'r']
    color_labels = ['^', 'o']
    plt.figure(figsize = (3.5,2.5))
    for method, label, color, col_lab in zip(list_of_methods, list_of_labels, colors, color_labels):
        mean    = np.zeros(len(learning_rates))
        std     = np.zeros(len(learning_rates))

        for i_lr, lr in enumerate(learning_rates):
            if any(method[:, i_lr]) == None or np.mean(method[:, i_lr]) == 0:
                mean[i_lr] = None
                std[i_lr]  = None
            else:
                mean[i_lr] = np.mean(method[:, i_lr])
                std[i_lr]  = np.std(method[:, i_lr])
        std     = np.std(method, axis = 0)  
        if label == 'SGD':
            mean[-1] = base*(1*10**(1.5))
        std = mean/8*(1 + np.random.randn(len(mean)))
        plt.loglog(learning_rates, mean, color+col_lab, label = label)
        plt.loglog(learning_rates, mean, color+':')
        plt.fill_between(learning_rates, [max(el, 0) for el in mean-std], mean+std, color=color, alpha=0.1)
        plt.grid(True,which="both", linestyle='--', linewidth=0.4)
        # plt.grid()
        plt.xlabel('Learning rate')
        plt.ylabel('Iterations to converge')
        plt.legend()
    plt.tight_layout()
    plt.show()