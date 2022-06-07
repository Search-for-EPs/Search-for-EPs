"""Performs zero search for 2D input"""

import numpy as np
from scipy import optimize


def test_func(x):
    return [x[1] ** 2 + x[0] + x[1],
            x[0] + 3 * x[1] + 1]


def test_jac(x):
    return np.array([[1,
                      1],
                     [2 * x[1] + 1,
                      3]])


def zero_search(fun, guess):
    return optimize.root(fun, np.array([guess.real, guess.imag]))


if __name__ == '__main__':
    sol = optimize.root(test_func, np.array([10, 10]))
    print(sol.x)
