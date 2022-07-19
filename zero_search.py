"""Performs zero search for 2D input"""

import numpy as np
from scipy import optimize


def zero_search(fun, guess):
    sol = optimize.root(fun, np.array([guess.real, guess.imag]))
    return np.array([complex(sol.x[0], sol.x[1])])
