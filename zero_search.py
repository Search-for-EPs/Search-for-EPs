"""Performs zero search for 2D input"""

import numpy as np
from scipy import optimize


def zero_search(fun, guess):
    """Performs root finding

    Parameters
    ----------
    fun
        Callable function on which you want to perform root finding
    guess
        Initial guess of the root

    Returns
    -------
    np.ndarray
        Root of the callable function
    """
    sol = optimize.root(fun, np.array([guess.real, guess.imag]))
    return np.array([complex(sol.x[0], sol.x[1])])
