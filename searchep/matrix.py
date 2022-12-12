"""Creates matrix of model system"""

import numpy as np


def parametrization(kappa_0, r, s):
    """Parametrization of a circle in the parameter plane

    Creates a circle around a given center (kappa_0) with a radius (r) and steps (s). The steps are equidistantly
    distributed from 0 to 2pi. 0 and 2pi are both included.

    Parameters
    ----------
    kappa_0 : complex
        Complex value for the center of the circle in parameter plane.
    r : float
        Radius of the circle.
    s : int
        Number of steps.

    Returns
    ----------
    (np.ndarray, numpy.ndarray)
        Returns two 1D arrays. The first one contains all complex values of kappa which belong to each step. The second
        array contains all angles from 0 to 2pi.
    """
    phi = np.arange(0, 2 * np.pi + 2 * np.pi / s, 2 * np.pi / s)  # + 2 * np.pi / s
    kappa = kappa_0 + r * np.exp(1j * phi)
    return kappa, phi


def matrix_random(n, kappa):
    """Creates random complex symmetric matrices

    Creates n dimensional random complex symmetric matrices with the simple model in the top left corner.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    kappa : np.ndarray
        Contains all kappa values for the off diagonal elements in the simple model.

    Returns
    ----------
    np.ndarray
        Returns an array which contains a random complex symmetric matrix for each kappa value of the circle
        parametrization.
    """
    matrices = []
    m = np.random.rand(n, n) + np.random.rand(n, n) * 1j
    m = np.matmul(m, m.T)
    m[0][0] = 1.0
    m[1][1] = -1.0
    for i, val in enumerate(kappa):
        matrices.append(m.copy())
        matrices[i][0][1] = val
        matrices[i][1][0] = val
    return np.array(matrices)


def matrix_one_close_re(kappa):
    """Creates symmetric matrices

    Creates 5 dimensional symmetric matrices with the simple model in the top left corner. One resonance is close to
    the EP.

    Parameters
    ----------
    kappa : np.ndarray
        Contains all kappa values for the off diagonal elements in the simple model.

    Returns
    ----------
    np.ndarray
        Returns an array which contains a 5 dimensional symmetric matrix for each kappa value of the circle
        parametrization.
    """
    matrices = []
    m = np.array([[1. + 0.j, 1.5 + 0.5j,
                   1.61335586 + 0.j, 1.16219976 + 0.j,
                   1.75018533 + 0.j],
                  [1.5 + 0.5j, -1. + 0.j,
                   1.8117192 + 0.j, 1.52927993 + 0.j,
                   1.47493566 + 0.j],
                  [1.61335586 + 0.j, 1.8117192 + 0.j,
                   2.42166818 + 0.j, 2.03118958 + 0.j,
                   1.40894905 + 0.j],
                  [1.16219976 + 0.j, 1.52927993 + 0.j,
                   2.03118958 + 0.j, 1.8493137 + 0.j,
                   0.80460232 + 0.j],
                  [1.75018533 + 0.j, 1.47493566 + 0.j,
                   1.40894905 + 0.j, 0.80460232 + 0.j,
                   2.05262763 + 0.j]])
    m[0][0] = 1.0
    m[1][1] = -1.0
    for i, val in enumerate(kappa):
        matrices.append(m.copy())
        matrices[i][0][1] = val
        matrices[i][1][0] = val
    return np.array(matrices)


def matrix_two_close_im(kappa):
    """Creates complex symmetric matrices

    Creates 5 dimensional complex symmetric matrices with the simple model in the top left corner. Two resonances are
    close to the EP.

    Parameters
    ----------
    kappa : np.ndarray
        Contains all kappa values for the off diagonal elements in the simple model.

    Returns
    ----------
    np.ndarray
        Returns an array which contains a 5 dimensional complex symmetric matrix for each kappa value of the circle
        parametrization.
    """
    matrices = []
    m = np.array([[1. + 0.j, 1.5 + 0.5j,
                   0.83666469 + 1.76291345j, 0.77765868 + 1.27203888j,
                   0.77484115 + 2.52414992j],
                  [1.5 + 0.5j, -1. + 0.j,
                   0.49205075 + 2.48351576j, 0.53956258 + 2.0062303j,
                   -0.36174332 + 3.62957809j],
                  [0.83666469 + 1.76291345j, 0.49205075 + 2.48351576j,
                   1.27425649 + 1.91246804j, 0.85158089 + 1.65873395j,
                   0.98463941 + 3.18559508j],
                  [0.77765868 + 1.27203888j, 0.53956258 + 2.0062303j,
                   0.85158089 + 1.65873395j, 1.19988497 + 1.18525684j,
                   0.7565852 + 3.01046936j],
                  [0.77484115 + 2.52414992j, -0.36174332 + 3.62957809j,
                   0.98463941 + 3.18559508j, 0.7565852 + 3.01046936j,
                   -0.07367223 + 5.11505034j]])
    m[0][0] = 1.0
    m[1][1] = -1.0
    for i, val in enumerate(kappa):
        matrices.append(m.copy())
        matrices[i][0][1] = val
        matrices[i][1][0] = val
    return np.array(matrices)


def eigenvalues(matrices):
    """Get the eigenvalues of a matrix

    Parameters
    ----------
    matrices : np.ndarray
        Array containing one or more matrices of n dimension.

    Returns
    ----------
    np.ndarray
        Returns all eigenvalues for every matrix in the input array. The shape of this array depends on the Number of
        matrices M and their dimension n: (M, n).
    """
    w = np.linalg.eigvals(matrices)
    return w


def ev_one_close(kappa):
    mat = matrix_one_close_re(kappa)
    return np.linalg.eigvals(mat)


def ev_two_close(kappa):
    mat = matrix_two_close_im(kappa)
    return np.linalg.eigvals(mat)
