"""Perform gaussian processes with GPflow"""

import numpy as np
import gpflow
from gpflow.utilities import print_summary


def gp_diff_angle(ev, phi):
    """Gaussian process model for eigenvalue difference with respect to the angle

    GPflow is used to make a prediction model for the eigenvalue (ev) difference with respect to the angle (phi).
    Due to complex eigenvalues there are two models. One for the real and one for the imaginary part.
    Only works for 2D matrix.

    Parameters
    ----------
    ev : np.ndarray
        2D complex array which contains the eigenvalues of the 2D matrix
    phi : np.ndarray
        1D array which contains the related angles

    Returns
    ----------
    (gpflow.models.GPR, gpflow.models.GPR)
        Returns two GPR models. One GPR model for the real part of the eigenvalue difference and one GPR model for the
        imaginary part of the eigenvalue difference
    """
    ev_diff_re = ((ev[::, 0] - ev[::, 1]) ** 2).real.reshape(-1, 1)
    ev_diff_im = ((ev[::, 0] - ev[::, 1]) ** 2).imag.reshape(-1, 1)
    phi = phi.reshape(-1, 1)
    k = gpflow.kernels.Matern52()
    m_re = gpflow.models.GPR(data=(phi, ev_diff_re), kernel=k, mean_function=None)
    m_im = gpflow.models.GPR(data=(phi, ev_diff_im), kernel=k, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    # opt_logs_re = opt.minimize(m_re.training_loss, m_re.trainable_variables, options=dict(maxiter=100))
    # opt_logs_im = opt.minimize(m_im.training_loss, m_im.trainable_variables, options=dict(maxiter=100))
    opt.minimize(m_re.training_loss, m_re.trainable_variables)
    opt.minimize(m_im.training_loss, m_im.trainable_variables)
    return m_re, m_im


def gp_diff_kappa(ev, kappa):
    ev_diff_re = ((ev[::, 0] - ev[::, 1]) ** 2).real.reshape(-1, 1)  # .astype(np.float64)
    ev_diff_im = ((ev[::, 0] - ev[::, 1]) ** 2).imag.reshape(-1, 1)  # .astype(np.float64)
    kappa_sep = np.column_stack([kappa.real, kappa.imag])  # .astype(np.float64)
    k = gpflow.kernels.RBF()
    m_re = gpflow.models.GPR(data=(kappa_sep, ev_diff_re), kernel=k, mean_function=None)
    m_im = gpflow.models.GPR(data=(kappa_sep, ev_diff_im), kernel=k, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    # opt_logs_re = opt.minimize(m_re.log_marginal_likelihood, m_re.trainable_variables, options=dict(maxiter=1000))
    # opt_logs_im = opt.minimize(m_im.log_marginal_likelihood, m_im.trainable_variables, options=dict(maxiter=1000))
    # opt_logs_re = opt.minimize(m_re.training_loss, m_re.trainable_variables)  # , method='krylov')
    # opt_logs_im = opt.minimize(m_im.training_loss, m_im.trainable_variables)  # , method='krylov')
    opt.minimize(m_re.training_loss, m_re.trainable_variables)
    opt.minimize(m_im.training_loss, m_im.trainable_variables)
    return m_re, m_im
