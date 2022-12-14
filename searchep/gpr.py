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


def gp_2d_diff_kappa(data):
    """2D gaussian process model for eigenvalue difference with respect to kappa

    GPflow is used to make a prediction model for the eigenvalue (ev) difference with respect to kappa.
    Due to complex kappa- and eigenvalues it is a 2D on 2D model.

    Parameters
    ----------
    data : data_preprocessing.Data
        Class which contains all scale-, kappa- and eigenvalues
    # ev : np.ndarray
    #     2D complex array which contains the two eigenvalues belonging to the EP
    # kappa : np.ndarray
    #     1D array which contains the complex kappa values

    Returns
    ----------
    (gpflow.models.GPR, np.ndarray)
        Returns a 2D GPR model for the complex eigenvalue difference squared with respect to kappa and a 1D array which
        contains the kernel eigenvalues of the input space.
    """
    ev_diff_complex = ((data.ev[::, 0] - data.ev[::, 1]) ** 2)  # .astype(np.float64)
    ev_diff = np.column_stack((ev_diff_complex.real, ev_diff_complex.imag))
    ev_diff = ev_diff / data.diff_scale
    kappa_sep = np.column_stack((data.kappa_scaled.real, data.kappa_scaled.imag))  # .astype(np.float64)
    model, kernel_ev = gp_create_matern52_model(kappa_sep, ev_diff, data)
    return model, kernel_ev


def gp_2d_sum_kappa(data):
    """2D gaussian process model for eigenvalue sum with respect to kappa

    GPflow is used to make a prediction model for the eigenvalue (ev) sum with respect to kappa.
    Due to complex kappa- and eigenvalues it is a 2D on 2D model.

    Parameters
    ----------
    data : data_preprocessing.Data
        Class which contains all scale-, kappa- and eigenvalues
    # ev : np.ndarray
    #     2D complex array which contains the two eigenvalues belonging to the EP
    # kappa : np.ndarray
    #     1D array which contains the complex kappa values

    Returns
    ----------
    (gpflow.models.GPR, np.ndarray)
        Returns a 2D GPR model for the complex eigenvalue sum with respect to kappa and a 1D array which
        contains the kernel eigenvalues of the input space.
    """
    ev_sum_complex = (0.5 * (data.ev[::, 0] + data.ev[::, 1]))
    ev_sum_real = (ev_sum_complex.real - data.sum_mean_complex.real)
    ev_sum_imag = (ev_sum_complex.imag - data.sum_mean_complex.imag)
    ev_sum = np.column_stack((ev_sum_real, ev_sum_imag))
    # ev_sum = np.column_stack([ev_sum_complex.real, ev_sum_complex.imag])
    ev_sum = ev_sum / data.sum_scale
    kappa_sep = np.column_stack((data.kappa_scaled.real, data.kappa_scaled.imag))
    model, kernel_ev = gp_create_matern52_model(kappa_sep, ev_sum, data)
    return model, kernel_ev


def gp_2d_diff_kappa_no_noise(ev, kappa):
    ev_diff_complex = ((ev[::, 0] - ev[::, 1]) ** 2)  # .astype(np.float64)
    ev_diff = np.column_stack([ev_diff_complex.real, ev_diff_complex.imag])
    kappa_sep = np.column_stack([kappa.real, kappa.imag])  # .astype(np.float64)
    model, kernel_ev = gp_create_matern52_model_no_noise(kappa_sep, ev_diff)
    return model, kernel_ev


def gp_2d_sum_kappa_no_noise(ev, kappa):
    ev_sum_complex = (0.5 * (ev[::, 0] + ev[::, 1]))
    ev_sum = np.column_stack([ev_sum_complex.real, ev_sum_complex.imag])
    kappa_sep = np.column_stack([kappa.real, kappa.imag])
    model, kernel_ev = gp_create_matern52_model_no_noise(kappa_sep, ev_sum)
    return model, kernel_ev


def gp_create_rbf_model(kappa, validation_data, data):
    """2D gaussian process model with rbf kernel

    GPflow is used to make a 2D prediction model with the rbf kernel.

    Parameters
    ----------
    kappa : np.ndarray
        2D array which contains the real and the imaginary part of all kappa values
    validation_data : np.ndarray
        2D array which contains usually the real and imaginary part of the validation data

    Returns
    ----------
    (gpflow.models.GPR, np.ndarray)
        Returns a 2D GPR model created with the rbf kernel and a 1D array which
        contains the kernel eigenvalues of the input space.
    """
    k = gpflow.kernels.RBF()
    kernel_ev = np.linalg.eigvals(k.K(kappa))
    model = gpflow.models.GPR(data=(kappa, validation_data), kernel=k, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    return model, kernel_ev


def gp_create_matern52_model(kappa, validation_data, data):
    """2D gaussian process model with matern52 kernel

    GPflow is used to make a 2D prediction model with the matern52 kernel.

    Parameters
    ----------
    kappa : np.ndarray
        2D array which contains the real and the imaginary part of all kappa values
    validation_data : np.ndarray
        2D array which contains usually the real and imaginary part of the validation data

    Returns
    ----------
    (gpflow.models.GPR, np.ndarray)
        Returns a 2D GPR model created with the matern52 kernel and a 1D array which
        contains the kernel eigenvalues of the input space.
    """
    try:
        k = gpflow.kernels.Matern52(lengthscales=[3., 0.1])  # gpflow.kernels.Matern52(active_dims=[0]) + gpflow.kernels.Matern52(active_dims=[1])
        kernel_ev = np.linalg.eigvals(k.K(kappa))
        if np.any(kernel_ev < 0):
            raise FloatingPointError("Negative kernel eigenvalues.\n\tMatrix is not invertible.")
        model = gpflow.models.GPR(data=(kappa, validation_data), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        return model, kernel_ev
    except FloatingPointError:
        print("Error: Negative kernel eigenvalues.\n\tMatrix is not invertible.")
        data.exception = True
        return 0, 0
    except Exception as e:
        print("The error raised is: ", e)
        data.exception = True
        return 0, 0


def gp_create_matern52_model_no_noise(kappa, validation_data):
    k = gpflow.kernels.Matern52()
    kernel_ev = np.linalg.eigvals(k.K(kappa))
    model = gpflow.models.GPR(data=(kappa, validation_data), kernel=k, mean_function=None, noise_variance=1.00000001e-6)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    return model, kernel_ev
