import numpy as np
import matrix
import plots
import time
import gaussian_process_regression as gpr
import zero_search as zs
import matplotlib.pyplot as plt
import GPFlow_model_class as GPFmc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import jax.numpy as jnp
from jax import vmap
import data_preprocessing as dpp
import ast

ep_one_close = 1.18503337 + 1.00848169j
ep_two_close = 0.88278093 + 1.09360073j


def exact_ep():
    kappa, phi = matrix.parametrization(ep_two_close, 1e-8, 200)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev)
    plots.energy_plane_plotly(ev, phi)


if __name__ == '__main__':
    # exact_ep()
    # mat = matrix.matrix_one_close_re(np.array([ep]))
    # ev = matrix.eigenvalues(mat)
    # print(ev)
    # print(ev[0, 2]-ev[0, 3])
    kappa_0 = 1.247 + 88.698 * 1j
    guess = 1. + 88.j
    kappa, ev, phi = dpp.load_dat_file("../Punkt23/output_031_1.dat")
    kappa = kappa[::4]
    phi = phi[::4]
    training_steps_color = [0 for _ in kappa]
    ev = dpp.initial_dataset(ev)
    ev = ev[::4]
    print(ev)
    ev_diff_complex = ((ev[::, 0] - ev[::, 1]) ** 2)  # .astype(np.float64)
    ev_diff = np.column_stack([ev_diff_complex.real, ev_diff_complex.imag])
    print(ev_diff)
    print(np.shape(ev_diff))
    plots.energy_plane_plotly(ev, phi)
    model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
    model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    plots.three_d_eigenvalue_kappa_2d_model_plotly(kappa_0, 0.003, model_diff)
    plots.three_d_eigenvalue_kappa_2d_model_plotly(kappa_0, 0.003, model_sum)
    gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
    gpflow_function = gpflow_model.get_model_generator()
    kappa_new = zs.zero_search(gpflow_function, kappa_0)
    kappa = np.concatenate((kappa, kappa_new))
    training_steps_color.append(1)
    # plots.parameter_plane_plotly(kappa, training_steps_color)
    print("Predicted EP: ", kappa_new)
