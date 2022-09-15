import jax.numpy as np
import numpy
from jax import vmap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matrix


def load_dat_file(filename):
    df = pd.read_csv(filename, sep='\s+', skiprows=7, skip_blank_lines=False,
                     names=["Z1", "Z2", "Z3", "Z4", "Z5", "f", "gamma", "Z8", "Z9", "Z10", "Z11", "ev_re", "ev_im",
                            "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", "Z22", "Z23", "Z24", "Z25", "Z26",
                            "Z27", "Z28", "Z29", "Z30", "phi"])

    f = clump(numpy.array(df.f))
    gamma = clump(numpy.array(df.gamma))
    kappa = gamma + f * 1j
    ev_re = clump(numpy.array(df.ev_re))
    ev_im = clump(numpy.array(df.ev_im))
    ev = ev_re + ev_im * 1j
    phi = clump(numpy.array(df.phi))
    return kappa[:, 0], ev, phi[:, 0]


def clump(a):
    return numpy.array([a[s] for s in numpy.ma.clump_unmasked(numpy.ma.masked_invalid(a))])


def initial_dataset(ev):
    """Get initial dataset

    Selecting the eigenvalues belonging to the EP by ordering all eigenvalues and comparing the first end last point.
    If it is greater than 0 the eigenvalues exchange their positions and belong to the EP.

    Parameters
    ----------
    ev : np.ndarray
        All exact complex eigenvalues

    Returns
    -------
    np.ndarray
        Usually 2D array containing the two eigenvalues belonging to the EP
    """
    nearest_neighbour = np.array(
        [vmap(lambda x, y: abs(x - y), in_axes=(0, 0), out_axes=0)(ev, np.roll(np.roll(ev, -1, axis=0), -i, axis=1))
         for i in range(np.shape(ev)[1])])
    ev_all = []
    for i in range(np.shape(nearest_neighbour)[2]):
        ev_sorted = [ev[0, i]]
        l = i
        for j in range(np.shape(nearest_neighbour)[1]):
            l = (np.argmin(nearest_neighbour[:, j, l]) + l) % np.shape(ev)[1]
            if j + 1 != np.shape(ev)[0]:
                ev_sorted.append(ev[(j + 1), l])
        ev_all.append(ev_sorted)
    ev_all_sorted = numpy.column_stack([ev_all[k] for k in range(np.shape(ev_all)[0])])
    ep_ev_index = np.argwhere(abs(ev_all_sorted[0, :] - ev_all_sorted[-1, :]) > 3.e-6)
    # print(type(numpy.array(ev_all_sorted[:, ep_ev_index])))
    return numpy.column_stack([ev_all_sorted[:, n] for n in ep_ev_index])  # ev_all_sorted[:, ep_ev_index])


def getting_new_ev_of_ep_old(kappa, ev, model_diff, model_sum):
    """Getting new eigenvalues belonging to the EP

    Selecting the two eigenvalues of a new point belonging to the EP by comparing it to a GPR model prediction and
    its variance.

    Parameters
    ----------
    kappa : np.ndarray
        All complex kappa values
    ev : np.ndarray
        Containing all old eigenvalues belonging to the EP
    model_diff : gpflow.models.GPR
        2D GPR model for eigenvalue difference squared
    model_sum : gpflow.models.GPR
        2D GPR model for eigenvalue sum

    Returns
    -------
    np.ndarray
        2D array containing all old and the new eigenvalues belonging to the EP
    """
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    xx, yy = numpy.meshgrid(kappa.real, kappa.imag)
    grid = numpy.array((xx.ravel(), yy.ravel())).T
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    pairs_diff_all = np.empty(0)
    pairs_sum_all = np.empty(0)
    ev_1 = np.empty(0)
    ev_2 = np.empty(0)
    for i, val in enumerate(ev_new[0, ::]):
        pairs_diff = vmap(lambda a, b: np.power((a - b), 2), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        pairs_sum = vmap(lambda a, b: 0.5 * np.add(a, b), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        ev_1 = np.concatenate((ev_1, np.array([val for _ in range(len(ev_new[0, (i + 1):]))])))
        ev_2 = np.concatenate((ev_2, ev_new[0, (i + 1):]))
        pairs_diff_all = np.concatenate((pairs_sum_all, np.array(pairs_diff)))
        pairs_sum_all = np.concatenate((pairs_sum_all, np.array(pairs_sum)))
    compatibility = - np.power(pairs_diff_all.real - mean_diff.numpy()[0, 0], 2) / (2 * var_diff.numpy()[0, 0]) \
                    - np.power(pairs_diff_all.imag - mean_diff.numpy()[0, 1], 2) / (2 * var_diff.numpy()[0, 1]) \
                    - np.power(pairs_sum_all.real - mean_sum.numpy()[0, 0], 2) / (2 * var_sum.numpy()[0, 0]) \
                    - np.power(pairs_sum_all.imag - mean_sum.numpy()[0, 1], 2) / (2 * var_sum.numpy()[0, 1])
    c = np.array([0 for _ in compatibility])
    fig1 = px.scatter(x=c, y=abs(compatibility), log_y=True)
    fig1.show()
    new = np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers', name="All eigenvalues"))
    fig.add_trace(go.Scatter(x=new.ravel().real, y=new.ravel().imag, mode='markers', name="Eigenvalues of EP",
                             marker=dict(color='red')))
    fig.show()
    return numpy.concatenate((ev, np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])))


def getting_new_ev_of_ep(kappa, ev, model_diff, model_sum):
    """Getting new eigenvalues belonging to the EP

    Selecting the two eigenvalues of a new point belonging to the EP by comparing it to a GPR model prediction and
    its variance.

    Parameters
    ----------
    kappa : np.ndarray
        All complex kappa values
    ev : np.ndarray
        Containing all old eigenvalues belonging to the EP
    model_diff : gpflow.models.GPR
        2D GPR model for eigenvalue difference squared
    model_sum : gpflow.models.GPR
        2D GPR model for eigenvalue sum

    Returns
    -------
    np.ndarray
        2D array containing all old and the new eigenvalues belonging to the EP
    """
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    grid = numpy.column_stack((kappa.real, kappa.imag))
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    pairs_diff_all = np.empty(0)
    pairs_sum_all = np.empty(0)
    pairs_difference = np.empty(0)
    ev_1 = np.empty(0)
    ev_2 = np.empty(0)
    for i, val in enumerate(ev_new[0, ::]):
        pairs_diff_squared = vmap(lambda a, b: np.power((a - b), 2), in_axes=(None, 0), out_axes=0)(val,
                                                                                                    ev_new[0, (i + 1):])
        pairs_sum = vmap(lambda a, b: 0.5 * np.add(a, b), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        pairs_diff = vmap(lambda a, b: abs(a - b), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        ev_1 = np.concatenate((ev_1, np.array([val for _ in range(len(ev_new[0, (i + 1):]))])))
        ev_2 = np.concatenate((ev_2, ev_new[0, (i + 1):]))
        pairs_diff_all = np.concatenate((pairs_diff_all, np.array(pairs_diff_squared)))
        pairs_sum_all = np.concatenate((pairs_sum_all, np.array(pairs_sum)))
        pairs_difference = np.concatenate((pairs_difference, np.array(pairs_diff)))
    compatibility = - np.power(pairs_diff_all.real - mean_diff.numpy()[0, 0], 2) / (2 * var_diff.numpy()[0, 0]) \
                    - np.power(pairs_diff_all.imag - mean_diff.numpy()[0, 1], 2) / (2 * var_diff.numpy()[0, 1]) \
                    - np.power(pairs_sum_all.real - mean_sum.numpy()[0, 0], 2) / (2 * var_sum.numpy()[0, 0]) \
                    - np.power(pairs_sum_all.imag - mean_sum.numpy()[0, 1], 2) / (2 * var_sum.numpy()[0, 1])
    c = np.array([0 for _ in compatibility])
    fig1 = px.scatter(x=c, y=abs(compatibility), log_y=True)
    # fig1.write_html("docs/source/_pages/images/compatibility_%1d.html" % np.shape(ev)[0])
    # fig1.show()
    new = np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])
    new_diff = np.array([[ev_1[np.argmin(pairs_difference)], ev_2[np.argmin(pairs_difference)]]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers', name="All eigenvalues"))
    fig.add_trace(go.Scatter(x=new.ravel().real, y=new.ravel().imag, mode='markers', name="Eigenvalues of EP",
                             marker=dict(color='red')))
    # fig.write_html("docs/source/_pages/images/selected_eigenvalues_%1d.html" % np.shape(ev)[0])
    # fig.show()
    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers', name="All eigenvalues"))
    fig_diff.add_trace(go.Scatter(x=new_diff.ravel().real, y=new_diff.ravel().imag, mode='markers',
                                  name="Eigenvalues of EP", marker=dict(color='red')))
    # fig_diff.write_html("docs/source/_pages/images/selected_eigenvalues_%1d.html" % np.shape(ev)[0])
    # fig_diff.show()
    return numpy.concatenate((ev, np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])))
