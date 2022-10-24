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


def two_close_5d_works():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    m = False
    n = False
    j = 0
    i = 0
    p = 16
    ep = np.array([1.18503351 + 1.00848184j])
    ep_two_close = np.array([0.88278093 + 1.09360073j])
    eps = 1.e-15
    eps_var = 1.e-6
    eps_diff = 5.e-3
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    # df = pd.read_csv('model_noise_dependency_55_color.csv', header=0, skiprows=0,
    #                 names=["kappa_with_noise", "kappa_no_noise", "ev_with_noise", "ev_no_noise",
    #                        "training_steps_color"])
    # kappa = np.array(df.kappa).astype(complex)
    # kappa_new = kappa
    # kappa_no_noise = np.array(df.kappa_no_noise).astype(complex)
    # kappa_with_noise = np.array(df.kappa_with_noise).astype(complex)
    # training_steps_color = np.array(df.training_steps_color)
    # df1 = pd.DataFrame(df.ev_with_noise)
    # ev = df1.to_numpy()
    # ev = np.array([ast.literal_eval(df1.to_numpy()[i, 0]) for i in range(len(df1.to_numpy()[:, 0]))])
    # ev = np.empty((0, 2))
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    # plots.energy_plane_plotly(ev_new, phi)
    ev = dpp.initial_dataset(ev_new)
    # ev = ev[::11]
    # plots.energy_plane_plotly(ev, phi)
    # ev_no_noise = ev[::11]
    ev = ev[::11]
    kappa = kappa[::11]
    entropy_diff = []
    entropy_sum = []
    training_steps = []
    training_steps_color = [0 for _ in kappa]
    print(kappa)
    # plots.energy_plane_plotly(ev, phi)
    # training_steps_color = [i for _ in kappa_with_noise]
    # model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa_with_noise)
    # model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa_with_noise)
    # ev_diff_complex = ((ev[::, 0] - ev[::, 1]) ** 2)
    # ev_sum_complex = (0.5 * (ev[::, 0] + ev[::, 1]))
    # plots.control_model_2d_plotly(kappa_with_noise[24:], ev_diff_complex[24:], ev_sum_complex[24:], model_diff,
    #                              model_sum)
    # plots.model_noise_dependency_plotly(kappa_with_noise, kappa_no_noise, training_steps_color)
    # df = pd.DataFrame()
    while not n:
        model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
        model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
        # xx, yy = np.meshgrid(kappa_with_noise[-1].real, kappa_with_noise[-1].imag)
        # grid = np.array((xx.ravel(), yy.ravel())).T
        # mean_diff_old, var_diff_old = model_diff_with_noise.predict_f(grid)
        # mean_sum_old, var_sum_old = model_sum_with_noise.predict_f(grid)
        if np.any(abs(kernel_ev_diff) < eps):
            print("DIFF MODEL FULLY TRAINED!")
            print(kernel_ev_diff)
            # n = True
        if np.any(abs(kernel_ev_sum) < eps):
            print("SUM MODEL FULLY TRAINED!")
        # plt.figure(i)
        """entropy_step_diff = -np.sum((abs(kernel_ev_diff) / np.sum(kernel_ev_diff)) *
                                    np.log(abs(kernel_ev_diff) / np.sum(kernel_ev_diff))) / np.log(len(kernel_ev_diff))
        entropy_step_sum = -np.sum((abs(kernel_ev_sum) / len(kernel_ev_sum)) *
                                   np.log(abs(kernel_ev_sum) / len(kernel_ev_sum)))
        entropy_diff.append(entropy_step_diff)
        entropy_sum.append(entropy_step_sum)"""
        training_steps.append(i)
        # df = pd.concat([df, pd.Series(kernel_ev_diff.tolist())], ignore_index=True, axis=1)
        # plt.ylabel("Entropy")
        # plt.xlabel("# of kappa points")
        # plt.plot(entropy)
        # print(entropy)
        # plt.subplot(1, 2, 1)
        # plt.xlabel("Re(\\kappa)")
        # plt.ylabel("Im(\\kappa)")
        # plt.scatter(x=kappa[:].real, y=kappa[:].imag, c=plot_color, cmap="plasma")
        # plt.subplot(1, 2, 2)
        # plt.ylabel("Kernel eigenvalues")
        # plt.xlabel("# of kappa points")
        # plt.yscale("log")
        # plt.plot(kernel_ev)
        # plt.savefig("entropy_step_%1d" % i)
        # df = pd.DataFrame()
        # df['kappa'] = kappa.tolist()
        # df['kernel_ev'] = kernel_ev.tolist()
        # df['ev_diff'] = ev_diff.tolist()
        # df = pd.concat([df, pd.DataFrame(ev)], axis=1)
        # df.columns = ['kappa', 'kernel_ev', 'ev_diff', 'ev1', 'ev2']
        # df.to_csv('data_step_%1d.csv' % i)
        gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
        gpflow_function = gpflow_model.get_model_generator()
        kappa_new = zs.zero_search(gpflow_function, kappa_0)
        kappa = np.concatenate((kappa, kappa_new))
        ev = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum)
        i += 1
        training_steps_color.append(i)
        if i == 2:
            kappa_extra = 2 * kappa[-1] - kappa[-2]
            kappa = np.concatenate((kappa, np.array([complex(kappa_extra.real, kappa_extra.imag)])))
            ev = dpp.getting_new_ev_of_ep(np.array([complex(kappa_extra.real, kappa_extra.imag)]),
                                          ev, model_diff, model_sum)
            training_steps_color.append(i)
        print("Root with noise: ", kappa_new)
        print("Real EP: ", ep)
        ev_diff = ev[-1, 0] - ev[-1, 1]
        print(ev_diff)
        print(ev_diff.real)
        print(ev_diff.imag)
        if abs(ev_diff.real) < eps_diff and abs(ev_diff.imag) < eps_diff:
            print("Could find EP:")
            # print("j: ", j)
            print(kappa_new)
            print("Real EP: ")
            # print(ep)
            print(ep_two_close)
            print("Eigenvalue difference:")
            print(ev_diff)
            # print(ev_with_noise)
            n = True
        if i == 25:
            print("Could not find EP yet:")
            # print("j: ", j)
            print(kappa_new)
            print("Real EP: ")
            # print(ep)
            print(ep_two_close)
            n = True
    fig1 = px.scatter(x=kappa.real, y=kappa.imag, color=training_steps_color,
                      labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    # fig2 = px.scatter(x=kappa_no_noise.real, y=kappa_no_noise.imag, color=training_steps_color,
    #                  labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    fig_ep = px.scatter(x=ep_two_close.real, y=ep_two_close.imag, color_discrete_sequence=['#00CC96'], color=["EP"],
                        labels=dict(x="Re(EP)", y="Im(EP)"))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(fig1["data"][0], row=1, col=1)
    fig.add_trace(fig_ep["data"][0], row=1, col=1)
    fig.update_xaxes(title_text="Re(kappa)", row=1, col=1)
    fig.update_yaxes(title_text="Im(kappa)", row=1, col=1)
    fig.update_layout(coloraxis={'colorbar': dict(title="# of training steps", len=0.9)})
    fig.show()
    # fig.write_html("docs/source/_pages/images/trained_parameterspace_extra3")
    fig1 = plt.figure(1, figsize=(10, 7.5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_ylabel("Entropy")
    ax1.set_xlabel("\\# of training steps")
    ax1.plot(entropy_diff)
    # fig1.savefig("entropy_diff.pdf")
    df = pd.DataFrame()
    df['kappa'] = kappa.tolist()
    # df['kappa_no_noise'] = kappa_no_noise.tolist()
    df = pd.concat([df, pd.DataFrame(ev)], axis=1)
    # df['ev'] = ev_with_noise.tolist()
    # df['ev_no_noise'] = ev_no_noise.tolist()
    df['training_steps_color'] = training_steps_color
    # df = pd.concat([df, pd.DataFrame(ev)], axis=1)
    df.columns = ['kappa', 'ev1', 'ev2', 'training_steps_color']
    # df.to_csv('paper_data_kappa_trained_model_extra3.csv')


if __name__ == '__main__':
    # exact_ep()
    two_close_5d_works()
    """kappa_0 = 1.247 + 88.698 * 1j
    guess = 1. + 88.j
    kappa, ev, phi = dpp.load_dat_file("../Punkt23/output_031_1.dat")
    # kappa = kappa[::4]
    # phi = phi[::4]
    training_steps_color = [0 for _ in kappa]
    ev = dpp.initial_dataset(ev)
    # ev = ev[::4]
    print(ev)
    ev_diff_complex = ((ev[::, 0] - ev[::, 1]) ** 2)  # .astype(np.float64)
    print(np.max(abs(ev_diff_complex.real)))
    print(np.max(abs(ev_diff_complex.imag)))
    print(np.max([np.max(abs(ev_diff_complex.real)), np.max(abs(ev_diff_complex.imag))]))
    ev_diff = np.column_stack([ev_diff_complex.real, ev_diff_complex.imag])
    print(np.max(abs(ev_diff)))
    ev_diff = ev_diff / np.max(abs(ev_diff))
    print(ev_diff)
    print(np.shape(ev_diff))
    ev_sum_complex = (0.5 * (ev[::, 0] + ev[::, 1]))
    ev_sum_real = (ev_sum_complex.real - np.mean(ev_sum_complex.real))
    ev_sum_imag = (ev_sum_complex.imag - np.mean(ev_sum_complex.imag))
    ev_sum = np.column_stack([ev_sum_real, ev_sum_imag])
    print(np.max(abs(ev_sum)))
    ev_sum = ev_sum / np.max(abs(ev_sum))
    print(ev_sum)
    print(ev_sum.shape)
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
    plots.parameter_plane_plotly(kappa, training_steps_color)
    print("Predicted EP: ", kappa_new)"""

