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

ep = 1.185034 + 1.008482j
ep_two_close = 0.88278093 + 1.09360073j

if __name__ == '__main__':
    plots.init_matplotlib()

    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    m = False
    n = False
    j = 0
    i = 0
    p = 16
    eps = 1.e-15
    eps_var = 1.e-6
    eps_diff = 1.e-2
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
    ev_with_noise = ev[::11]
    kappa_with_noise = kappa[::11]
    # kappa_no_noise = kappa[::11]
    entropy_diff = []
    entropy_sum = []
    training_steps = []
    training_steps_color = [0 for _ in kappa_with_noise]
    print(kappa_with_noise)
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
        model_diff_with_noise, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev_with_noise, kappa_with_noise)
        model_sum_with_noise, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev_with_noise, kappa_with_noise)
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
        # model_diff_no_noise, kernel_ev_diff = gpr.gp_2d_diff_kappa_no_noise(ev_no_noise, kappa_no_noise)
        # model_sum_no_noise, kernel_ev_sum = gpr.gp_2d_sum_kappa_no_noise(ev_no_noise, kappa_no_noise)
        # print("GPR old ev difference squared: ", mean_diff_old.numpy().ravel())
        # print("Variance diff old: ", var_diff_old.numpy())
        # print("Variance sum old: ", var_sum_old.numpy())
        # if np.any(kernel_ev_diff < 0) or np.any(kernel_ev_sum < 0) or np.any(type(kernel_ev_diff[::]) ==
        # complex) or np.any(type(kernel_ev_sum[::]) == complex):
        #    break
        # plt.figure(i)
        entropy_step_diff = -np.sum((abs(kernel_ev_diff) / np.sum(kernel_ev_diff)) *
                                    np.log(abs(kernel_ev_diff) / np.sum(kernel_ev_diff))) / np.log(len(kernel_ev_diff))
        entropy_step_sum = -np.sum((abs(kernel_ev_sum) / len(kernel_ev_sum)) *
                                   np.log(abs(kernel_ev_sum) / len(kernel_ev_sum)))
        entropy_diff.append(entropy_step_diff)
        entropy_sum.append(entropy_step_sum)
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
        gpflow_model_with_noise = GPFmc.GPFlowModel(model_diff_with_noise, model_sum_with_noise)
        gpflow_function_with_noise = gpflow_model_with_noise.get_model_generator()
        kappa_new_with_noise = zs.zero_search(gpflow_function_with_noise, kappa_0)
        kappa_with_noise = np.concatenate((kappa_with_noise, kappa_new_with_noise))
        ev_with_noise = dpp.getting_new_ev_of_ep(kappa_new_with_noise, ev_with_noise, model_diff_with_noise,
                                                 model_sum_with_noise)
        # gpflow_model_no_noise = GPFmc.GPFlowModel(model_diff_no_noise, model_sum_no_noise)
        # gpflow_function_no_noise = gpflow_model_no_noise.get_model_generator()
        # kappa_new_no_noise = zs.zero_search(gpflow_function_no_noise, kappa_0)
        # kappa_no_noise = np.concatenate((kappa_no_noise, kappa_new_no_noise))
        # ev_no_noise = dpp.getting_new_ev_of_ep(kappa_new_no_noise, ev_no_noise, model_diff_no_noise,
        #                                       model_sum_no_noise)
        i += 1
        training_steps_color.append(i)
        print("Root with noise: ", kappa_new_with_noise)
        print("Real EP: ", ep)
        ev_diff = ev_with_noise[-1, 0]-ev_with_noise[-1, 1]
        print(ev_diff)
        print(ev_diff.real)
        print(ev_diff.imag)
        # print("Ev difference: ", ev[-1, 0]-ev[-1, 1])
        # print("Ev difference squared: ", (ev[-1, 0] - ev[-1, 1]) ** 2)
        # print(ev[-1, 0] - ev[-1, 1])
        # print(kernel_ev)
        # grid = np.column_stack((kappa_new_with_noise.real, kappa_new_with_noise.imag))
        # grid = np.array((xx.ravel(), yy.ravel())).T
        # mean_diff, var_diff = model_diff_with_noise.predict_f(grid)
        # mean_sum, var_sum = model_sum_with_noise.predict_f(grid)
        # print("GPR ev difference squared: ", mean_diff.numpy().ravel())
        if abs(ev_diff.real) < eps_diff and abs(ev_diff.imag) < eps_diff:
            print("Could find EP:")
            # print("j: ", j)
            print(kappa_new_with_noise)
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
            print(kappa_new_with_noise)
            print("Real EP: ")
            # print(ep)
            print(ep_two_close)
            # print(ev_with_noise)
            # print("Variance diff: ", var_diff.numpy())
            # print("Variance sum: ", var_sum.numpy())
            # print("Ev1: ", ev[-1, 0])
            # print("Ev2: ", ev[-1, 1])
            # print("Ev difference: ", ev[-1, 0] - ev[-1, 1])
            # print("GPR ev difference squared: ", mean_diff.numpy().ravel())
            # print("GPR ev difference: ", np.sqrt(mean_diff.numpy().ravel()))
            # print(var)
            # print(var_re)
            # print(var_im)
            n = True
    ep = np.array([1.18503351 + 1.00848184j])
    ep_two_close = np.array([0.88278093 + 1.09360073j])
    fig1 = px.scatter(x=kappa_with_noise.real, y=kappa_with_noise.imag, color=training_steps_color,
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
    print(ev_with_noise)
    # fig_no_noise = make_subplots(rows=1, cols=1)
    # fig_no_noise.add_trace(fig2["data"][0], row=1, col=1)
    # fig_no_noise.add_trace(fig_ep["data"][0], row=1, col=1)
    # fig_no_noise.update_xaxes(title_text="Re(kappa)", row=1, col=1)
    # fig_no_noise.update_yaxes(title_text="Im(kappa)", row=1, col=1)
    # fig_no_noise.update_layout(coloraxis={'colorbar': dict(title="# of training steps", len=0.9)})
    # fig_no_noise.show()
    fig1 = plt.figure(1, figsize=(10, 7.5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_ylabel("Entropy")
    ax1.set_xlabel("\\# of training steps")
    ax1.plot(entropy_diff)
    # fig1.savefig("entropy_diff.pdf")
    """df = pd.DataFrame()
    df['kappa_with_noise'] = kappa_with_noise.tolist()
    df['kappa_no_noise'] = kappa_no_noise.tolist()
    df['ev_with_noise'] = ev_with_noise.tolist()
    df['ev_no_noise'] = ev_no_noise.tolist()
    df['training_steps_color'] = training_steps_color
    # df = pd.concat([df, pd.DataFrame(ev)], axis=1)
    # df.columns = ['kappa', 'kernel_ev', 'ev_diff', 'ev1', 'ev2']
    df.to_csv('model_noise_dependency_55_color.csv')"""
    """while not m:
        kappa, phi = matrix.parametrization(kappa_0, r, steps)
        kappa_new = kappa
        # plot_color = [0 for _ in kappa]
        ev = np.empty((0, 2))
        symmatrix = matrix.matrix_one_close_re(kappa)
        ev_new = matrix.eigenvalues(symmatrix)
        plots.energy_plane_plotly(ev_new, phi)
        ev = dpp.initial_dataset(ev_new)
        plots.energy_plane_plotly(ev, phi)
        # ep = zs.zero_search()
        # ev_1_one_close_re = np.concatenate((ev_new[:28, 3], ev_new[28:30, 2], ev_new[30:, 1]))
        # ev_2_one_close_re = np.concatenate((ev_new[:14, 1], ev_new[14:28, 2], ev_new[28:30, 1], ev_new[30:78, 2],
        #                                    ev_new[78:, 3]))
        # ev_one = ev[::, 1]
        # ev_two = ev[::, 2]
        # ev = np.column_stack([ev_one, ev_two])
        # plots.energy_plane_plotly(ev, phi)
        # ev_one_close_re = np.concatenate((ev, np.column_stack([ev_1_one_close_re, ev_2_one_close_re])))
        # ev_1_two_close_im = ev_new[::, 1]
        # ev_2_two_close_im = ev_new[::, 2]
        # ev_two_close_im = np.concatenate((ev, np.column_stack([ev_1_two_close_im, ev_2_two_close_im])))
        # df = pd.DataFrame()
        # df = pd.concat([df, pd.DataFrame(ev_one_close_re)], axis=1)
        # df.columns = ['ev1', 'ev2']
        # df.to_csv('eigenvalues_5d_one_close_re.csv')"""
    """df = pd.read_csv('data_kappa.csv', header=0, skiprows=0,
                         names=["kappa", "ev1", "ev2"])
        ev = np.concatenate((ev, np.column_stack([np.array(df.ev1).astype(complex),
                                                  np.array(df.ev2).astype(complex)])))
        kappa = np.array(df.kappa).astype(complex)
        model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
        model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
        gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
        ep_search_fun = gpflow_model.get_ep_evs()
        print(kappa[-1])
        ep = zs.zero_search(ep_search_fun, kappa[-1])
        print(ep)"""
    """ev = ev[::p]
        kappa = kappa[::p]
        n = False
        print(np.shape(kappa))
        # phi = phi[::6]
        entropy_diff = []
        entropy_sum = []
        training_steps = []
        training_steps_color = [i for _ in kappa]
        while not n:
            model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
            model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
            xx, yy = np.meshgrid(kappa[-1].real, kappa[-1].imag)
            grid = np.array((xx.ravel(), yy.ravel())).T
            mean_diff_old, var_diff_old = model_diff.predict_f(grid)
            mean_sum_old, var_sum_old = model_sum.predict_f(grid)
            if np.any(abs(kernel_ev_diff) < eps):
                print("DIFF MODEL FULLY TRAINED!")
                n = True
            if np.any(abs(kernel_ev_sum) < eps):
                print("SUM MODEL FULLY TRAINED!")
            # print("GPR old ev difference squared: ", mean_diff_old.numpy().ravel())
            # print("Variance diff old: ", var_diff_old.numpy())
            # print("Variance sum old: ", var_sum_old.numpy())
            # if np.any(kernel_ev_diff < 0) or np.any(kernel_ev_sum < 0) or np.any(type(kernel_ev_diff[::]) ==
            # complex) or np.any(type(kernel_ev_sum[::]) == complex):
            #    break
            # plt.figure(i)
            # entropy_step_diff = np.sum(-kernel_ev_diff * np.log(kernel_ev_diff)) / len(kernel_ev_diff)
            # entropy_step_sum = np.sum(-kernel_ev_sum * np.log(kernel_ev_sum)) / len(kernel_ev_sum)
            # entropy_diff.append(entropy_step_diff)
            # entropy_sum.append(entropy_step_sum)
            # training_steps.append(i)
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
            print("Root: ", kappa_new)
            # print("Ev difference: ", ev[-1, 0]-ev[-1, 1])
            print("Ev difference squared: ", (ev[-1, 0]-ev[-1, 1])**2)
            # print(ev[-1, 0] - ev[-1, 1])
            # print(kernel_ev)
            grid = np.column_stack((kappa_new.real, kappa_new.imag))
            # grid = np.array((xx.ravel(), yy.ravel())).T
            mean_diff, var_diff = model_diff.predict_f(grid)
            mean_sum, var_sum = model_sum.predict_f(grid)
            print("GPR ev difference squared: ", mean_diff.numpy().ravel())
            if i == 50:
                print("Could not find EP yet:")
                print("j: ", j)
                print(kappa_new)
                print("Variance diff: ", var_diff.numpy())
                print("Variance sum: ", var_sum.numpy())
                print("Ev1: ", ev[-1, 0])
                print("Ev2: ", ev[-1, 1])
                print("Ev difference: ", ev[-1, 0]-ev[-1, 1])
                print("GPR ev difference squared: ", mean_diff.numpy().ravel())
                print("GPR ev difference: ", np.sqrt(mean_diff.numpy().ravel()))
                # print(var)
                # print(var_re)
                # print(var_im)
                n = True
        if abs((ev[-1, 0] - ev[-1, 1]).real) < eps_diff and abs((ev[-1, 0] - ev[-1, 1]).imag) < eps_diff:
            print("Found EP:")
            print("j: ", j)
            print(kappa_new)
            print("Variance diff: ", var_diff.numpy())
            print("Variance sum: ", var_sum.numpy())
            print("Ev1: ", ev[-1, 0])
            print("Ev2: ", ev[-1, 1])
            print((ev[-1, 0] - ev[-1, 1]).real)
            print("Ev difference: ", ev[-1, 0] - ev[-1, 1])
            print("GPR ev difference squared: ", mean_diff.numpy().ravel())
            m = True
        j += 1
        kappa_0 = complex(round(kappa_new[0].real, 6), round(kappa_new[0].imag, 6))
        r = r * 10**(-1)
        p += 2
        if j == 10:
            print("Could not find EP yet:")
            print("j: ", j)
            print(kappa_new)
            print("Variance diff: ", var_diff.numpy())
            print("Variance sum: ", var_sum.numpy())
            print("Ev1: ", ev[-1, 0])
            print("Ev2: ", ev[-1, 1])
            print("Ev difference: ", ev[-1, 0] - ev[-1, 1])
            print("GPR ev difference squared: ", mean_diff.numpy().ravel())
            print("GPR ev difference: ", np.sqrt(mean_diff.numpy().ravel()))
            m = True
    # end1 = time.time()
    # print("Time for while loop without jit: ", end1 - start1)
    # fig = px.scatter(x=kappa.real, y=kappa.imag, color=training_steps_color,
    #                 labels=dict(x="Re(\\kappa)", y="Im(\\kappa)", color="# of training steps"))
    # fig.show()"""
    """print(np.shape(kappa))
    ev_diff = ev[::, 0] - ev[::, 1]
    ev_diff_squared = ((ev[::, 0]-ev[::, 1])**2)
    ev_sum = (0.5 * (ev[::, 0] + ev[::, 1]))
    model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
    model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    grid = np.column_stack((kappa.real, kappa.imag))
    # print(xx)
    # grid = np.array(xx.ravel()).T
    # print(grid)
    mean_diff_old, var_diff_old = model_diff.predict_f(grid)
    mean_sum_old, var_sum_old = model_sum.predict_f(grid)
    # print(mean_diff_old.numpy())
    # print(np.shape(mean_diff_old.numpy()))
    # print(var_diff_old.numpy())
    # print(np.shape(var_diff_old.numpy()))
    df = pd.DataFrame()
    df['kappa'] = kappa.tolist()
    # df['kernel_ev'] = kernel_ev.tolist()
    # df['ev_diff'] = ev_diff.tolist()
    df = pd.concat([df, pd.DataFrame(ev)], axis=1)
    df['ev_diff'] = ev_diff.tolist()
    df['ev_diff_squared'] = ev_diff_squared.tolist()
    df = pd.concat([df, pd.DataFrame(mean_diff_old.numpy())], axis=1)
    df = pd.concat([df, pd.DataFrame(var_diff_old.numpy())], axis=1)
    df['ev_sum'] = ev_sum.tolist()
    df = pd.concat([df, pd.DataFrame(mean_sum_old.numpy())], axis=1)
    df = pd.concat([df, pd.DataFrame(var_sum_old.numpy())], axis=1)
    df.columns = ['kappa', 'ev1', 'ev2', 'ev_diff', 'ev_diff_squared', 'model_diff_squared_real',
                  'model_diff_squared_imag', 'model_var_diff_real', 'model_var_diff_imag', 'ev_sum', 'model_sum_real',
                  'model_sum_imag', 'model_var_sum_real', 'model_var_sum_imag']
    df.to_csv('data.csv')
    ev_diff_complex = ((ev[::, 0] - ev[::, 1]) ** 2)
    ev_sum_complex = (0.5 * (ev[::, 0] + ev[::, 1]))"""
    # plots.control_model_3d_plotly(kappa, ev_diff_complex, ev_sum_complex, model_diff, model_sum)
    # df.to_csv('data_kernel_ev_25.csv')
    # plots.entropy_kernel_ev_matplotlib()
    """fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_ylabel("Entropy")
    ax2.set_xlabel("\\# of training steps")
    ax2.plot(entropy_sum)
    ax2.set_title("Entropy calculated with $\\nicefrac{\\lambda_{\\text{k}, i}}{\\text{len}\\left(\\vec{\\lambda}_\\text{k}\\right)}$")
    print(fig2.get_size_inches())
    fig2.savefig("entropy_sum_divided_length.pdf")"""

    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    # plots.parameter_plane_plotly(kappa, phi)
    # plots.energy_plane_plotly(ev, phi)
    # plots.eigenvalues_angle_plotly(ev, phi, m_re, m_im)
    # plots.eigenvalues_angle_matplotlib(ev, phi, m_re, m_im)
    # plt.show()
