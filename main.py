import numpy as np
import matrix
import plots
import time
import gaussian_process_regression as gpr
import zerosearch as zs
import matplotlib.pyplot as plt
import GPFlow_model_class as GPFmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jax.numpy as jnp
from jax import vmap
import data_preprocessing as dpp


if __name__ == '__main__':
    # plots.init_matplotlib()

    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    m = False
    j = 0
    i = 0
    p = 16
    eps = 1.e-15
    eps_var = 1.e-6
    eps_diff = 1.e-2
    while not m:
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
        # df.to_csv('eigenvalues_5d_one_close_re.csv')
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
        ev = ev[::p]
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
    # fig.show()
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
    # plots.control_model_2d_plotly(kappa[18:], ev_diff_complex[18:], ev_sum_complex[18:], model_diff, model_sum)
    # plots.control_model_3d_plotly(kappa, ev_diff_complex, ev_sum_complex, model_diff, model_sum)
    """plt.figure(1)
    plt.ylabel("Entropy")
    plt.xlabel("# of training steps")
    plt.plot(entropy_diff)
    plt.savefig("entropy_diff.pdf")
    plt.figure(2)
    plt.ylabel("Entropy")
    plt.xlabel("# of training steps")
    plt.plot(entropy_sum)
    plt.savefig("entropy_sum.pdf")"""

    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    # plots.parameter_plane_plotly(kappa, phi)
    # plots.energy_plane_plotly(ev, phi)
    # plots.eigenvalues_angle_plotly(ev, phi, m_re, m_im)
    # plots.eigenvalues_angle_matplotlib(ev, phi, m_re, m_im)
    # plt.show()
