import numpy as np
import matrix
import plots
import time
import gaussian_process_regression as gpr
import zerosearch as zs
import matplotlib.pyplot as plt
import GPFlow_model_class as GPFmc
import pandas as pd
from jax import jit


if __name__ == '__main__':
    # plots.init_matplotlib()

    kappa_0 = 0.5 + 0.5j
    r = 1
    steps = 10
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    kappa_new = kappa
    plot_color = [0 for _ in kappa]
    ev = np.empty((0, 2))
    n = False
    i = 0
    eps = 1.e-15
    # symmatrix = matrix.matrix_random(2, kappa_new)
    # end1 = time.time()
    # print("Matrix created in ", end1-start1)
    # start2 = time.time()
    # ev_new = matrix.eigenvalues(symmatrix)
    # ev = np.concatenate((ev, ev_new))
    # m, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    # m_re, m_im, kernel_ev = gpr.gp_diff_kappa(ev, kappa)
    # x_data = [0.5, 0.5]
    # xx, yy = np.meshgrid(x_data[0], x_data[1])
    # grid = np.array((xx.ravel(), yy.ravel())).T
    # mean, var = m.predict_f(grid)
    # print(mean.numpy()[0][0])
    # gpflow_model = GPFmc.GPFlowModelTest(m)
    # gpflow_function = gpflow_model.get_model_generator()
    # sol = zs.zero_search(gpflow_function, kappa_0)
    # print(sol.x)
    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    """plt.figure(1)
    plt.xlabel("Re(\\kappa)")
    plt.ylabel("Im(\\kappa)")
    plt.scatter(x=kappa[:].real, y=kappa[:].imag, c=plot_color, cmap="plasma")
    # plt.savefig("parameterspace_step_%1d" % i)
    m_re, m_im, kernel_ev = gpr.gp_diff_kappa_test(ev, kappa)
    gpflow_model = GPFmc.GPFlowModel(m_re, m_im)
    gpflow_function = gpflow_model.get_model_generator()
    sol = zs.zero_search(gpflow_function, kappa_0)
    plt.figure(2)
    plt.ylabel("Kernel eigenvalues")
    plt.xlabel("# of kappa points")
    plt.plot(kernel_ev)
    plt.savefig("kernel_ev_step_%1d" % i)
    # kappa_new = np.array([complex(sol.x[0], sol.x[1])])
    # kappa = np.concatenate((kappa, kappa_new))
    df = pd.DataFrame()
    df['kappa'] = kappa.tolist()
    df['kernel_ev'] = kernel_ev.tolist()
    df = pd.concat([df, pd.DataFrame(ev)], axis=1)
    df.columns = ['kappa', 'kernel_ev', 'ev1', 'ev2']
    # df.to_csv('data_step_%1d.csv' % i)
    a = "data_step_%1d" % i
    # print(kappa)
    # print(plot_color)"""
    start1 = time.time()
    while not n:
        # start1 = time.time()
        symmatrix = matrix.matrix_random(2, kappa_new)
        # end1 = time.time()
        # print("Matrix created in ", end1-start1)
        # start2 = time.time()
        ev_new = matrix.eigenvalues(symmatrix)
        ev = np.concatenate((ev, ev_new))
        # print(np.shape(ev_new))
        # print(np.shape(ev))
        # end2 = time.time()
        # print("EV calculated in ", end2-start2)
        # ev_training = ev[0::11]
        # phi_training = phi[0::11]
        # kappa_training = kappa[0::11]
        # start3 = time.time()
        ev_diff = (ev[::, 0] - ev[::, 1]) ** 2
        # m_re, m_im, kernel_ev = gpr.gp_diff_kappa(ev, kappa)
        model, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
        plt.figure(i)
        entropy = kernel_ev * np.log(kernel_ev)
        plt.ylabel("Entropy")
        plt.xlabel("# of kappa points")
        plt.plot(entropy)
        # plt.subplot(1, 2, 1)
        # plt.xlabel("Re(\\kappa)")
        # plt.ylabel("Im(\\kappa)")
        # plt.scatter(x=kappa[:].real, y=kappa[:].imag, c=plot_color, cmap="plasma")
        # m_re, m_im, kernel_ev = gpr.gp_diff_kappa_test(ev, kappa)
        # plt.subplot(1, 2, 2)
        # plt.ylabel("Kernel eigenvalues")
        # plt.xlabel("# of kappa points")
        # plt.yscale("log")
        # plt.plot(kernel_ev)
        plt.savefig("entropy_step_%1d" % i)
        # df = pd.DataFrame()
        # df['kappa'] = kappa.tolist()
        # df['kernel_ev'] = kernel_ev.tolist()
        # print(kernel_ev)
        # print(ev_diff)
        # df['ev_diff'] = ev_diff.tolist()
        # df = pd.concat([df, pd.DataFrame(ev)], axis=1)
        # df.columns = ['kappa', 'kernel_ev', 'ev_diff', 'ev1', 'ev2']
        # df.to_csv('data_step_%1d.csv' % i)
        # end3 = time.time()
        # print("Model optimized in ", end3-start3)
        # gpflow_model = GPFmc.GPFlowModel(m_re, m_im)
        gpflow_model = GPFmc.GPFlowModelTest(model)
        gpflow_function = gpflow_model.get_model_generator()
        sol = zs.zero_search(gpflow_function, kappa_0)
        kappa_new = np.array([complex(sol.x[0], sol.x[1])])
        # root = kappa_new
        kappa = np.concatenate((kappa, kappa_new))
        # xx, yy = np.meshgrid(sol.x[0], sol.x[1])
        # grid = np.array((xx.ravel(), yy.ravel())).T
        # _, var = model.predict_f(grid)
        # _, var_re = m_re.predict_f(grid)
        # _, var_im = m_im.predict_f(grid)
        i += 1
        plot_color.append(i)
        print("Root: ", sol.x)
        # print(ev[-1, 0] - ev[-1, 1])
        # print(kernel_ev)
        if np.any(np.array(kernel_ev) < eps):
            print("Found EP:")
            print(sol.x)
            # v, w = np.linalg.eig(matrix.matrix_random(2, kappa_new))
            # print(v, w)
            # print(v[0, 0], w[:, 0])
            # print(v[0, 1], w[:, 1])
            # print(v[0, 0]-v[0, 1])
            # print(var)
            # print(var_re)
            # print(var_im)
            n = True
        if i == 50:
            print("Could not find EP:")
            print(sol.x)
            # print(var)
            # print(var_re)
            # print(var_im)
            n = True
    end1 = time.time()
    print("Time for while loop without jit: ", end1 - start1)
    """n = False
    start2 = time.time()
    while not n:
        # start1 = time.time()
        fast_matrix = jit(matrix.matrix_random)
        symmatrix = fast_matrix(2, kappa_new)
        # end1 = time.time()
        # print("Matrix created in ", end1-start1)
        # start2 = time.time()
        fast_ev = jit(matrix.eigenvalues)
        ev_new = fast_ev(symmatrix)
        ev = np.concatenate((ev, ev_new))
        # print(np.shape(ev_new))
        # print(np.shape(ev))
        # end2 = time.time()
        # print("EV calculated in ", end2-start2)
        # ev_training = ev[0::11]
        # phi_training = phi[0::11]
        # kappa_training = kappa[0::11]
        # start3 = time.time()
        ev_diff = (ev[::, 0] - ev[::, 1]) ** 2
        # m_re, m_im, kernel_ev = gpr.gp_diff_kappa(ev, kappa)
        fast_model = jit(gpr.gp_2d_diff_kappa)
        model, kernel_ev = fast_model(ev, kappa)
        plt.figure(i)
        entropy = kernel_ev * np.log(kernel_ev)
        plt.ylabel("Entropy")
        plt.xlabel("# of kappa points")
        plt.plot(entropy)
        # plt.subplot(1, 2, 1)
        # plt.xlabel("Re(\\kappa)")
        # plt.ylabel("Im(\\kappa)")
        # plt.scatter(x=kappa[:].real, y=kappa[:].imag, c=plot_color, cmap="plasma")
        # m_re, m_im, kernel_ev = gpr.gp_diff_kappa_test(ev, kappa)
        # plt.subplot(1, 2, 2)
        # plt.ylabel("Kernel eigenvalues")
        # plt.xlabel("# of kappa points")
        # plt.yscale("log")
        # plt.plot(kernel_ev)
        # plt.savefig("entropy_step_%1d" % i)
        # df = pd.DataFrame()
        # df['kappa'] = kappa.tolist()
        # df['kernel_ev'] = kernel_ev.tolist()
        # print(kernel_ev)
        # print(ev_diff)
        # df['ev_diff'] = ev_diff.tolist()
        # df = pd.concat([df, pd.DataFrame(ev)], axis=1)
        # df.columns = ['kappa', 'kernel_ev', 'ev_diff', 'ev1', 'ev2']
        # df.to_csv('data_step_%1d.csv' % i)
        # end3 = time.time()
        # print("Model optimized in ", end3-start3)
        # gpflow_model = GPFmc.GPFlowModel(m_re, m_im)
        gpflow_model = GPFmc.GPFlowModelTest(model)
        gpflow_function = gpflow_model.get_model_generator()
        fast_zs = jit(zs.zero_search)
        sol = fast_zs(gpflow_function, kappa_0)
        kappa_new = np.array([complex(sol.x[0], sol.x[1])])
        root = kappa_new
        kappa = np.concatenate((kappa, kappa_new))
        xx, yy = np.meshgrid(sol.x[0], sol.x[1])
        grid = np.array((xx.ravel(), yy.ravel())).T
        # _, var = model.predict_f(grid)
        # _, var_re = m_re.predict_f(grid)
        # _, var_im = m_im.predict_f(grid)
        i += 1
        plot_color.append(i)
        print("Root: ", sol.x)
        # print(ev[-1, 0] - ev[-1, 1])
        # print(kernel_ev)
        if np.any(np.array(kernel_ev) < eps):
            print("Found EP:")
            print(sol.x)
            # print(var)
            # print(var_re)
            # print(var_im)
            n = True
        if i == 50:
            print("Could not find EP:")
            print(sol.x)
            # print(var)
            # print(var_re)
            # print(var_im)
            n = True
    end2 = time.time()
    print("Time for while loop with jit: ", end2 - start2)"""

    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    # plots.parameter_plane_plotly(kappa, phi)
    # plots.energy_plane_plotly(ev, phi)
    # plots.eigenvalues_angle_plotly(ev, phi, m_re, m_im)
    # plots.eigenvalues_angle_matplotlib(ev, phi, m_re, m_im)
    # plt.show()
