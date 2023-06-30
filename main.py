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
import configurematplotlib.plot as confmp
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch


def ep_2d_exchange():
    kappa_0 = 0. + 1.j
    r = 0.1
    steps = 50
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    kappa_new = kappa
    ev = np.empty((0, 2))
    symmatrix = matrix.matrix_random(2, kappa_new)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = np.concatenate((ev, ev_new))
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev)[1])]).ravel())
    ep = 0. + 1.j
    fig, axes = confmp.newfig(nrows=1, aspect=1., ncols=2, left=46, bottom=34, wspace=54,
                              right=51)  #, gridspec={'width_ratios': [1, 1]}), aspect=1.,
    print(axes.shape)
    axes[0].set_xlabel("Re($\\kappa$)")
    axes[0].set_ylabel("Im($\\kappa$)")
    cax1 = axes[0].scatter(x=kappa[:].real, y=kappa[:].imag, marker='o', s=8, c=phi, cmap="plasma")
    axes[0].scatter(x=ep.real, y=ep.imag, marker='x', s=8, c="tab:green", label="EP")
    axes[0].set_yticks([0.9, 0.95, 1., 1.05, 1.1])
    # confmp.legend(axes[0], loc=1)
    axes[0].legend(loc=1)
    # axes[0].colorbar("Angle / \\si{\\radian}")
    axes[1].set_xlabel("Re($\\lambda$)")
    axes[1].set_ylabel("Im($\\lambda$)")
    axes[1].set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    cax2 = axes[1].scatter(x=ev.ravel().real, y=ev.ravel().imag, marker='o', s=8, c=phi_all, cmap="plasma")
    # axes[1].colorbar(label="Angle / \\si{\\radian}")
    # cb = fig.colorbar(cax, ax=axes[1], label="Angle / \\si{\\radian}", ticks=[0, np.pi/2, np.pi, 3*np.pi/2])
    cbar, cax = confmp.cbar_beside(fig, axes, cax2, dx=0.01)
    cbar.set_label("$\\phi$")
    cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    confmp.subfig_label(axes[0], 0, 'left', 0, dx=-47, va='top', y=1, dy=0)
    confmp.subfig_label(axes[1], 1, 'right', 0, dx=-32, va='top', y=1, dy=0)
    plt.savefig("../../mastersthesis/plots/plots/EPexample2d_parameter_energyTEST6.pdf")


def gpr_model_output_colormap():
    kappa_0 = 0. + 1.j
    r = 0.1
    steps = 12
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    kappa_new = kappa
    ev = np.empty((0, 2))
    symmatrix = matrix.matrix_random(2, kappa_new)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = np.concatenate((ev, ev_new))
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.1, 0.1, 100)
    y = np.linspace(0.9, 1.1, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(nrows=2, ncols=2, left=44, top=17, bottom=34, right=50,
                              wspace=5, hspace=22, sharex=True, sharey=True)
    axes[0, 0].set_title("Re($p$)")
    for i in range(np.shape(axes)[0]):
        axes[i, 0].set_ylabel("Im($\kappa$)")
        for j in range(np.shape(axes)[1]):
            if i == 0:
                im_diff = confmp.imshow(axes[0, j], mean_diff[:, j].numpy().reshape(-100, 100), extent)
            if i == 1:
                im_sum = confmp.imshow(axes[1, j], mean_sum[:, j].numpy().reshape(-100, 100), extent)
            axes[1, j].set_xlabel("Re($\kappa$)")
            axes[i, j].set_xticks([-0.05, 0, 0.05])
            axes[i, j].set_yticks([0.95, 1, 1.05])
    axes[0, 1].set_title("Im($p$)")
    axes[1, 0].set_title("Re($s$)")
    axes[1, 1].set_title("Im($s$)")
    confmp.cbar_beside(fig, axes[0, :], im_diff, dx=0.01)
    confmp.cbar_beside(fig, axes[1, :], im_sum, dx=0.01)
    plt.savefig("../../mastersthesis/plots/plots/GPRPlane.pdf")


def gpr_model_output_p_presentation():
    kappa_0 = 0. + 1.j
    r = 0.1
    steps = 12
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    kappa_new = kappa
    ev = np.empty((0, 2))
    symmatrix = matrix.matrix_random(2, kappa_new)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = np.concatenate((ev, ev_new))
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.1, 0.1, 100)
    y = np.linspace(0.9, 1.1, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(width=0.45, aspect=1., nrows=2, ncols=1, left=44, top=17, bottom=34, right=47,
                              hspace=22, sharex=True)
    axes[0].set_title("Re($p$)")
    axes[1].set_xlabel("Re($\kappa$)")
    for i in range(np.shape(axes)[0]):
        axes[i].set_ylabel("Im($\kappa$)")
        im_diff = confmp.imshow(axes[i], mean_diff[:, i].numpy().reshape(-100, 100), extent, cmap='seismic')
        axes[i].set_xticks([-0.05, 0, 0.05])
        axes[i].set_yticks([0.95, 1, 1.05])
    axes[1].set_title("Im($p$)")
    confmp.cbar_beside(fig, axes, im_diff)
    plt.savefig("../../mastersthesis/plots/plots/GPRPlaneOnlyP.pdf")


def gpr_model_output_diff_colormap():
    ep = 0. + 1.j
    kappa_0 = 0. + 1.j
    r = 0.1
    steps = 12
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    kappa_new = kappa
    ev = np.empty((0, 2))
    symmatrix = matrix.matrix_random(2, kappa_new)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = np.concatenate((ev, ev_new))
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.1, 0.1, 100)
    y = np.linspace(0.9, 1.1, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(aspect=1., nrows=1, ncols=2, left=45, top=16, bottom=34, right=46,
                              wspace=5, sharey=True)  # left=46, bottom=34, wspace=54, right=51
    axes[0].set_title("Re($p$)", size=12)
    axes[0].set_ylabel("Im($\kappa$)")
    for i in range(np.shape(axes)[0]):
        axes[i].set_xlabel("Re($\kappa$)")
        im_diff = confmp.imshow(axes[i], mean_diff[:, i].numpy().reshape(-100, 100), extent, cmap='seismic')
        axes[i].scatter(x=ep.real, y=ep.imag, marker='x', s=8, c="tab:green", label="EP")
        axes[i].set_xticks([-0.05, 0, 0.05])
        axes[i].set_yticks([0.95, 1, 1.05])
    axes[1].set_title("Im($p$)", size=12)
    cbar, cax = confmp.cbar_beside(fig, axes, im_diff, dx=0.01)
    #cbar.set_label("$\\left(\\lambda_1 - \\lambda_2\\right)^2$")
    #cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    #cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    confmp.subfig_label(axes[0], 0, 'left', 0, dx=-46, va='bottom', y=1, dy=3)
    confmp.subfig_label(axes[1], 1, 'left', 0, dx=-2, va='bottom', y=1, dy=3)
    plt.savefig("../../mastersthesis/plots/plots/2DGPRPlaneDiff.pdf")


if __name__ == '__main__':
    # plots.init_matplotlib()
    # ep_2d_exchange()
    # gpr_model_output_p_presentation()
    #gpr_model_output_diff_colormap()

    kappa_0 = 0. + 1.j
    ep = 0. + 1.j
    r = 0.1
    steps = 12
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    kappa_new = kappa
    plot_color = [0 for _ in kappa]
    ev = np.empty((0, 2))
    n = False
    i = 0
    eps = 1.e-15
    symmatrix = matrix.matrix_random(2, kappa_new)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = np.concatenate((ev, ev_new))
    all_kernel_ev = dict()
    ev_diff = ev[:, 0] - ev[:, 1]
    #print(ev_diff)
    distances = np.linalg.norm(np.array([ev_diff.real, ev_diff.imag]), axis=0)
    #print(distances)
    mean = np.mean(distances)
    #print(mean)
    distance_ev_all = [mean]

    #ev_difference = np.linalg.norm()
    # plots.parameter_plane_plotly(kappa, phi)
    # symmatrix = matrix.matrix_random(2, kappa_new)
    # end1 = time.time()
    # print("Matrix created in ", end1-start1)
    # start2 = time.time()
    # ev_new = matrix.eigenvalues(symmatrix)
    # ev = np.concatenate((ev, ev_new))
    # plots.energy_plane_plotly(ev, phi)
    # plt.show()
    # plots.thesis_parameter_energy(kappa, ev, phi)
    # m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    # m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    # m_re, m_im, kernel_ev = gpr.gp_diff_kappa(ev, kappa)
    # x_data = [0.5, 0.5]
    # plots.three_d_eigenvalue_kappa_2d_model_plotly(kappa_0, r, m_diff)
    # mean, var = m.predict_f(grid)
    # print(mean.numpy()[0][0])
    # gpflow_model = GPFmc.GPFlowModelTest(m)
    # gpflow_function = gpflow_model.get_model_generator()
    # sol = zs.zero_search(gpflow_function, kappa_0)
    # print(sol.x)
    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    while not n:
        # start1 = time.time()
        # end1 = time.time()
        # print("Matrix created in ", end1-start1)
        # start2 = time.time()
        # m_re, m_im, kernel_ev = gpr.gp_diff_kappa(ev, kappa)
        model, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
        all_kernel_ev[i] = kernel_ev
        gpflow_model = GPFmc.GPFlowModelTest(model)
        gpflow_function = gpflow_model.get_model_generator()
        sol = zs.zero_search(gpflow_function, kappa_0)
        kappa_new = np.array([complex(sol.x[0], sol.x[1])])
        # root = kappa_new
        kappa = np.concatenate((kappa, kappa_new))
        i += 1
        plot_color.append(i)
        print("Root: ", sol.x)
        symmatrix = matrix.matrix_random(2, kappa_new)
        ev_new = matrix.eigenvalues(symmatrix)
        ev = np.concatenate((ev, ev_new))
        ev_diff = ev[-1, 0] - ev[-1, 1]
        print(ev_diff)
        distance_ev = np.linalg.norm(np.array([ev_diff.real, ev_diff.imag]))
        distance_ev_all.append(distance_ev)
        print(kernel_ev)
        if abs(ev_diff.real) < 2.e-8 and abs(ev_diff.imag) < 2.e-8:
            print("Found EP:")
            print(sol.x)
            n = True
        if np.any(np.array(kernel_ev) < eps):
            print("Found EP:")
            print(sol.x)
            diff = np.array([ep.real, ep.imag]) - sol.x
            print(diff)
            norm = np.linalg.norm(diff)
            print("Abweichung: ", norm)
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
    # print("Time for while loop without jit: ", end1 - start1)

    fig, axes = confmp.newfig(aspect=1., nrows=1, ncols=2, gridspec=True, left=45, right=3,
                              top=56, bottom=34, wspace=25, width_ratios=np.array([2, 1]))
    ax1 = fig.add_subplot(axes[0])
    ax2 = fig.add_subplot(axes[1])
    cmap, vmin, vmax, ticks = confmp.discrete_colormap(plot_color)
    ax1.set_ylabel("Im($\kappa$)")
    ax1.set_xlabel("Re($\kappa$)")
    con1 = ConnectionPatch(xyA=(-0.00000000001, 0.99999999), xyB=(0.002, 0.997),
                           coordsA="data", coordsB="data",
                           axesA=ax2, axesB=ax1,
                           color="black", lw=0.5)
    con2 = ConnectionPatch(xyA=(-0.00000000001, 1.000000175), xyB=(0.002, 1.003),
                           coordsA="data", coordsB="data",
                           axesA=ax2, axesB=ax1,
                           color="black", lw=0.5)
    fig.add_artist(con1)
    fig.add_artist(con2)
    ax1.scatter(kappa.real, kappa.imag, c=plot_color, cmap=cmap, marker="8", s=8, vmin=vmin, vmax=vmax)
    ax2.set_ylim([0.99999999, 1.000000175])
    ax2.set_xlim([-0.00000000001, 0.00000000001])
    ax2.set_xticks([-0.5e-11, 0, 0.5e-11])
    ax2.set_title("$\\times 10^{-7} + 1$", loc="left", fontsize=12, pad=2)
    ax2.set_xlabel("$\\times 10^{-12}$", loc="right", fontsize=12, labelpad=1)
    ax2.set_yticks([1.0000000000000000, 1. + 0.5e-7, 1. + 1.e-7, 1. + 1.5e-7],
                   labels=[0, 0.5, 1, 1.5])
    ax2.set_xticks([-5.e-12, 0.0, 5.e-12], labels=[-5,0,5])
    # matplotlib.ticker.ScalarFormatter(useOffset=1.00000000000000)
    # ax2.ticklabel_format(useOffset=1.0)
    # ax2.scatter(ep.real, ep.imag, marker='x', s=16, c="tab:green", label="EP")
    im = ax2.scatter(kappa.real, kappa.imag, c=plot_color, cmap=cmap, marker="8", s=8, vmin=vmin, vmax=vmax)
    ax1.add_patch(Rectangle((-0.002, 0.997), 0.004, 0.006,
                            edgecolor='black',
                            fill=False,
                            lw=0.5))
    # cax = fig.add_axes((ax1.get_position().xmin, ax1.get_position().ymax + 0.1,
    #                    ax2.get_position().xmax - ax1.get_position().xmin, (10/72) / fig.get_figheight()))
    # cbar = fig.colorbar()
    cbar, cax = confmp.cbar_above(fig, (ax1, ax2), im, dy=0.055)
    cbar.set_label("\# of training steps")
    cbar.ax.set_xticks([0, 1, 2, 3])
    #fig.savefig("../../mastersthesis/plots/plots/2dTrainingTEST2.pdf")


    plot_color_ev = np.sort([plot_color.copy() for _ in range(2)])
    fig_ev, ax_ev = confmp.newfig(nrows=1, ncols=1, left=45, right=35, bottom=17)
    ax_ev.set_ylabel("Im($\kappa$)")
    ax_ev.set_xlabel("Re($\kappa$)")
    im_ev = ax_ev.scatter(ev.real, ev.imag, c=plot_color_ev, cmap=cmap, marker="o", s=8, vmin=vmin, vmax=vmax)
    cbar_ev, cax_ev = confmp.cbar_beside(fig_ev, ax_ev, im_ev, dx=0.01)
    cbar_ev.set_label("\# of training steps")
    cbar_ev.ax.set_yticks([0, 1, 2, 3])
    #fig_ev.savefig("../../mastersthesis/plots/plots/2dTrainingEigenvaluedifference.pdf")

    kernel_ev0 = all_kernel_ev[0]
    plot_x0 = np.array(list(range(len(kernel_ev0)))) + 1
    kernel_ev1 = all_kernel_ev[1]
    plot_x1 = np.array(list(range(len(kernel_ev1)))) + 1
    kernel_ev2 = all_kernel_ev[2]
    plot_x2 = np.array(list(range(len(kernel_ev2)))) + 1
    kernel_ev_all = [kernel_ev0, kernel_ev1, kernel_ev2]
    plot_x_all = [plot_x0, plot_x1, plot_x2]
    """fig_kernel_ev, ax_kernel_ev = confmp.newfig(aspect=1., nrows=2, ncols=2, left=52, bottom=33, wspace=55, hspace=37) # right=35,
    ax_kernel_ev = ax_kernel_ev.ravel()
    for i in range(len(kernel_ev_all)):
        ax_kernel_ev[i].set_xlabel("$X$")
        ax_kernel_ev[i].set_ylabel("$\\lambda_{\\vb{K}}")
        ax_kernel_ev[i].plot(plot_x_all[i], abs(kernel_ev_all[i]))
        ax_kernel_ev[i].set_yscale('log')
    ax_kernel_ev[3].set_ylabel("$\\Delta\\lambda = \\lambda_1 - \\lambda_2$")
    ax_kernel_ev[3].set_xlabel("\\# of training steps")
    ax_kernel_ev[3].plot([1,2,3], distance_ev_all)
    ax_kernel_ev[3].set_yscale('log')
    for i in range(len(ax_kernel_ev)):
        confmp.subfig_label(ax_kernel_ev[i], i, 'left', x=0, dx=-53, va='top', y=1, dy=0)"""
    # fig_kernel_ev.savefig("../../mastersthesis/plots/plots/2dConvergenceKernelEvDiff.pdf")

    fig_convergence, ax_convergence = confmp.newfig(aspect=1.3, nrows=1, ncols=2, left=52, bottom=33, wspace=55)
    ax_convergence[0].set_xlabel("\\# of training steps")
    ax_convergence[0].set_ylabel("$\\lambda_{\\vb{K}}")
    ax_convergence[0].set_xlim([0.2, 3.8])
    ax_convergence[0].semilogy([1 for _ in kernel_ev0], abs(kernel_ev0), "_", ms=20, c=cmap(1))
    ax_convergence[0].semilogy([2 for _ in kernel_ev1], abs(kernel_ev1), "_", ms=20, c=cmap(2))
    ax_convergence[0].semilogy([3 for _ in kernel_ev2], abs(kernel_ev2), "_", ms=20, c=cmap(3))
    ax_convergence[1].set_ylabel("$\\Delta\\lambda = \\abs{\\lambda_1 - \\lambda_2}$")
    ax_convergence[1].set_xlabel("\\# of training steps")
    ax_convergence[1].semilogy([0,1,2,3], distance_ev_all)
    for i in range(len(ax_convergence)):
        confmp.subfig_label(ax_convergence[i], i, 'left', x=0, dx=-53, va='top', y=1, dy=0)
    fig_convergence.savefig("../../mastersthesis/plots/plots/2dConvergenceKernelEvDiffVersion2.pdf")


    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    # plots.parameter_plane_plotly(kappa, plot_color)
    # plots.energy_plane_plotly(ev, phi)
    # plots.eigenvalues_angle_plotly(ev, phi, m_re, m_im)
    # plots.eigenvalues_angle_matplotlib(ev, phi, m_re, m_im)
    # plt.show()
