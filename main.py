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
import plotly.graph_objects as go
import jax.numpy as jnp
from jax import vmap
import data_preprocessing as dpp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import configurematplotlib.plot as confmp
import pickle

ep_one_close = np.array([1.18503351 + 1.00848184j])
ep_two_close = np.array([0.88278093 + 1.09360073j])

def energy_plane_5d_model():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev_new)[1])]).ravel())
    ep = ep_one_close
    fig, axes = confmp.newfig(nrows=1, ncols=1, left=48, right=50, bottom=35)
    im = axes.scatter(ev_new.real, ev_new.imag, c=phi_all, cmap='plasma', marker='o', s=8)
    axes.set_xlabel("Re($\lambda$)")
    axes.set_ylabel("Im($\lambda$)")
    axes.set_yticks([-1, -0.5, 0, 0.5])
    rect = [0.6, 0.12, 0.38, 0.38]
    pos = axes.get_position()
    width = pos.width
    height = pos.height
    inax_position = axes.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height])
    subax.scatter(kappa.real, kappa.imag, marker='o', s=8, c=phi, cmap='plasma')
    subax.scatter(ep.real, ep.imag, marker='x', s=8, c='tab:green', label="EP")
    subax.set_ylabel("Im($\kappa$)", labelpad=1.5, size=12 * rect[3] ** 0.5)
    subax.set_xlabel("Re($\kappa$)", labelpad=1.5, size=12 * rect[2] ** 0.5)
    x_ticklabelsize = subax.get_xticklabels()[0].get_size()
    y_ticklabelsize = subax.get_yticklabels()[0].get_size()
    x_ticklabelsize *= rect[2] ** 0.5
    y_ticklabelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_ticklabelsize)
    subax.yaxis.set_tick_params(labelsize=y_ticklabelsize)
    subax.legend(loc=4)
    cbar_width = 10 / 72
    cax = fig.add_axes((pos.xmax + 0.01, pos.ymin, cbar_width / fig.get_figwidth(), pos.ymax - pos.ymin))
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("Angle / \\si{\\radian}")
    cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    fig.savefig("../../mastersthesis/plots/plots/EPexample5d_energy.pdf")


def energy_plane_5d_model2():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev_new)[1])]).ravel())
    ep = ep_two_close
    fig, axes = confmp.newfig(nrows=1, ncols=1, left=32, right=44, bottom=31)
    im = axes.scatter(ev_new.real, ev_new.imag, c=phi_all, cmap='plasma', marker='o', s=8)
    axes.set_xlabel("Re($\lambda$)")
    axes.set_ylabel("Im($\lambda$)")
    # axes.set_yticks([-1, -0.5, 0, 0.5])
    rect = [0.13, 0.53, 0.55, 0.45]
    pos = axes.get_position()
    width = pos.width
    height = pos.height
    inax_position = axes.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height])
    subax.scatter(kappa.real, kappa.imag, marker='o', s=8, c=phi, cmap='plasma')
    subax.scatter(ep.real, ep.imag, marker='x', s=6, lw=0.5, c='tab:green', label="EP")
    subax.set_ylabel("Im($\kappa$)", labelpad=1.5, size=(10) * rect[3] ** 0.5)
    subax.set_xlabel("Re($\kappa$)", labelpad=1.5, size=(10) * rect[2] ** 0.5)
    x_ticklabelsize = subax.get_xticklabels()[0].get_size()
    y_ticklabelsize = subax.get_yticklabels()[0].get_size()
    x_ticklabelsize *= rect[2] ** 0.5
    y_ticklabelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_ticklabelsize)
    subax.yaxis.set_tick_params(labelsize=y_ticklabelsize)
    subax.legend(loc=4)
    cbar_width = 10 / 72
    cax = fig.add_axes((pos.xmax + 0.01, pos.ymin, cbar_width / fig.get_figwidth(), pos.ymax - pos.ymin))
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("$\\phi$")
    cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    fig.savefig("../../../ITP1/paper/ep_paper/iop_style/plots/plots/EPexample5d_energy_model2_1.pdf")


def energy_plane_5d_model2_presentation():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev_new)[1])]).ravel())
    ep = ep_two_close
    fig, axes = confmp.newfig(aspect=1.9, nrows=1, ncols=1, left=40, right=45, bottom=29)
    im = axes.scatter(ev_new.real, ev_new.imag, c=phi_all, cmap='plasma', marker='o', s=2)
    axes.set_xlabel("Re$(\lambda)$")
    axes.set_ylabel("Im$(\lambda)$")
    # axes.set_yticks([-1, -0.5, 0, 0.5])
    rect = [0.12, 0.48, 0.55, 0.5]
    pos = axes.get_position()
    width = pos.width
    height = pos.height
    inax_position = axes.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height])
    subax.scatter(kappa.real, kappa.imag, marker='o', s=2, c=phi, cmap='plasma')
    subax.scatter(ep.real, ep.imag, marker='x', s=6, lw=1, c='tab:green', label="EP")
    subax.set_ylabel("Im($\kappa$)", labelpad=1.5, size=(34.8/3) * rect[3] ** 0.5)
    subax.set_xlabel("Re($\kappa$)", labelpad=1.5, size=(34.8/3) * rect[2] ** 0.5)
    x_ticklabelsize = subax.get_xticklabels()[0].get_size()
    y_ticklabelsize = subax.get_yticklabels()[0].get_size()
    x_ticklabelsize *= rect[2] ** 0.5
    y_ticklabelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_ticklabelsize)
    subax.yaxis.set_tick_params(labelsize=y_ticklabelsize)
    subax.legend(loc=4)
    cbar_width = 10 / 72
    cax = fig.add_axes((pos.xmax + 0.01, pos.ymin, cbar_width / fig.get_figwidth(), pos.ymax - pos.ymin))
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("$\\phi$")
    cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    fig.savefig("../../talk/fig/EPexample5d_energy_model2.pdf")


def energy_plane_both_models():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix0 = matrix.matrix_one_close_re(kappa)
    ev_new0 = matrix.eigenvalues(symmatrix0)
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev_new0)[1])]).ravel())
    ep = ep_one_close
    fig, axes = confmp.newfig(aspect=1.7, nrows=2, ncols=1, left=48, right=49, bottom=35, hspace=40)
    im = axes[0].scatter(ev_new0.real, ev_new0.imag, c=phi_all, cmap='plasma', marker='o', s=2)
    axes[0].set_xlabel("Re($\lambda$)")
    axes[0].set_ylabel("Im($\lambda$)")
    axes[0].set_yticks([-1, -0.5, 0, 0.5])
    rect0 = [0.6, 0.12, 0.38, 0.38]
    pos0 = axes[0].get_position()
    width0 = pos0.width
    height0 = pos0.height
    inax_position0 = axes[0].transAxes.transform(rect0[0:2])
    transFigure0 = fig.transFigure.inverted()
    infig_position0 = transFigure0.transform(inax_position0)
    x0 = infig_position0[0]
    y0 = infig_position0[1]
    width0 *= rect0[2]
    height0 *= rect0[3]
    subax0 = fig.add_axes([x0, y0, width0, height0])
    subax0.scatter(kappa.real, kappa.imag, marker='o', s=2, c=phi, cmap='plasma')
    subax0.scatter(ep.real, ep.imag, marker='x', s=6, lw=1, c='tab:green', label="EP")
    subax0.set_ylabel("Im($\kappa$)", labelpad=1.5, size=12 * rect0[3] ** 0.5)
    subax0.set_xlabel("Re($\kappa$)", labelpad=1.5, size=12 * rect0[2] ** 0.5)
    x_ticklabelsize0 = subax0.get_xticklabels()[0].get_size()
    y_ticklabelsize0 = subax0.get_yticklabels()[0].get_size()
    x_ticklabelsize0 *= rect0[2] ** 0.5
    y_ticklabelsize0 *= rect0[3] ** 0.5
    subax0.xaxis.set_tick_params(labelsize=x_ticklabelsize0)
    subax0.yaxis.set_tick_params(labelsize=y_ticklabelsize0)
    subax0.legend(loc=4)

    symmatrix1 = matrix.matrix_two_close_im(kappa)
    ev_new1 = matrix.eigenvalues(symmatrix1)
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev_new1)[1])]).ravel())
    ep = ep_two_close
    im = axes[1].scatter(ev_new1.real, ev_new1.imag, c=phi_all, cmap='plasma', marker='o', s=2)
    axes[1].set_xlabel("Re($\lambda$)")
    axes[1].set_ylabel("Im($\lambda$)")
    # axes.set_yticks([-1, -0.5, 0, 0.5])
    rect1 = [0.12, 0.48, 0.55, 0.5]
    pos1 = axes[1].get_position()
    width1 = pos1.width
    height1 = pos1.height
    inax_position1 = axes[1].transAxes.transform(rect1[0:2])
    transFigure1 = fig.transFigure.inverted()
    infig_position1 = transFigure1.transform(inax_position1)
    x1 = infig_position1[0]
    y1 = infig_position1[1]
    width1 *= rect1[2]
    height1 *= rect1[3]
    subax1 = fig.add_axes([x1, y1, width1, height1])
    subax1.scatter(kappa.real, kappa.imag, marker='o', s=2, c=phi, cmap='plasma')
    subax1.scatter(ep.real, ep.imag, marker='x', s=6, lw=1, c='tab:green', label="EP")
    subax1.set_ylabel("Im($\kappa$)", labelpad=1.5, size=12 * rect1[3] ** 0.5)
    subax1.set_xlabel("Re($\kappa$)", labelpad=1.5, size=12 * rect1[2] ** 0.5)
    x_ticklabelsize1 = subax1.get_xticklabels()[0].get_size()
    y_ticklabelsize1 = subax1.get_yticklabels()[0].get_size()
    x_ticklabelsize1 *= rect1[2] ** 0.5
    y_ticklabelsize1 *= rect1[3] ** 0.5
    subax1.xaxis.set_tick_params(labelsize=x_ticklabelsize1)
    subax1.yaxis.set_tick_params(labelsize=y_ticklabelsize1)
    subax1.legend(loc=4)

    #cbar_width = 10 / 72
    #cax = fig.add_axes((pos1.xmax + 0.01, pos1.ymin, cbar_width / fig.get_figwidth(), pos1.ymax - pos1.ymin))
    #cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar, cax = confmp.cbar_beside(fig, axes, im, dx=0.01)
    cbar.set_label("$\\phi$")
    cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    confmp.subfig_label(axes[0], 0, 'left', 0, dx=-49, va='top', y=1, dy=0)
    confmp.subfig_label(axes[1], 1, 'left', 0, dx=-49, va='top', y=1, dy=0)
    fig.savefig("../../mastersthesis/plots/plots/EPexample5d_energy_both.pdf")


def gpr_plane_trained():
    ep = ep_two_close  # np.array([0.88278093 + 1.09360073j])
    df_extra2 = pd.read_csv('../../mastersthesis/plots/data/paper_data_kappa_trained_model_extra2.csv',
                            header=0, skiprows=0,
                            names=["kappa", "ev1", "ev2", "training_steps_color"])
    ev_extra2 = np.column_stack([np.array(df_extra2.ev1).astype(complex),
                                 np.array(df_extra2.ev2).astype(complex)])
    kappa_extra2 = np.array(df_extra2.kappa).astype(complex)
    training_steps_color_extra2 = np.array(df_extra2.training_steps_color)

    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev_extra2[:-1], kappa_extra2[:-1])
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev_extra2[:-1], kappa_extra2[:-1])
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(nrows=2, ncols=2, left=39, top=17, bottom=34, right=46,
                              wspace=5, hspace=25, sharex=True, sharey=True)
    axes[0, 0].set_title("Re($p$)")
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    vmin_sum = np.min(mean_sum.numpy())
    vmax_sum = np.max(mean_sum.numpy())
    for i in range(np.shape(axes)[0]):
        axes[i, 0].set_ylabel("Im($\kappa$)")
        for j in range(np.shape(axes)[1]):
            if i == 0:
                im_diff = confmp.imshow(axes[0, j], mean_diff[:, j].numpy().reshape(-100, 100), extent,
                                        cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
                axes[0, j].scatter(ep.real, ep.imag, marker='x', s=8, c='tab:green')
            if i == 1:
                im_sum = confmp.imshow(axes[1, j], mean_sum[:, j].numpy().reshape(-100, 100), extent,
                                       vmin=vmin_sum, vmax=vmax_sum)
            axes[1, j].set_xlabel("Re($\kappa$)")
            axes[i, j].set_xticks([0., 0.5, 1.])
            axes[i, j].set_yticks([0., 0.5, 1.])
    axes[0, 1].set_title("Im($p$)")
    axes[1, 0].set_title("Re($s$)")
    axes[1, 1].set_title("Im($s$)")
    confmp.cbar_beside(fig, axes[0, :], im_diff, dx=0.01)
    confmp.cbar_beside(fig, axes[1, :], im_sum, dx=0.01)
    plt.savefig("../../mastersthesis/plots/plots/GPRPlane5DModelTrained.pdf")


def gpr_plane_trained_only_p():
    ep = ep_two_close  # np.array([0.88278093 + 1.09360073j])
    df_extra2 = pd.read_csv('../../mastersthesis/plots/data/paper_data_kappa_trained_model_extra2.csv',
                            header=0, skiprows=0,
                            names=["kappa", "ev1", "ev2", "training_steps_color"])
    ev_extra2 = np.column_stack([np.array(df_extra2.ev1).astype(complex),
                                 np.array(df_extra2.ev2).astype(complex)])
    kappa_extra2 = np.array(df_extra2.kappa).astype(complex)
    training_steps_color_extra2 = np.array(df_extra2.training_steps_color)

    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev_extra2[:-1], kappa_extra2[:-1])
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev_extra2[:-1], kappa_extra2[:-1])
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(nrows=1, ncols=2, left=39, top=47, bottom=34,
                              wspace=5, sharey=True)
    axes[0].set_title("Re($p$)")
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    vmin_sum = np.min(mean_sum.numpy())
    vmax_sum = np.max(mean_sum.numpy())
    axes[0].set_ylabel("Im($\kappa$)")
    for i in range(np.shape(axes)[0]):
        im_diff = confmp.imshow(axes[i], mean_diff[:, i].numpy().reshape(-100, 100), extent,
                                cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axes[i].scatter(ep.real, ep.imag, marker='x', s=8, c='tab:green')
        axes[i].set_xlabel("Re($\kappa$)")
        axes[i].set_xticks([0., 0.5, 1.])
        axes[i].set_yticks([0., 0.5, 1.])
    axes[1].set_title("Im($p$)")
    # axes[1, 0].set_title("Re($s$)")
    # axes[1, 1].set_title("Im($s$)")
    confmp.cbar_above(fig, axes[:], im_diff, dy=0.1)
    # confmp.cbar_beside(fig, axes[1, :], im_sum, dx=0.01)
    plt.savefig("../../mastersthesis/plots/plots/GPRPlane5DModelTrainedOnlyP.pdf")


def gpr_plane_trained_only_p_with_kappa_space():
    ep = ep_two_close  # np.array([0.88278093 + 1.09360073j])
    df_extra2 = pd.read_csv('../../mastersthesis/plots/data/paper_data_kappa_trained_model_extra2.csv',
                            header=0, skiprows=0,
                            names=["kappa", "ev1", "ev2", "training_steps_color"])
    ev_extra2 = np.column_stack([np.array(df_extra2.ev1).astype(complex),
                                 np.array(df_extra2.ev2).astype(complex)])
    kappa_extra2 = np.array(df_extra2.kappa).astype(complex)
    training_steps_color_extra2 = np.array(df_extra2.training_steps_color)

    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev_extra2[:-1], kappa_extra2[:-1])
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev_extra2[:-1], kappa_extra2[:-1])
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(nrows=2, ncols=2, left=39, top=17, bottom=34, right=41,
                              wspace=5, hspace=38)
    axes[0, 0].set_title("Re($p$)")
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    vmin_sum = np.min(mean_sum.numpy())
    vmax_sum = np.max(mean_sum.numpy())
    axes[0, 0].set_ylabel("Im($\kappa$)")
    for i in range(np.shape(axes)[0]):
        im_diff = confmp.imshow(axes[0, i], mean_diff[:, i].numpy().reshape(-100, 100), extent,
                                cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axes[0, i].scatter(ep.real, ep.imag, marker='x', s=6, lw=0.5, c='tab:green')
        axes[0, i].set_xlabel("Re($\kappa$)")
        axes[0, i].set_ylim([-0.5, 1.5])
        axes[0, i].set_xticks([0., 0.5, 1.])
        axes[0, i].set_yticks([0., 0.5, 1.])
    axes[0, 1].set_title("Im($p$)")
    axes[0, 1].set(yticklabels=[])
    # axes[1, 0].set_title("Re($s$)")
    # axes[1, 1].set_title("Im($s$)")
    confmp.cbar_beside(fig, axes[0, :], im_diff, dx=0.01)
    # confmp.cbar_beside(fig, axes[1, :], im_sum, dx=0.01)
    axes[1, 0].set_xlabel("Re($\\kappa$)")
    axes[1, 0].set_ylabel("Im($\\kappa$)")
    # axes[1, 0].set_xlim([-0.6, 1.6])
    axes[1, 0].set_ylim([-0.6, 1.6])
    # ax1.set_title('Training points in kappa space')
    axes[1, 0].scatter(x=kappa_extra2.real, y=kappa_extra2.imag, marker='o', s=2, c=training_steps_color_extra2, cmap="viridis")
    # ax1.set_yticks([-0.5, 0., 0.5, 1., 1.5])
    axes[1, 0].add_patch(Rectangle((0.852, 1.047), 0.062, 0.0947,
                            edgecolor='black',
                            fill=False,
                            lw=0.5))
    con1 = ConnectionPatch(xyA=(0.8812, 1.0895), xyB=(0.914, 1.047),
                           coordsA="data", coordsB="data",
                           axesA=axes[1, 1], axesB=axes[1, 0],
                           color="black", lw=0.5)
    con2 = ConnectionPatch(xyA=(0.8812, 1.09755), xyB=(0.914, 1.1417),
                           coordsA="data", coordsB="data",
                           axesA=axes[1, 1], axesB=axes[1, 0],
                           color="black", lw=0.5)
    fig.add_artist(con1)
    fig.add_artist(con2)
    # ax2.set_xlabel("Re($\\kappa$)")
    # ax2.set_ylabel("Im($\\kappa$)")
    # ax2.set_title('No extra training points')
    axes[1, 1].set_xlim([0.8812, 0.88425])
    axes[1, 1].set_ylim([1.0895, 1.09755])
    # ax3.tick_params(labelleft=False, labelbottom=False)
    axes[1, 1].scatter(x=ep.real, y=ep.imag, marker='x', s=6, lw=0.5, c="tab:green", label="EP")
    axes[1, 1].set(yticklabels=[], xticklabels=[])
    axes[1, 1].tick_params(
        axis='both',          # changes apply to the both axes
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False) # labels along the bottom edge are off
    cb3 = axes[1, 1].scatter(x=kappa_extra2.real, y=kappa_extra2.imag, marker='8', s=2,
                      c=training_steps_color_extra2, cmap="viridis")
    axes[1, 1].legend()
    cbar, cax = confmp.cbar_beside(fig, axes[1, :], cb3, dx=0.01)
    cbar.set_label("\\# of training steps")
    axes[1, 1].text(0.5, -0.2, "$\\mathbb{L}_2$ norm: \\num{1.344e-6}", size=12, ha="center",
                    transform=axes[1, 1].transAxes)
    # fig.colorbar(cb3, ax=axes[1, 1], label="\\# of training steps", ticks=[0, 1, 2, 3])
    fig.savefig("../../mastersthesis/plots/plots/GPRPlane5DModelTrainedOnlyPWithKappaSpaceViridis.pdf")


def energy_plane_trained():
    ep = np.array([0.88278093 + 1.09360073j])
    df_extra2 = pd.read_csv('../../mastersthesis/plots/data/paper_data_kappa_trained_model_extra2.csv',
                            header=0, skiprows=0,
                            names=["kappa", "ev1", "ev2", "training_steps_color"])
    ev_extra2 = np.column_stack([np.array(df_extra2.ev1).astype(complex),
                                 np.array(df_extra2.ev2).astype(complex)])
    kappa_extra2 = np.array(df_extra2.kappa).astype(complex)
    training_steps_color_extra2 = np.array(df_extra2.training_steps_color)
    fig, axes = confmp.newfig(width=.61, left=36, right=41, bottom=33, top=9)
    im = axes.scatter(ev_extra2[:, 0].real, ev_extra2[:, 0].imag, marker='o', s=2,
                      c=training_steps_color_extra2, cmap='plasma')
    im = axes.scatter(ev_extra2[:, 1].real, ev_extra2[:, 1].imag, marker='o', s=2,
                      c=training_steps_color_extra2, cmap='plasma')
    axes.set_xlabel("Re($\lambda$)")
    axes.set_ylabel("Im($\lambda$)")
    pos = axes.get_position()
    cax = fig.add_axes((pos.xmax + 0.0164, pos.ymin, (10./72.) / fig.get_figwidth(), pos.ymax - pos.ymin))
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("\# of training steps")
    cbar.ax.set_yticks([0, 1, 2, 3])
    fig.savefig("../../mastersthesis/plots/plots/5DModelEnergyPlaneTrainedPosterSmall.pdf")


def compatibility_and_energy_plane_trained():
    ep = np.array([0.88278093 + 1.09360073j])
    df_extra2 = pd.read_csv('../../mastersthesis/plots/data/paper_data_kappa_trained_model_extra2.csv',
                            header=0, skiprows=0,
                            names=["kappa", "ev1", "ev2", "training_steps_color"])
    df_select_evs = pd.read_csv('../../mastersthesis/plots/data/paper_data_selecting_evs.csv',
                                header=0, skiprows=0,
                                names=["compatibility"])
    c = np.array(df_select_evs.compatibility)
    ev_extra2 = np.column_stack([np.array(df_extra2.ev1).astype(complex),
                                 np.array(df_extra2.ev2).astype(complex)])
    kappa_extra2 = np.array(df_extra2.kappa).astype(complex)
    training_steps_color_extra2 = np.array(df_extra2.training_steps_color)
    fig, axes = confmp.newfig(aspect=.61, ncols=2, left=34, right=42, bottom=34, top=4, wspace=40, gridspec=True,
                              width_ratios=np.array([1, 5]))
    ax1 = fig.add_subplot(axes[0])
    ax2 = fig.add_subplot(axes[1])
    x = [0 for _ in c]
    ax1.semilogy(x, abs(c), "_", ms=20, color=(0,81/256,158/256))
    ax1.set_ylabel("$c$")
    ax1.tick_params(labelbottom=False, bottom=False)
    im = ax2.scatter(ev_extra2[:, 0].real, ev_extra2[:, 0].imag, marker='o', s=2,
                      c=training_steps_color_extra2, cmap='plasma')
    im = ax2.scatter(ev_extra2[:, 1].real, ev_extra2[:, 1].imag, marker='o', s=2,
                      c=training_steps_color_extra2, cmap='plasma')
    ax2.set_xlabel("Re($\lambda$)")
    ax2.set_ylabel("Im($\lambda$)")
    pos = ax2.get_position()
    cax = fig.add_axes((pos.xmax + 0.01, pos.ymin, (10./72.) / fig.get_figwidth(), pos.ymax - pos.ymin))
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("\# of training steps")
    cbar.ax.set_yticks([0, 1, 2, 3])
    fig.savefig("../../mastersthesis/plots/plots/CompatibilityAnd5DModelEnergyPlaneTrainedPoster.pdf")


def compatibility_datasets():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    print(ev.shape)
    kappa = kappa[::11]
    model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
    model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
    gpflow_function = gpflow_model.get_model_generator()
    kappa_new = zs.zero_search(gpflow_function, kappa_0)
    kappa = np.concatenate((kappa, kappa_new))
    ev, compatibility = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum)
    df = pd.DataFrame()
    df['compatibility'] = compatibility.tolist()
    df.to_csv('../../mastersthesis/plots/data/compatibility_one_close.csv')

    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    print(ev.shape)
    kappa = kappa[::11]
    model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
    model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
    gpflow_function = gpflow_model.get_model_generator()
    kappa_new = zs.zero_search(gpflow_function, kappa_0)
    kappa = np.concatenate((kappa, kappa_new))
    ev, compatibility = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum, m=1)
    df = pd.DataFrame()
    df['compatibility'] = compatibility.tolist()
    df.to_csv('../../mastersthesis/plots/data/compatibility_two_close.csv')


def model1_datasets():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    n = False
    eps = 1.e-15
    eps_diff = 5.e-3
    for a in range(3):
        print("a: ", a)
        n = False
        i = 0
        kappa, phi = matrix.parametrization(kappa_0, r, steps)
        symmatrix = matrix.matrix_one_close_re(kappa)
        ev_new = matrix.eigenvalues(symmatrix)
        ev = dpp.initial_dataset(ev_new)
        ev = ev[::11]
        kappa = kappa[::11]
        plot_color = [0 for _ in kappa]
        all_kernel_ev = dict()
        ev_diff = ev[:, 0] - ev[:, 1]
        #print(ev_diff)
        distances = np.linalg.norm(np.array([ev_diff.real, ev_diff.imag]), axis=0)
        #print(distances)
        mean = np.mean(distances)
        #print(mean)
        distance_ev_all = [mean]
        print("ev shape: ", ev.shape)
        print("kappa shape: ", kappa.shape)
        while not n:
            model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
            model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
            all_kernel_ev[i] = kernel_ev_diff
            gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
            gpflow_function = gpflow_model.get_model_generator()
            kappa_new = zs.zero_search(gpflow_function, kappa_0)
            print("Root: ", kappa_new)
            kappa = np.concatenate((kappa, kappa_new))
            ev, compatibility = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum, m=0)
            i += 1
            plot_color.append(i)
            ev_diff = ev[-1, 0] - ev[-1, 1]
            distance_ev = np.linalg.norm(np.array([ev_diff.real, ev_diff.imag]))
            distance_ev_all.append(distance_ev)
            if a == 1 and i == 2:
                kappa_new = np.array([2 * kappa[-1] - kappa[-2]])
                kappa = np.concatenate((kappa, kappa_new))
                ev, c = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum, m=0)
                plot_color.append(i)
            if a == 2 and i == 3:
                kappa_new = np.array([2 * kappa[-1] - kappa[-2]])
                kappa = np.concatenate((kappa, kappa_new))
                ev, c = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum, m=0)
                plot_color.append(i)
            if abs(ev_diff.real) < eps_diff and abs(ev_diff.imag) < eps_diff:
                print("Found EP:")
                print(kappa_new)
                n = True
            if i == 25:
                print("Could not find EP:")
                print(kappa_new)
                n = True
        plots.parameter_plane_plotly(kappa, plot_color)
        df = pd.DataFrame()
        df['kappa'] = kappa.tolist()
        # df['kernel_ev'] = kernel_ev.tolist()
        # df['ev_diff'] = ev_diff.tolist()
        df = pd.concat([df, pd.DataFrame(ev)], axis=1)
        df['training_steps_color'] = plot_color
        df.columns = ['kappa', 'ev1', 'ev2', 'training_steps_color']
        df.to_csv('../../mastersthesis/plots/data/model1_training_%1d.csv' % a)
        df1 = pd.DataFrame()
        df1['ev_diff'] = distance_ev_all
        df1.to_csv('../../mastersthesis/plots/data/model1_convergence_ev_diff_%1d.csv' % a)
        with open('../../mastersthesis/plots/data/model1_convergence_kernel_ev_%1d.pkl' % a, 'wb') as f:
            pickle.dump(all_kernel_ev, f)


def model2_datasets():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    n = False
    eps = 1.e-15
    eps_diff = 5.e-3
    for a in range(3):
        print("a: ", a)
        n = False
        i = 0
        kappa, phi = matrix.parametrization(kappa_0, r, steps)
        symmatrix = matrix.matrix_two_close_im(kappa)
        ev_new = matrix.eigenvalues(symmatrix)
        ev = dpp.initial_dataset(ev_new)
        ev = ev[::11]
        kappa = kappa[::11]
        plot_color = [0 for _ in kappa]
        all_kernel_ev = dict()
        ev_diff = ev[:, 0] - ev[:, 1]
        #print(ev_diff)
        distances = np.linalg.norm(np.array([ev_diff.real, ev_diff.imag]), axis=0)
        #print(distances)
        mean = np.mean(distances)
        #print(mean)
        distance_ev_all = [mean]
        print("ev shape: ", ev.shape)
        print("kappa shape: ", kappa.shape)
        while not n:
            model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(ev, kappa)
            model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
            all_kernel_ev[i] = kernel_ev_diff
            gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
            gpflow_function = gpflow_model.get_model_generator()
            kappa_new = zs.zero_search(gpflow_function, kappa_0)
            print("Root: ", kappa_new)
            kappa = np.concatenate((kappa, kappa_new))
            ev, compatibility = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum, m=1)
            i += 1
            plot_color.append(i)
            ev_diff = ev[-1, 0] - ev[-1, 1]
            distance_ev = np.linalg.norm(np.array([ev_diff.real, ev_diff.imag]))
            distance_ev_all.append(distance_ev)
            if a == 1 and i == 2:
                kappa_new = np.array([2 * kappa[-1] - kappa[-2]])
                kappa = np.concatenate((kappa, kappa_new))
                ev, c = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum, m=1)
                plot_color.append(i)
            if a == 2 and i == 3:
                kappa_new = np.array([2 * kappa[-1] - kappa[-2]])
                kappa = np.concatenate((kappa, kappa_new))
                ev, c = dpp.getting_new_ev_of_ep(kappa_new, ev, model_diff, model_sum, m=1)
                plot_color.append(i)
            if abs(ev_diff.real) < eps_diff and abs(ev_diff.imag) < eps_diff:
                print("Found EP:")
                print(kappa_new)
                n = True
            if i == 25:
                print("Could not find EP:")
                print(kappa_new)
                n = True
        plots.parameter_plane_plotly(kappa, plot_color)
        df = pd.DataFrame()
        df['kappa'] = kappa.tolist()
        # df['kernel_ev'] = kernel_ev.tolist()
        # df['ev_diff'] = ev_diff.tolist()
        df = pd.concat([df, pd.DataFrame(ev)], axis=1)
        df['training_steps_color'] = plot_color
        df.columns = ['kappa', 'ev1', 'ev2', 'training_steps_color']
        df.to_csv('../../mastersthesis/plots/data/model2_training_%1d.csv' % a)
        df1 = pd.DataFrame()
        df1['ev_diff'] = distance_ev_all
        df1.to_csv('../../mastersthesis/plots/data/model2_convergence_ev_diff_%1d.csv' % a)
        with open('../../mastersthesis/plots/data/model2_convergence_kernel_ev_%1d.pkl' % a, 'wb') as f:
            pickle.dump(all_kernel_ev, f)


def gpr_model1_output_diff_colormap():
    ep = np.array([1.18503351 + 1.00848184j])
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    kappa = kappa[::11]
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(aspect=1., nrows=1, ncols=2, left=39, top=16, bottom=34, right=43,
                              wspace=5, sharey=True)  # left=46, bottom=34, wspace=54, right=51
    axes[0].set_title("Re($p$)", size=12)
    axes[0].set_ylabel("Im($\kappa$)")
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    for i in range(np.shape(axes)[0]):
        axes[i].set_xlabel("Re($\kappa$)")
        im_diff = confmp.imshow(axes[i], mean_diff[:, i].numpy().reshape(-100, 100), extent,
                                cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axes[i].scatter(x=ep.real, y=ep.imag, marker='x', s=8, c="tab:green", label="EP")
        axes[i].set_xticks([0.0, 0.5, 1])
        axes[i].set_yticks([0.0, 0.5, 1])
    axes[1].set_title("Im($p$)", size=12)
    cbar, cax = confmp.cbar_beside(fig, axes, im_diff, dx=0.01)
    #cbar.set_label("$\\left(\\lambda_1 - \\lambda_2\\right)^2$")
    #cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    #cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    confmp.subfig_label(axes[0], 0, 'left', 0, dx=-40, va='bottom', y=1, dy=3)
    confmp.subfig_label(axes[1], 1, 'left', 0, dx=-2, va='bottom', y=1, dy=3)
    plt.savefig("../../mastersthesis/plots/plots/5DGPRPlaneDiffModel1.pdf")


def gpr_model2_output_diff_colormap():
    ep = np.array([0.88278093 + 1.09360073j])
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    kappa = kappa[::11]
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(aspect=1., nrows=1, ncols=2, left=39, top=16, bottom=34, right=43,
                              wspace=5, sharey=True)  # left=46, bottom=34, wspace=54, right=51
    axes[0].set_title("Re($p$)", size=12)
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    axes[0].set_ylabel("Im($\kappa$)")
    for i in range(np.shape(axes)[0]):
        axes[i].set_xlabel("Re($\kappa$)")
        im_diff = confmp.imshow(axes[i], mean_diff[:, i].numpy().reshape(-100, 100), extent,
                                cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axes[i].scatter(x=ep.real, y=ep.imag, marker='x', s=8, c="tab:green", label="EP")
        axes[i].set_xticks([0.0, 0.5, 1])
        axes[i].set_yticks([0.0, 0.5, 1])
    axes[1].set_title("Im($p$)", size=12)
    cbar, cax = confmp.cbar_beside(fig, axes, im_diff, dx=0.01)
    #cbar.set_label("$\\left(\\lambda_1 - \\lambda_2\\right)^2$")
    #cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    #cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    confmp.subfig_label(axes[0], 0, 'left', 0, dx=-40, va='bottom', y=1, dy=3)
    confmp.subfig_label(axes[1], 1, 'left', 0, dx=-2, va='bottom', y=1, dy=3)
    plt.savefig("../../mastersthesis/plots/plots/5DGPRPlaneDiffModel2.pdf")


def gpr_model2_output_diff_colormap_presentation():
    ep = np.array([0.88278093 + 1.09360073j])
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    kappa = kappa[::11]
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(width=.4, aspect=1., nrows=2, ncols=1, left=34, top=16, bottom=29, right=38,
                              hspace=23, sharex=True)  # left=46, bottom=34, wspace=54, right=51
    axes[0].set_title("Re$(p)$", size=12)
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    axes[1].set_xlabel("Re$(\kappa)$")
    for i in range(np.shape(axes)[0]):
        axes[i].set_ylabel("Im$(\kappa)$")
        im_diff = confmp.imshow(axes[i], mean_diff[:, i].numpy().reshape(-100, 100), extent,
                                cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axes[i].scatter(x=ep.real, y=ep.imag, marker='x', s=8, c="tab:green", label="EP")
        axes[i].set_xticks([0.0, 0.5, 1])
        axes[i].set_yticks([0.0, 0.5, 1])
    axes[1].set_title("Im$(p)$", size=12)
    cbar, cax = confmp.cbar_beside(fig, axes, im_diff, dx=0.03)
    #cbar.set_label("$\\left(\\lambda_1 - \\lambda_2\\right)^2$")
    #cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    #cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    #confmp.subfig_label(axes[0], 0, 'left', 0, dx=-40, va='bottom', y=1, dy=3)
    #confmp.subfig_label(axes[1], 1, 'left', 0, dx=-2, va='bottom', y=1, dy=3)
    plt.savefig("../../talk/fig/5DGPRPlaneDiffModel2.pdf")


def gpr_both_models_output_diff_colormap():
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200

    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    kappa = kappa[::11]
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(aspect=1., nrows=2, ncols=2, left=39, top=16, bottom=34, right=43,
                              wspace=5, hspace=55, sharey=True)  # left=46, bottom=34, wspace=54, right=51
    axes[0,0].set_title("Re($p$)", size=12)
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    axes[0,0].set_ylabel("Im($\kappa$)")
    for i in range(np.shape(axes)[0]):
        axes[0,i].set_xlabel("Re($\kappa$)")
        im_diff = confmp.imshow(axes[0,i], mean_diff[:, i].numpy().reshape(-100, 100), extent,
                                cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axes[0,i].scatter(x=ep_one_close.real, y=ep_one_close.imag, marker='x', s=8, c="tab:green", label="EP")
        axes[0,i].set_xticks([0.0, 0.5, 1])
        axes[0,i].set_yticks([0.0, 0.5, 1])
    axes[0,1].set_title("Im($p$)", size=12)
    cbar, cax = confmp.cbar_beside(fig, axes[0], im_diff, dx=0.01)
    #cbar.set_label("$\\left(\\lambda_1 - \\lambda_2\\right)^2$")
    #cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    #cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    confmp.subfig_label(axes[0,0], 0, 'left', 0, dx=-40, va='bottom', y=1, dy=3)
    confmp.subfig_label(axes[0,1], 1, 'left', 0, dx=-2, va='bottom', y=1, dy=3)

    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    kappa = kappa[::11]
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    axes[1,0].set_title("Re($p$)", size=12)
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    axes[1,0].set_ylabel("Im($\kappa$)")
    for i in range(np.shape(axes)[0]):
        axes[1,i].set_xlabel("Re($\kappa$)")
        im_diff = confmp.imshow(axes[1,i], mean_diff[:, i].numpy().reshape(-100, 100), extent,
                                cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axes[1,i].scatter(x=ep_two_close.real, y=ep_two_close.imag, marker='x', s=8, c="tab:green", label="EP")
        axes[1,i].set_xticks([0.0, 0.5, 1])
        axes[1,i].set_yticks([0.0, 0.5, 1])
    axes[1,1].set_title("Im($p$)", size=12)
    cbar, cax = confmp.cbar_beside(fig, axes[1], im_diff, dx=0.01)
    #cbar.set_label("$\\left(\\lambda_1 - \\lambda_2\\right)^2$")
    #cbar.ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    #cbar.ax.set_yticklabels(["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"])
    confmp.subfig_label(axes[1,0], 2, 'left', 0, dx=-40, va='bottom', y=1, dy=3)
    confmp.subfig_label(axes[1,1], 3, 'left', 0, dx=-2, va='bottom', y=1, dy=3)
    plt.savefig("../../mastersthesis/plots/plots/5DGPRPlaneDiffBothModels.pdf")


def gpr_model2_output_diff_im_colormap_presentation():
    ep = np.array([0.88278093 + 1.09360073j])
    kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    r = 1
    steps = 200
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    symmatrix = matrix.matrix_two_close_im(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    ev = dpp.initial_dataset(ev_new)
    ev = ev[::11]
    kappa = kappa[::11]
    m_diff, kernel_ev = gpr.gp_2d_diff_kappa(ev, kappa)
    m_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(ev, kappa)
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, _ = m_diff.predict_f(grid)
    mean_sum, _ = m_sum.predict_f(grid)
    fig, axes = confmp.newfig(width=1.1615, aspect=1., nrows=1, ncols=1, left=30, right=290, top=88.4, bottom=30) #, #left=39, top=16, bottom=34, right=43,
                              #wspace=5, sharey=True)  # left=46, bottom=34, wspace=54, right=51
    max_diff = np.max(abs(mean_diff.numpy()))
    vmin_diff = -max_diff
    vmax_diff = max_diff
    confmp.imshow(axes, mean_diff[:, 1].numpy().reshape(-100, 100), extent,
                       cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
    axes.scatter(x=ep.real, y=ep.imag, marker='x', s=8, c="tab:green", label="EP")
    axes.tick_params(axis='both', which='both', bottom=False, top=False,
                     left=False, right=False, labelbottom=False, labelleft=False)
    fig.savefig("../../talk/config/titlefig.jpg")
    plt.close(fig)


if __name__ == '__main__':
    # plots.init_matplotlib()
    energy_plane_5d_model2()
    # energy_plane_5d_model2_presentation()
    # energy_plane_both_models()
    # gpr_plane_trained_only_p_with_kappa_space()
    # energy_plane_trained()
    # compatibility_and_energy_plane_trained()
    # compatibility_datasets()
    # model1_datasets()
    # model2_datasets()
    # gpr_model1_output_diff_colormap()
    # gpr_model2_output_diff_colormap()
    # gpr_model2_output_diff_colormap_presentation()
    # gpr_both_models_output_diff_colormap()
    #gpr_model2_output_diff_im_colormap_presentation()


    #kappa_0 = 0.5 + 0.5j  # 1.1847 + 1.0097j
    #r = 1
    #steps = 200
    #m = False
    #j = 0
    #i = 0
    #p = 16
    #eps = 1.e-15
    #eps_var = 1.e-6
    #eps_diff = 1.e-2
    # kappa, phi = matrix.parametrization(kappa_0, r, steps)
    # symmatrix = matrix.matrix_one_close_re(kappa)
    # ev_new = matrix.eigenvalues(symmatrix)
    # plots.energy_plane_plotly(ev_new, phi)
    # ev = dpp.initial_dataset(ev_new)

    """while not m:
        kappa, phi = matrix.parametrization(kappa_0, r, steps)
        kappa_new = kappa
        # plot_color = [0 for _ in kappa]
        # ev = np.empty((0, 2))
        symmatrix = matrix.matrix_one_close_re(kappa)
        ev_new = matrix.eigenvalues(symmatrix)
        # plots.energy_plane_plotly(ev_new, phi)
        ev = dpp.initial_dataset(ev_new)
        # plots.energy_plane_plotly(ev, phi)
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

    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    # plots.parameter_plane_plotly(kappa, phi)
    # plots.energy_plane_plotly(ev, phi)
    # plots.eigenvalues_angle_plotly(ev, phi, m_re, m_im)
    # plots.eigenvalues_angle_matplotlib(ev, phi, m_re, m_im)
    # plt.show()
