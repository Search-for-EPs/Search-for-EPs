"""Plots to analyze the results"""

import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
# import configurematplotlib as confmp


def init_matplotlib():
    """Initialize matplotlib with correct fontstyle etc."""
    plt.rc('font', **{
        'family': 'serif',
        'size': 14,
    })

    plt.rc('text', **{
        'usetex': True,
        'latex.preamble': '\\usepackage{siunitx} \\usepackage{amsmath} \\usepackage{amssymb} \\usepackage{physics}'
                          '\\usepackage{nicefrac}',
    })

    plt.style.use('tableau-colorblind10')


def linear(x, a, b):
    return a * x + b


def parameter_plane_plotly(kappa, phi):
    """Plotly plot for parameter plane

    Shows parameter plane where each kappa value is color coded by the respective angle.

    Parameters
    ----------
    kappa : np.ndarray
        Contains all complex values of kappa for the parameterized circle.
    phi : np.ndarray
        Contains all angles for the parameterized circle from 0 to 2pi. Used for color code.
    """
    fig = px.scatter(x=kappa[:].real, y=kappa[:].imag, color=phi, labels=dict(x="Re(\\kappa)", y="Im(\\kappa)",
                                                                              color="Angle"))
    fig.show()


def parameter_plane_matplotlib(kappa, phi):
    """Matplotlib plot for parameter plane

    Shows parameter plane where each kappa value is color coded by the respective angle.

    Parameters
    ----------
    kappa : np.ndarray
        Contains all complex values of kappa for the parameterized circle.
    phi : np.ndarray
        Contains all angles for the parameterized circle from 0 to 2pi. Used for color code.
    """
    plt.xlabel("Re(\\kappa)")
    plt.ylabel("Im(\\kappa)")
    plt.scatter(x=kappa[:].real, y=kappa[:].imag, c=phi, cmap="plasma")
    plt.colorbar("Angle / \\si{\\degree}")
    # plt.show()


def energy_plane_plotly(ev, phi):
    """Plotly plot for energy plane

    Shows energy plane where each eigenvalue is color coded by the respective angle.

    Parameters
    ----------
    ev : np.ndarray
        Contains all complex eigenvalues for each matrix of the parameterized circle.
    phi : np.ndarray
        Contains all angles for the parameterized circle from 0 to 2pi. Used for color code.
    """
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev)[1])]).ravel())
    fig = px.scatter(x=ev.ravel().real, y=ev.ravel().imag, color=phi_all.tolist(),
                     labels=dict(x="Re(\\lambda)", y="Im(\\lambda)", color="Angle / \\si{\\degree}"))
    fig.show()


def energy_plane_matplotlib(ev, phi):
    """Matplotlib plot for energy plane

    Shows energy plane where each eigenvalue is color coded by the respective angle.

    Parameters
    ----------
    ev : np.ndarray
        Contains all complex eigenvalues for each matrix of the parameterized circle.
    phi : np.ndarray
        Contains all angles for the parameterized circle from 0 to 2pi. Used for color code.
    """
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev)[1])]).ravel())
    plt.xlabel("Re(\\lambda)")
    plt.ylabel("Im(\\lambda)")
    plt.scatter(x=ev.ravel().real, y=ev.ravel().imag, c=phi_all, cmap="plasma")
    plt.colorbar(label="Angle / \\si{\\degree}")
    # plt.show()


def eigenvalues_angle_plotly(ev, phi, m_re, m_im):
    """Plotly plot for GPR model prediction of the eigenvalue difference with respect to the angle

    Shows the GPR model predictions for the real and imaginary part of the eigenvalue difference with respect to the
    angle. The mean function as well as the variance is illustrated. Illustration of variance does not work in plotly.

    Parameters
    ----------
    ev : np.ndarray
        Contains all complex eigenvalues for each matrix of the parameterized circle.
    phi : np.ndarray
        Contains all angles for the parameterized circle from 0 to 2pi. Used for color code.
    m_re : gpflow.models.GPR
        GPR model for the real part of the eigenvalue difference.
    m_im: gpflow.models.GPR
        GPR model for the imaginary part of the eigenvalue difference.
    """
    # ev_sum = ev[::, 0] + ev[::, 1]
    ev_diff = (ev[::, 0] - ev[::, 1]) ** 2
    xx = np.linspace(0.0, 2 * np.pi, 400).reshape(-1, 1)
    mean_re, var_re = m_re.predict_f(xx)
    mean_im, var_im = m_im.predict_f(xx)
    x = xx.ravel()
    # x_rev = x[::-1]
    # var_re_up = mean_re.numpy().ravel() + 1.96 * np.sqrt(var_re.numpy().ravel())
    # var_re_low = mean_re.numpy().ravel() - 1.96 * np.sqrt(var_re.numpy().ravel())
    # var_re_low = var_re_low[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi, y=ev_diff.real, mode='markers', name="Re"))
    fig.add_trace(go.Scatter(x=phi, y=ev_diff.imag, mode='markers', name="Im"))
    fig.add_trace(go.Scatter(x=x, y=mean_re.numpy().ravel(), mode='lines', name="GPR Re",
                             line=dict(color='#636EFA')))
    # fig.add_trace(go.Scatter(x=x + x_rev, y=var_re_up+var_re_low, fill='toself',
    #                         fillcolor='rgba(99,110,250,0.2)', line_color='rgba(255,255,255,0)', showlegend=True))
    fig.add_trace(go.Scatter(x=x, y=mean_im.numpy().ravel(), mode='lines', name="GPR Im",
                             line=dict(color='#EF553B')))
    # fig.add_trace(go.Scatter(x=x + x_rev, y=y1_upper + y1_lower, fill='toself', fillcolor='rgba(0,100,80,0.2)',
    #                          line_color='rgba(255,255,255,0)', showlegend=False, name='Fair'))
    # fig.update_layout(labels=dict(x="Angle / \\si{\\degree}", y="$(\\lambda_1 - \\lambda_2)^2"))
    fig.show()


def eigenvalues_angle_matplotlib(ev, phi, m_re, m_im):
    """Matplotlib plot for GPR model prediction of the eigenvalue difference with respect to the angle

    Shows the GPR model predictions for the real and imaginary part of the eigenvalue difference with respect to the
    angle. The mean function as well as the variance is illustrated.

    Parameters
    ----------
    ev : np.ndarray
        Contains all complex eigenvalues for each matrix of the parameterized circle.
    phi : np.ndarray
        Contains all angles for the parameterized circle from 0 to 2pi. Used for color code.
    m_re : gpflow.models.GPR
        GPR model for the real part of the eigenvalue difference.
    m_im : gpflow.models.GPR
        GPR model for the imaginary part of the eigenvalue difference.
    """
    # ev_sum = ev[::, 0] + ev[::, 1]
    ev_diff = (ev[::, 0] - ev[::, 1]) ** 2
    xx = np.linspace(0.0, 2 * np.pi, 400).reshape(-1, 1)
    mean_re, var_re = m_re.predict_f(xx)
    mean_im, var_im = m_im.predict_f(xx)
    plt.xlabel("Angle / rad")
    plt.ylabel("$(\\lambda_1 - \\lambda_2)^2$")
    # plt.scatter(phi, ev_sum.real)
    # plt.scatter(phi, ev_sum.imag)
    plt.scatter(phi, ev_diff.real, label="Re")
    plt.scatter(phi, ev_diff.imag, label="Im")
    plt.legend()
    plt.plot(xx, mean_re, "C0", lw=2)
    plt.plot(xx, mean_im, color="tab:orange", lw=2)
    plt.fill_between(
        xx[:, 0],
        mean_re[:, 0] - 1.96 * np.sqrt(var_re[:, 0]),
        mean_re[:, 0] + 1.96 * np.sqrt(var_re[:, 0]),
        color="C0",
        alpha=0.2,
    )
    plt.fill_between(
        xx[:, 0],
        mean_im[:, 0] - 1.96 * np.sqrt(var_im[:, 0]),
        mean_im[:, 0] + 1.96 * np.sqrt(var_im[:, 0]),
        color="tab:orange",
        alpha=0.2,
    )
    # plt.show()


def three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im):
    """3D plotly plot for eigenvalues

    The model for the preprocessed eigenvalues with respect to kappa is plotted with plotly. Two models one for the
    real and one for the imaginary part of the preprocessed eigenvalues are given and plotted separately.

    Parameters
    ----------
    kappa_0
        Center of circulation in parameter space
    r
        Radius of circulation in parameter space
    m_re : gpflow.models.GPR
        GPR model for the real part of the preprocessed eigenvalues
    m_im : gpflow.models.GPR
        GPR model for the imaginary part of the preprocessed eigenvalues
    """
    x = np.linspace(kappa_0.real - r, kappa_0.real + r, 50)
    y = np.linspace(kappa_0.imag - r, kappa_0.imag + r, 50)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_re, var_re = m_re.predict_f(grid)
    mean_im, var_im = m_im.predict_f(grid)
    df = px.data.iris()
    fig_re = px.scatter_3d(df, x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean_re.numpy().ravel(),
                           color=mean_re.numpy().ravel())
    fig_im = px.scatter_3d(df, x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean_im.numpy().ravel(),
                           color=mean_im.numpy().ravel())
    fig_re.show()
    fig_im.show()


def three_d_eigenvalue_kappa_2d_model_plotly(kappa_0, r, m):
    """3D plotly plot for eigenvalues

    The 2D model for preprocessed eigenvalues is plotted in two plots. One shows the real and the other one the
    imaginary part.

    Parameters
    ----------
    kappa_0
        Center of circulation in parameter space
    r
        Radius of circulation in parameter space as part of the field strength
    m : gpflow.models.GPR
        2D GPR model for the preprocessed eigenvalues
    """
    x = np.linspace(kappa_0.real - r, kappa_0.real + r, 20)
    y = np.linspace(kappa_0.imag - r, kappa_0.imag + r, 20)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean, var = m.predict_f(grid)
    df = px.data.iris()
    fig_re = px.scatter_3d(df, x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean.numpy()[::, 0],
                           color=mean.numpy()[::, 0])
    fig_im = px.scatter_3d(df, x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean.numpy()[::, 1],
                           color=mean.numpy()[::, 1])
    fig_re.show()
    fig_im.show()
    # fig_re.write_html("../my_calculations/Punkt29/Punkt29_model_real.html")
    # fig_im.write_html("../my_calculations/Punkt29/Punkt29_model_imag.html")


def control_model_2d_plotly(kappa, ev_diff, ev_sum, model_diff, model_sum):
    """Plotly plot to control model accuracy

    2 times 4 subplots are plotted to compare the smoothness and accuracy of the models to the exact values.
    4 subplots are needed to plot all possible combinations of a 2D x 2D model.

    Parameters
    ----------
    kappa : np.ndarray
        1D complex array containing all kappa values
    ev_diff : np.ndarray
        1D complex array containing the eigenvalue difference squared for each step
    ev_sum : np.ndarray
        1D complex array containing the eigenvalue sum for each step
    model_diff : gpflow.models.GPR
        2D GPR model of the eigenvalue difference squared
    model_sum : gpflow.models.GPR
        2D GPR model of the eigenvalue sum
    """
    xdata = np.linspace(kappa.real[0], kappa.real[-1], 400)
    z = np.polyfit(kappa.real, kappa.imag, 3)
    p = np.poly1d(z)
    grid = np.column_stack((xdata, p(xdata)))
    # grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    grid1 = np.column_stack((kappa.real, kappa.imag))
    # grid1 = np.array((x1.ravel(), y1.ravel())).T
    kappa_mean_diff, kappa_var_diff = model_diff.predict_f(grid1)
    kappa_mean_sum, kappa_var_sum = model_sum.predict_f(grid1)
    kappa_sep = [kappa.real, kappa.real, kappa.imag, kappa.imag]
    con_sep = [xdata, xdata, p(xdata), p(xdata)]
    ev_diff_sep = [ev_diff.real, ev_diff.imag, ev_diff.real, ev_diff.imag]
    ev_sum_sep = [ev_sum.real, ev_sum.imag, ev_sum.real, ev_sum.imag]
    kappa_mean_diff_sep = [kappa_mean_diff.numpy()[:, 0], kappa_mean_diff.numpy()[:, 1], kappa_mean_diff.numpy()[:, 0],
                           kappa_mean_diff.numpy()[:, 1]]
    kappa_mean_sum_sep = [kappa_mean_sum.numpy()[:, 0], kappa_mean_sum.numpy()[:, 1], kappa_mean_sum.numpy()[:, 0],
                          kappa_mean_sum.numpy()[:, 1]]
    mean_diff_sep = [mean_diff.numpy()[:, 0], mean_diff.numpy()[:, 1], mean_diff.numpy()[:, 0], mean_diff.numpy()[:, 1]]
    mean_sum_sep = [mean_sum.numpy()[:, 0], mean_sum.numpy()[:, 1], mean_sum.numpy()[:, 0], mean_sum.numpy()[:, 1]]
    row = [1, 1, 2, 2]
    col = [1, 2, 1, 2]
    legend = [True, False, False, False]
    x_axis_label = ["Re(\\kappa)", "Re(\\kappa)", "Im(\\kappa)", "Im(\\kappa)"]
    y_axis_label_diff = ["Re(s)", "Im(s)", "Re(s)", "Im(s)"]
    y_axis_label_sum = ["Re(p)", "Im(p)", "Re(p)", "Im(p)"]
    fig_diff = make_subplots(rows=2, cols=2, subplot_titles=("re_re", "re_im", "im_re", "im_im"))
    fig_sum = make_subplots(rows=2, cols=2, subplot_titles=("re_re", "re_im", "im_re", "im_im"))
    for i in range(4):
        fig_diff.add_trace(go.Scatter(x=kappa_sep[i], y=ev_diff_sep[i], mode='markers', name='exact',
                                      showlegend=legend[i], marker=dict(color="#636EFA")), row=row[i], col=col[i])
        fig_diff.add_trace(go.Scatter(x=kappa_sep[i], y=kappa_mean_diff_sep[i], mode='markers', name='model',
                                      showlegend=legend[i], marker=dict(color="#EF553B")), row=row[i], col=col[i])
        fig_diff.add_trace(go.Scatter(x=con_sep[i], y=mean_diff_sep[i], mode='lines', name='continuity',
                                      showlegend=legend[i], line=dict(color="#00CC96")), row=row[i], col=col[i])
        fig_sum.add_trace(go.Scatter(x=kappa_sep[i], y=ev_sum_sep[i], mode='markers', name='exact',
                                     showlegend=legend[i], marker=dict(color="#636EFA")), row=row[i], col=col[i])
        fig_sum.add_trace(go.Scatter(x=kappa_sep[i], y=kappa_mean_sum_sep[i], mode='markers', name='model',
                                     showlegend=legend[i], marker=dict(color="#EF553B")), row=row[i], col=col[i])
        fig_sum.add_trace(go.Scatter(x=con_sep[i], y=mean_sum_sep[i], mode='lines', name='continuity',
                                     showlegend=legend[i], line=dict(color="#00CC96")), row=row[i], col=col[i])
        fig_diff.update_xaxes(title_text=x_axis_label[i], row=row[i], col=col[i])
        fig_diff.update_yaxes(title_text=y_axis_label_diff[i], row=row[i], col=col[i])
        fig_sum.update_xaxes(title_text=x_axis_label[i], row=row[i], col=col[i])
        fig_sum.update_yaxes(title_text=y_axis_label_sum[i], row=row[i], col=col[i])
    fig_diff.update_layout(title_text="Diff")
    fig_sum.update_layout(title_text="Sum")
    fig_diff.show()
    fig_sum.show()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=kappa.real, y=kappa.imag, mode="markers"))
    fig1.add_trace(go.Scatter(x=xdata, y=p(xdata), mode="lines"))
    fig1.show()


def control_model_2d_matplotlib(kappa, ev_diff, ev_sum, model_diff, model_sum):
    """Matplotlib plot to control model accuracy

    2 times 4 subplots are plotted to compare the smoothness and accuracy of the models to the exact values.
    4 subplots are needed to plot all possible combinations of a 2D x 2D model.

    Parameters
    ----------
    kappa : np.ndarray
        1D complex array containing all kappa values
    ev_diff : np.ndarray
        1D complex array containing the eigenvalue difference squared for each step
    ev_sum : np.ndarray
        1D complex array containing the eigenvalue sum for each step
    model_diff : gpflow.models.GPR
        2D GPR model of the eigenvalue difference squared
    model_sum : gpflow.models.GPR
        2D GPR model of the eigenvalue sum
    """
    xdata = np.linspace(kappa.real[0], kappa.real[-1], 400)
    z = np.polyfit(kappa.real, kappa.imag, 3)
    p = np.poly1d(z)
    grid = np.column_stack((xdata, p(xdata)))
    # grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    grid1 = np.column_stack((kappa.real, kappa.imag))
    # grid1 = np.array((x1.ravel(), y1.ravel())).T
    kappa_mean_diff, kappa_var_diff = model_diff.predict_f(grid1)
    kappa_mean_sum, kappa_var_sum = model_sum.predict_f(grid1)
    kappa_sep = [kappa.real, kappa.real, kappa.imag, kappa.imag]
    con_sep = [xdata, xdata, p(xdata), p(xdata)]
    ev_diff_sep = [ev_diff.real, ev_diff.imag, ev_diff.real, ev_diff.imag]
    ev_sum_sep = [ev_sum.real, ev_sum.imag, ev_sum.real, ev_sum.imag]
    kappa_mean_diff_sep = [kappa_mean_diff[:, 0], kappa_mean_diff[:, 1], kappa_mean_diff[:, 0], kappa_mean_diff[:, 1]]
    kappa_mean_sum_sep = [kappa_mean_sum[:, 0], kappa_mean_sum[:, 1], kappa_mean_sum[:, 0], kappa_mean_sum[:, 1]]
    mean_diff_sep = [mean_diff[:, 0], mean_diff[:, 1], mean_diff[:, 0], mean_diff[:, 1]]
    mean_sum_sep = [mean_sum[:, 0], mean_sum[:, 1], mean_sum[:, 0], mean_sum[:, 1]]
    x_axis_label = ["$\\Re(\\kappa)$", "$\\Re(\\kappa)$", "$\\Im(\\kappa)$", "$\\Im(\\kappa)$"]
    y_axis_label_diff = ["$\\Re(p)$", "$\\Im(p)$", "$\\Re(p)$", "$\\Im(p)$"]
    y_axis_label_sum = ["$\\Re(s)$", "$\\Im(s)$", "$\\Re(s)$", "$\\Im(s)$"]
    fig_diff, ax_diff = plt.subplots(2, 2, constrained_layout=True, figsize=(9, 6.75))
    fig_diff.suptitle("Model accuracy for $p = \\left(\\lambda_1 - \\lambda_2\\right)^2$")
    for i, ax in enumerate(ax_diff.flatten()):
        ax.set_xlabel(x_axis_label[i])
        ax.set_ylabel(y_axis_label_diff[i])
        ax.scatter(x=kappa_sep[i], y=ev_diff_sep[i], s=20, label="exact")
        ax.scatter(x=kappa_sep[i], y=kappa_mean_diff_sep[i], s=20, label="model")
        ax.plot(con_sep[i], mean_diff_sep[i], color="tab:green", linewidth=1.5, label="continuity")
        ax.legend()
    fig_diff.savefig("docs/source/_pages/images/model_accuracy_diff-2.pdf")
    fig_diff.savefig("docs/source/_pages/images/model_accuracy_diff-2.png")
    fig_sum, ax_sum = plt.subplots(2, 2, constrained_layout=True, figsize=(9, 6.75))
    fig_sum.suptitle("Model accuracy for $s = \\frac{1}{2} \\cdot \\left(\\lambda_1 + \\lambda_2\\right)$")
    for i, ax in enumerate(ax_sum.flatten()):
        ax.set_xlabel(x_axis_label[i])
        ax.set_ylabel(y_axis_label_sum[i])
        ax.scatter(x=kappa_sep[i], y=ev_sum_sep[i], s=20, label="exact")
        ax.scatter(x=kappa_sep[i], y=kappa_mean_sum_sep[i], s=20, label="model")
        ax.plot(con_sep[i], mean_sum_sep[i], color="tab:green", linewidth=1.5, label="continuity")
        ax.legend()
    fig_sum.savefig("docs/source/_pages/images/model_accuracy_sum-2.pdf")
    fig_sum.savefig("docs/source/_pages/images/model_accuracy_sum-2.png")


def control_model_3d_plotly(kappa, ev_diff, ev_sum, model_diff, model_sum):
    x = np.linspace(min(kappa.real), max(kappa.real), 50)
    y = np.linspace(min(kappa.imag), max(kappa.imag), 50)
    xx, yy = np.meshgrid(x, y)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    # df = px.data.iris()
    fig_diff_re = go.Figure()
    fig_diff_re.add_trace(go.Scatter3d(x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean_diff.numpy()[::, 0],
                                       marker=dict(color=mean_diff.numpy()[::, 0])))
    fig_diff_re.add_trace(go.Scatter3d(x=kappa.real, y=kappa.imag, z=ev_diff.real))
    fig_diff_im = go.Figure()
    fig_diff_im.add_trace(go.Scatter3d(x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean_diff.numpy()[::, 1],
                                       marker=dict(color=mean_diff.numpy()[::, 1])))
    fig_diff_re.add_trace(go.Scatter3d(x=kappa.real, y=kappa.imag, z=ev_diff.imag))
    fig_diff_re.show()
    fig_diff_im.show()
    fig_sum_re = go.Figure()
    fig_sum_re.add_trace(go.Scatter3d(x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean_sum.numpy()[::, 0],
                                      marker=dict(color=mean_sum.numpy()[::, 0])))
    fig_sum_re.add_trace(go.Scatter3d(x=kappa.real, y=kappa.imag, z=ev_sum.real))
    fig_sum_im = go.Figure()
    fig_sum_im.add_trace(go.Scatter3d(x=grid[::, 0].ravel(), y=grid[::, 1].ravel(), z=mean_sum.numpy()[::, 1],
                                      marker=dict(color=mean_sum.numpy()[::, 1])))
    fig_sum_im.add_trace(go.Scatter3d(x=kappa.real, y=kappa.imag, z=ev_sum.imag))
    fig_sum_re.show()
    fig_sum_im.show()


def model_noise_dependency_plotly(kappa_with_noise, kappa_no_noise, training_steps_color):
    """Plotly plot for noise dependency of the model

    2 subplots are plotted to compare the behavior of the model for a default noise variance and a fixed small
    noise variance. The complex kappa plane shows the model predictions for the EP of each training steps.

    Parameters
    ----------
    kappa_with_noise : np.ndarray
        1D complex array containing all kappa values of the model with the default noise variance
    kappa_no_noise : np.ndarray
        1D complex array containing all kappa values of the model with the fixed small noise variance
    training_steps_color : np.ndarray
        1D array containing the training steps which is used for the color bar
    """
    """df = pd.read_csv('model_noise_dependency_55_color.csv', header=0, skiprows=0,
                     names=["kappa_with_noise", "kappa_no_noise", "ev_with_noise", "ev_no_noise",
                            "training_steps_color"])
    kappa_no_noise = np.array(df.kappa_no_noise).astype(complex)
    kappa_with_noise = np.array(df.kappa_with_noise).astype(complex)
    training_steps_color = np.array(df.training_steps_color)"""
    ep = np.array([1.18503351 + 1.00848184j])
    fig1 = px.scatter(x=kappa_with_noise.real, y=kappa_with_noise.imag, color=training_steps_color.tolist(),
                      labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    fig2 = px.scatter(x=kappa_no_noise.real, y=kappa_no_noise.imag, color=training_steps_color.tolist(),
                      labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    fig_ep = px.scatter(x=ep.real, y=ep.imag, color_discrete_sequence=['#00CC96'], color=["EP"],
                        labels=dict(x="Re(EP)", y="Im(EP)"))
    fig_ep_no = px.scatter(x=ep.real, y=ep.imag, color_discrete_sequence=['#00CC96'],
                           labels=dict(x="Re(EP)", y="Im(EP)"))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15,
                        subplot_titles=("Default noise variance setting",
                                        "Small fixed noise variance"))
    fig.add_trace(fig1["data"][0], row=1, col=1)
    fig.add_trace(fig_ep["data"][0], row=1, col=1)
    fig.add_trace(fig2["data"][0], row=1, col=2)
    fig.add_trace(fig_ep_no["data"][0], row=1, col=2)
    fig.update_xaxes(title_text="Re(kappa)", row=1, col=1)
    fig.update_xaxes(title_text="Re(kappa)", row=1, col=2)
    fig.update_yaxes(title_text="Im(kappa)", row=1, col=1)
    fig.update_yaxes(title_text="Im(kappa)", row=1, col=2)
    fig.update_layout(coloraxis={'colorbar': dict(title="# of trai-<br>ning steps", len=0.9)})
    fig.show()
    # fig.write_html("docs/source/_pages/images/model_noise_dependency_55-3")


def model_noise_dependency_matplotlib(kappa_with_noise, kappa_no_noise, training_steps_color):
    """Matplotlib plot for noise dependency of the model

    2 subplots are plotted to compare the behavior of the model for a default noise variance and a fixed small
    noise variance. The complex kappa plane shows the model predictions for the EP of each training steps.

    Parameters
    ----------
    kappa_with_noise : np.ndarray
        1D complex array containing all kappa values of the model with the default noise variance
    kappa_no_noise : np.ndarray
        1D complex array containing all kappa values of the model with the fixed small noise variance
    training_steps_color : np.ndarray
        1D array containing the training steps which is used for the color bar
    """
    """df = pd.read_csv('model_noise_dependency_55_color.csv', header=0, skiprows=0,
                     names=["kappa_with_noise", "kappa_no_noise", "ev_with_noise", "ev_no_noise",
                            "training_steps_color"])
    kappa_no_noise = np.array(df.kappa_no_noise).astype(complex)
    kappa_with_noise = np.array(df.kappa_with_noise).astype(complex)
    training_steps_color = np.array(df.training_steps_color)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))
    ax1.set_title("With noise (default: $\\text{noise variance} = 1$)")
    ax1.set_xlabel("$\\Re(\\kappa)$")
    ax1.set_ylabel("$\\Im(\\kappa)$")
    ax1.scatter(x=kappa_with_noise[:].real, y=kappa_with_noise[:].imag, c=training_steps_color, cmap="inferno")
    ax2.set_title("Without noise $\\left(\\text{noise variance} \\approx \\num{e-6})\\right)$")
    ax2.set_xlabel("$\\Re(\\kappa)$")
    ax2.set_ylabel("$\\Im(\\kappa)$")
    cb = ax2.scatter(x=kappa_no_noise[:].real, y=kappa_no_noise[:].imag, c=training_steps_color, cmap="inferno")
    fig.colorbar(cb, ax=ax2, label="\\# of training steps")
    fig.savefig("docs/source/_pages/images/model_noise_dependency_55-2.pdf")
    fig.savefig("docs/source/_pages/images/model_noise_dependency_55-2.png")


def entropy_kernel_ev_matplotlib():
    """Matplotlib plot for entropy

    The entropy is calculated from the kernel eigenvalues and plotted over the number of training steps.
    """
    df = pd.read_csv("data_kernel_ev_25.csv", skiprows=0)
    entropy = []
    for i in range(1, df.shape[1]):
        kernel_ev = np.array(df.iloc[:, i])
        kernel_ev = kernel_ev[~np.isnan(kernel_ev)]
        entropy_step = -np.sum((kernel_ev / np.linalg.norm(kernel_ev)) * np.log(kernel_ev / np.linalg.norm(kernel_ev)))
        entropy.append(entropy_step)
    fig1 = plt.figure(1, figsize=(10, 7.5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_ylabel("Entropy")
    ax1.set_xlabel("\\# of training steps")
    ax1.plot(entropy)
    ax1.set_title("Entropy calculated with $\\nicefrac{\\lambda_{\\text{k}, i}}{\\left|\\vec{\\lambda}_\\text{k}\\right|}$")
    # print(fig1.get_size_inches())
    # fig1.savefig("entropy_diff_normalized_3.pdf")
    # fig1.savefig("entropy_diff_normalized_3.png")


def check_selected_evs():
    pass
    # for i in range(26)
