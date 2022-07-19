"""Plots to analyze the results"""

import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def init_matplotlib():
    """Initialize matplotlib with correct fontstyle etc."""
    plt.rc('font', **{
        'family': 'serif',
        'size': 14,
    })

    plt.rc('text', **{
        'usetex': True,
        'latex.preamble': '\\usepackage{siunitx}',
    })


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
    m_im: gpflow.models.GPR
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
    x = np.linspace(kappa_0.real - r, kappa_0.real + r, 50)
    y = np.linspace(kappa_0.imag - r, kappa_0.imag + r, 50)
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


def control_model_2d_plotly(kappa, ev_diff, ev_sum, model_diff, model_sum):
    xdata = np.linspace(kappa.real[0], kappa.real[-1], 400)
    z = np.polyfit(kappa.real, kappa.imag, 5)
    p = np.poly1d(z)
    grid = np.column_stack((xdata, p(xdata)))
    # grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    grid1 = np.column_stack((kappa.real, kappa.imag))
    # grid1 = np.array((x1.ravel(), y1.ravel())).T
    kappa_mean_diff, kappa_var_diff = model_diff.predict_f(grid1)
    kappa_mean_sum, kappa_var_sum = model_sum.predict_f(grid1)
    fig_diff = make_subplots(rows=2, cols=2, subplot_titles=("re_re", "re_im", "im_re", "im_im"))
    fig_diff.add_trace(go.Scatter(x=kappa.real, y=ev_diff.real, mode='markers', name='exact',
                                  marker=dict(color="#636EFA")), row=1, col=1)
    fig_diff.add_trace(go.Scatter(x=kappa.real, y=kappa_mean_diff.numpy()[::, 0], mode='markers', name='model',
                                  marker=dict(color="#EF553B")), row=1, col=1)
    fig_diff.add_trace(go.Scatter(x=xdata, y=mean_diff.numpy()[::, 0], mode='lines', name='continuity',
                                  line=dict(color="#00CC96")), row=1, col=1)
    fig_diff.add_trace(go.Scatter(x=kappa.real, y=ev_diff.imag, mode='markers', showlegend=False,
                                  marker=dict(color="#636EFA")), row=1, col=2)
    fig_diff.add_trace(go.Scatter(x=kappa.real, y=kappa_mean_diff.numpy()[::, 1], mode='markers', showlegend=False,
                                  marker=dict(color="#EF553B")), row=1, col=2)
    fig_diff.add_trace(go.Scatter(x=xdata, y=mean_diff.numpy()[::, 1], mode='lines', showlegend=False,
                                  line=dict(color="#00CC96")), row=1, col=2)
    fig_diff.add_trace(go.Scatter(x=kappa.imag, y=ev_diff.real, mode='markers', showlegend=False,
                                  marker=dict(color="#636EFA")), row=2, col=1)
    fig_diff.add_trace(go.Scatter(x=kappa.imag, y=kappa_mean_diff.numpy()[::, 0], mode='markers', showlegend=False,
                                  marker=dict(color="#EF553B")), row=2, col=1)
    fig_diff.add_trace(go.Scatter(x=p(xdata), y=mean_diff.numpy()[::, 0], mode='lines', showlegend=False,
                                  line=dict(color="#00CC96")), row=2, col=1)
    fig_diff.add_trace(go.Scatter(x=kappa.imag, y=ev_diff.imag, mode='markers', showlegend=False,
                                  marker=dict(color="#636EFA")), row=2, col=2)
    fig_diff.add_trace(go.Scatter(x=kappa.imag, y=kappa_mean_diff.numpy()[::, 1], mode='markers', showlegend=False,
                                  marker=dict(color="#EF553B")), row=2, col=2)
    fig_diff.add_trace(go.Scatter(x=p(xdata), y=mean_diff.numpy()[::, 1], mode='lines', showlegend=False,
                                  line=dict(color="#00CC96")), row=2, col=2)
    fig_sum = make_subplots(rows=2, cols=2, subplot_titles=("re_re", "re_im", "im_re", "im_im"))
    fig_sum.add_trace(go.Scatter(x=kappa.real, y=ev_sum.real, mode='markers', name='exact',
                                 marker=dict(color="#636EFA")), row=1, col=1)
    fig_sum.add_trace(go.Scatter(x=kappa.real, y=kappa_mean_sum.numpy()[::, 0], mode='markers', name='model',
                                 marker=dict(color="#EF553B")), row=1, col=1)
    fig_sum.add_trace(go.Scatter(x=xdata, y=mean_sum.numpy()[::, 0], mode='lines', name='continuity',
                                 line=dict(color="#00CC96")), row=1, col=1)
    fig_sum.add_trace(go.Scatter(x=kappa.real, y=ev_sum.imag, mode='markers', showlegend=False,
                                 marker=dict(color="#636EFA")), row=1, col=2)
    fig_sum.add_trace(go.Scatter(x=kappa.real, y=kappa_mean_sum.numpy()[::, 1], mode='markers', showlegend=False,
                                 marker=dict(color="#EF553B")), row=1, col=2)
    fig_sum.add_trace(go.Scatter(x=xdata, y=mean_sum.numpy()[::, 1], mode='lines', showlegend=False,
                                 line=dict(color="#00CC96")), row=1, col=2)
    fig_sum.add_trace(go.Scatter(x=kappa.imag, y=ev_sum.real, mode='markers', showlegend=False,
                                 marker=dict(color="#636EFA")), row=2, col=1)
    fig_sum.add_trace(go.Scatter(x=kappa.imag, y=kappa_mean_sum.numpy()[::, 0], mode='markers', showlegend=False,
                                 marker=dict(color="#EF553B")), row=2, col=1)
    fig_sum.add_trace(go.Scatter(x=p(xdata), y=mean_sum.numpy()[::, 0], mode='lines', showlegend=False,
                                 line=dict(color="#00CC96")), row=2, col=1)
    fig_sum.add_trace(go.Scatter(x=kappa.imag, y=ev_sum.imag, mode='markers', showlegend=False,
                                 marker=dict(color="#636EFA")), row=2, col=2)
    fig_sum.add_trace(go.Scatter(x=kappa.imag, y=kappa_mean_sum.numpy()[::, 1], mode='markers', showlegend=False,
                                 marker=dict(color="#EF553B")), row=2, col=2)
    fig_sum.add_trace(go.Scatter(x=p(xdata), y=mean_sum.numpy()[::, 1], mode='lines', showlegend=False,
                                 line=dict(color="#00CC96")), row=2, col=2)
    fig_diff.update_layout(title_text="Diff")
    fig_diff.update_xaxes(title_text="Re(\\kappa)", row=1, col=1)
    fig_diff.update_xaxes(title_text="Re(\\kappa)", row=1, col=2)
    fig_diff.update_xaxes(title_text="Im(\\kappa)", row=2, col=1)
    fig_diff.update_xaxes(title_text="Im(\\kappa)", row=2, col=2)
    fig_diff.update_yaxes(title_text="Re(s)", row=1, col=1)
    fig_diff.update_yaxes(title_text="Im(s)", row=1, col=2)
    fig_diff.update_yaxes(title_text="Re(s)", row=2, col=1)
    fig_diff.update_yaxes(title_text="Im(s)", row=2, col=2)
    fig_sum.update_layout(title_text="Sum")
    fig_sum.update_xaxes(title_text="Re(\\kappa)", row=1, col=1)
    fig_sum.update_xaxes(title_text="Re(\\kappa)", row=1, col=2)
    fig_sum.update_xaxes(title_text="Im(\\kappa)", row=2, col=1)
    fig_sum.update_xaxes(title_text="Im(\\kappa)", row=2, col=2)
    fig_diff.update_yaxes(title_text="Re(p)", row=1, col=1)
    fig_diff.update_yaxes(title_text="Im(p)", row=1, col=2)
    fig_diff.update_yaxes(title_text="Re(p)", row=2, col=1)
    fig_diff.update_yaxes(title_text="Im(p)", row=2, col=2)
    fig_diff.show()
    fig_sum.show()


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
