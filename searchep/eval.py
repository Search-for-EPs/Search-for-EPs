import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import os
from . import data


def kappa_space_training_plotly(filename, data, show=False):
    fig1 = px.scatter(x=data.kappa.real, y=data.kappa.imag, color=data.training_steps_color,
                      labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(fig1["data"][0], row=1, col=1)
    fig.update_xaxes(title_text="Re(kappa)", row=1, col=1)
    fig.update_yaxes(title_text="Im(kappa)", row=1, col=1)
    fig.update_layout(coloraxis={'colorbar': dict(title="# of training steps", len=0.9)})
    fig.write_html(os.path.join(data.working_directory, filename))
    if show:
        fig.show()


def data_kappa_ev_ts_file(filename, data):
    df = pd.DataFrame()
    df['kappa'] = data.kappa.tolist()
    df = pd.concat([df, pd.DataFrame(data.ev)], axis=1)
    df['training_steps_color'] = data.training_steps_color
    df.columns = ['kappa', 'ev1', 'ev2', 'training_steps_color']
    df.to_csv(os.path.join(data.working_directory, filename))


def eigenvalue_space_plotly(filename):
    kappa, ev, phi = data.load_dat_file(filename)
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev)[1])]).ravel())
    fig = px.scatter(x=ev.ravel().real, y=ev.ravel().imag, color=phi_all,
                     labels=dict(x="Re(\\lambda)", y="Im(\\lambda)", color="Angle"))
    fig.show()
