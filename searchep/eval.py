"""Quick evaluation of trained models and new calculations"""

import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
import os
from . import data


def kappa_space_training_plotly(filename: str, training_data: data.Data, show: bool = False):
    """Plotly plot of trained kappa space

    Creates a html file in which the plot is saved. Usually used at the end of the training to instantly evaluate the
    results.

    Parameters
    ----------
    filename : str
        Name of the created html file
    training_data : data.Data
        Data class which contains all relevant values
    show : bool, optional
        If True, the plot is displayed
    """
    fig1 = px.scatter(x=training_data.kappa.real, y=training_data.kappa.imag, color=training_data.training_steps_color,
                      labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(fig1["data"][0], row=1, col=1)
    fig.update_xaxes(title_text="Re(kappa)", row=1, col=1)
    fig.update_yaxes(title_text="Im(kappa)", row=1, col=1)
    fig.update_layout(coloraxis={'colorbar': dict(title="# of training steps", len=0.9)})
    fig.write_html(os.path.join(training_data.working_directory, filename))
    if show:
        fig.show()


def data_kappa_ev_ts_file(filename: str, training_data: data.Data):
    """Kappa-, eigenvalues and training steps are written to a csv file

    Only the eigenvalues belonging to the EP are written to the csv. Usually used at the end of training to facilitate
    access to data.

    Parameters
    ----------
    filename : str
        Name of the crated csv file
    training_data : data.Data
        Data class which contains all relevant values
    """
    df = pd.DataFrame()
    df['kappa'] = training_data.kappa.tolist()
    df = pd.concat([df, pd.DataFrame(training_data.ev)], axis=1)
    df['training_steps_color'] = training_data.training_steps_color
    df.columns = ['kappa', 'ev1', 'ev2', 'training_steps_color']
    df.to_csv(os.path.join(training_data.working_directory, filename))


def eigenvalue_space_plotly(filename: str):
    """Plotly plot of the energy / eigenvalue space

    Reads an output file to get the kappa-, eigen-, and phi-values. Plot is used to identify exchange behavior of the
    eigenvalues which belong to an EP.

    Parameters
    ----------
    filename : str
        Name of the output file to be read
    """
    kappa, ev, phi, _, _ = data.load_dat_file(filename)
    phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(ev)[1])]).ravel())
    fig = px.scatter(x=ev.ravel().real, y=ev.ravel().imag, color=phi_all,
                     labels=dict(x="Re(\\lambda)", y="Im(\\lambda)", color="Angle"))
    fig.show()


def write_new_dataset(filename: str, init_data: data.Data):
    """Writes kappa-, eigen- and phi-values to a new csv file

    Usually used after the initial_dataset function to save the obtained eigenvalues belonging to the EP.

    Parameters
    ----------
    filename : str
        Name of the created csv file
    init_data : data.Data
        Data class which contains all relevant values 
    """
    df = pd.DataFrame()
    df['kappa'] = init_data.kappa.tolist()
    df = pd.concat([df, pd.DataFrame(init_data.ev)], axis=1)
    df['phi'] = init_data.phi.tolist()
    df.columns = ['kappa', 'ev1', 'ev2', 'phi']
    df.to_csv(os.path.join(init_data.working_directory, filename))


def save_kernel_evs(filename: str, training_data: data.Data):
    with open(os.path.join(training_data.working_directory, filename), 'wb') as f:
        pickle.dump(training_data.all_kernel_ev, f)
