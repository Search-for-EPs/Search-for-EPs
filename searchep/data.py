import jax.numpy as jnp
import numpy as np
from jax import vmap
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import GPFlow_model_class as GPFmc
import subprocess
import os
from typing import Tuple, Union

parameters = ["N_max", "F2min", "F2max", "delta", "parity",
              "abs(alpha)", "arg(alpha_i)", "arg(alpha_f)", "arg(alpha_del)",
              "nev", "statenr"]


class Data:
    """Data class containing all relevant Data

    Save and update all relevant values during the training process.
    """

    def __init__(self, filename: str, directory: str, output_name: int = 2, evs_ep: bool = True,
                 input_parameters: Union[dict, str] = None, distance: float = 3.e-6):
        """Constructor of the Data class

        Parameters
        ----------
        filename : str
            Name of file  which contains the relevant data
        directory : str
            Absolute path to working directory
        output_name : int, optional
            For new calculations a new file is created and this parameter controls the number of this new input file
        evs_ep : bool, optional
            If the given file contains only the eigenvalues of the EP and the corresponding kappa and phi values this
            parameter should be True and if not it should be False
        input_parameters : Union[dict, str], optional
            The parameters of the input file. Default are listed in User Guide. A filename of the input file which
            contains the parameters or a dictionary which contains the relevant parameters which are different from
            the default ones
        distance : float, optional
            Distance between first and last eigenvalue after one circulation. Needed for function initial_dataset
        """
        self.working_directory = directory
        if evs_ep:
            df = pd.read_csv(os.path.normpath(os.path.join(self.working_directory, filename)), header=0, skiprows=0,
                             names=["kappa", "ev1", "ev2", "phi"])
            self.kappa = np.array(df.kappa).astype(complex)
            self.ev = np.column_stack((np.array(df.ev1).astype(complex), np.array(df.ev2).astype(complex)))
            self.phi = np.array(df.phi)
        else:
            self.kappa, self.ev, self.phi, self.vec, self.vec_normalized = load_dat_file(
                os.path.normpath(os.path.join(self.working_directory, filename)))
            #phi_all = np.sort(np.array([self.phi.copy() for _ in range(np.shape(self.ev)[1])]).ravel())
            #fig_all = px.scatter(x=self.ev.ravel().real, y=self.ev.ravel().imag, color=phi_all,
            #                     labels=dict(x="Re(\\lambda)", y="Im(\\lambda)"))
            # fig_all.show()
            # self.ev_old = initial_dataset_old(self.ev, distance=distance)
            #self.ev = initial_dataset(self.vec, distance=distance) # self.vec_normalized
            #phi_all = np.sort(np.array([self.phi.copy() for _ in range(np.shape(self.ev)[1])]).ravel())
            #fig_ev = px.scatter(x=self.ev.ravel().real, y=self.ev.ravel().imag, color=phi_all,
            #                    labels=dict(x="Re(\\lambda)", y="Im(\\lambda)"))
            # fig_ev.show()
            # fig = make_subplots(rows=1, cols=1)
            # fig.add_trace(fig_all["data"][0], row=1, col=1, )
            # fig.add_trace(fig_ev["data"][0], row=1, col=1)
            # fig.show()
        if input_parameters:
            try:
                if type(input_parameters) == dict:
                    self.input_parameters = input_parameters
                elif type(input_parameters) == str:
                    self.input_parameters = dict()
                    with open(os.path.normpath(os.path.join(self.working_directory, input_parameters)),
                              encoding='utf-8') as f:
                        for line in f.readlines():
                            parameter = set(parameters).intersection(line.split())
                            if parameter:
                                if len(parameter) > 1:
                                    self.input_parameters[str(parameters[-1])] = line.split()[0]
                                elif "Green" in line.split():
                                    self.input_parameters[str(list(parameter)[0] + "_Green")] = line.split()[0]
                                else:
                                    self.input_parameters[str(list(parameter)[0])] = line.split()[0]
                else:
                    raise TypeError
            except TypeError as e:
                print("The optional argument \"input_parameters\" should be of type dict or string. Using default "
                      "input parameters for new calculations.")
                print(e)
                self.input_parameters = dict()
            except OSError as e:
                print("The optional argument \"input_parameters\" is specified as string but the file can not be found."
                      " Using default input parameters for new calculations.")
                print(e)
                self.input_parameters = dict()
            except Exception as e:
                print("Exception occurred for input parameters (see below). Using default input parameters for new "
                      "calculations.")
                print(e)
                self.input_parameters = dict()
        else:
            print("Using default input parameters for new calculations.")
            self.input_parameters = dict()

        """df = pd.DataFrame()
        df['kappa'] = self.kappa.tolist()
        df = pd.concat([df, pd.DataFrame(self.ev)], axis=1)
        df['phi'] = self.phi.tolist()
        df.columns = ['kappa', 'ev1', 'ev2', 'phi']"""
        self.all_kernel_ev = dict()
        self.training_steps_color = [0 for _ in self.kappa]

        self.diff_scale, self.sum_mean_complex, self.sum_scale = self.update_scaling()
        self.kappa_center = complex(0.5 * (np.max(self.kappa.real) + np.min(self.kappa.real)),
                                    0.5 * (np.max(self.kappa.imag) + np.min(self.kappa.imag)))
        self.kappa_scaling = complex(np.max(self.kappa.real) - self.kappa_center.real,
                                     np.max(self.kappa.imag) - self.kappa_center.imag)
        self.kappa_scaled = np.array((self.kappa.real - self.kappa_center.real) / self.kappa_scaling.real +
                                     ((self.kappa.imag - self.kappa_center.imag) / self.kappa_scaling.imag) * 1j)
        self.kappa_new_scaled = np.empty(0)
        self.kappa_new = np.empty(0)
        self.output_name = output_name
        self.exception = False
        # df.to_csv(os.path.join(self.working_directory, 'Punkt29_initial_dataset.csv'))

    def update_scaling(self):
        ev_diff_complex = ((self.ev[::, 0] - self.ev[::, 1]) ** 2)
        self.diff_scale = np.max(abs(np.column_stack((ev_diff_complex.real, ev_diff_complex.imag))))
        ev_sum_complex = (0.5 * (self.ev[::, 0] + self.ev[::, 1]))
        self.sum_mean_complex = complex(np.float64(np.mean(ev_sum_complex.real)),
                                        np.float64(np.mean(ev_sum_complex.imag)))
        self.sum_scale = np.max(abs(np.column_stack([ev_sum_complex.real - self.sum_mean_complex.real,
                                                     ev_sum_complex.imag - self.sum_mean_complex.imag])))
        return self.diff_scale, self.sum_mean_complex, self.sum_scale


def load_dat_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dat file (output of external program)

    Data file containing all kappa values, angles and respective eigenvalues.

    Parameters
    ----------
    filename : str
        Absolute path to dat file

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        First return is the kappa array which contains every kappa value only once. Second return is an array 
        containing all eigenvalues where each row belongs to one kappa value. Third return is the phi array
        similar to the kappa array.
    """
    df = pd.read_csv(filename, sep='\s+', skiprows=7, skip_blank_lines=False,
                     names=["Z1", "Z2", "Z3", "Z4", "Z5", "f", "gamma", "Z8", "Z9", "Z10", "Z11", "ev_re", "ev_im",
                            "Z14", "oscR_x_re", "oscR_x_im", "oscR_y_re", "oscR_y_im", "Z19", "Z20", "oscL_x_re",
                            "oscL_x_im", "oscL_y_re", "oscL_y_im", "Z25", "Z26", "yellow_re", "yellow_im", "green_re",
                            "green_im", "s_re", "s_im", "p_re", "p_im", "d_re", "d_im", "f_re", "f_im", "g_re", "g_im",
                            "h_re", "h_im", "i_re", "i_im", "j_re", "j_im", "k_re", "k_im", "l_re", "l_im", "phi"])

    f = clump(np.array(df.f))
    gamma = clump(np.array(df.gamma))
    kappa = gamma + f * 1j
    ev_re = clump(np.array(df.ev_re))
    ev_im = clump(np.array(df.ev_im))
    ev = ev_re + ev_im * 1j
    oscR_x_re = clump(np.array(df.oscR_x_re))
    oscR_x_im = clump(np.array(df.oscR_x_im))
    oscR_y_re = clump(np.array(df.oscR_y_re))
    oscR_y_im = clump(np.array(df.oscR_y_im))
    oscL_x_re = clump(np.array(df.oscL_x_re))
    oscL_x_im = clump(np.array(df.oscL_x_im))
    oscL_y_re = clump(np.array(df.oscL_y_re))
    oscL_y_im = clump(np.array(df.oscL_y_im))
    yellow_re = clump(np.array(df.yellow_re))
    yellow_im = clump(np.array(df.yellow_im))
    green_re = clump(np.array(df.green_re))
    green_im = clump(np.array(df.green_im))
    s_re = clump(np.array(df.s_re))
    s_im = clump(np.array(df.s_im))
    p_re = clump(np.array(df.p_re))
    p_im = clump(np.array(df.p_im))
    d_re = clump(np.array(df.d_re))
    d_im = clump(np.array(df.d_im))
    f_re = clump(np.array(df.f_re))
    f_im = clump(np.array(df.f_im))
    g_re = clump(np.array(df.g_re))
    g_im = clump(np.array(df.g_im))
    h_re = clump(np.array(df.h_re))
    h_im = clump(np.array(df.h_im))
    i_re = clump(np.array(df.i_re))
    i_im = clump(np.array(df.i_im))
    j_re = clump(np.array(df.j_re))
    j_im = clump(np.array(df.j_im))
    k_re = clump(np.array(df.k_re))
    k_im = clump(np.array(df.k_im))
    l_re = clump(np.array(df.l_re))
    l_im = clump(np.array(df.l_im))
    phi = clump(np.array(df.phi))
    oscR_x = oscR_x_re + oscR_x_im * 1j
    oscL_x = oscL_x_re + oscL_x_im * 1j
    osc_x = oscR_x * oscL_x
    oscR_y = oscR_y_re + oscR_y_im * 1j
    oscL_y = oscL_y_re + oscL_y_im * 1j
    osc_y = oscR_y * oscL_y
    s = s_re + s_im * 1j
    p = p_re + p_im * 1j
    d = d_re + d_im * 1j
    f = f_re + f_im * 1j
    g = g_re + g_im * 1j
    h = h_re + h_im * 1j
    i = i_re + i_im * 1j
    j = j_re + j_im * 1j
    k = k_re + k_im * 1j
    l = l_re + l_im * 1j
    fig = px.scatter(x=s.real.ravel(), y=s.imag.ravel(), color=phi.ravel(),
                     labels=dict(x="Re(s)", y="Im(s)", color="phi"))
    # fig.show()
    fig = px.scatter(x=p.real.ravel(), y=p.imag.ravel(), color=phi.ravel(),
                     labels=dict(x="Re(p)", y="Im(p)", color="phi"))
    # fig.show()
    names = [("oscR_x_re", "oscR_x_im"), ("oscR_y_re", "oscR_y_im"), ("oscL_x_re", "oscL_x_im"), ("oscL_y_re",
                    "oscL_y_im"), ("yellow_re", "yellow_im"), ("green_re", "green_im")]
    for index, i in enumerate([(oscR_x_re, oscR_x_im), (oscR_y_re, oscR_y_im), (oscL_x_re, oscL_x_im), (oscL_y_re, oscL_y_im),
              (yellow_re, yellow_im), (green_re, green_im)]):
        # phi_all = np.sort(np.array([phi.copy() for _ in range(np.shape(i[0])[1])]).ravel())
        fig_all = px.scatter(x=i[0][:, 40], y=i[1][:, 40], color=np.sort(phi[:, 40]),
                             labels=dict(x="%s" % str(names[index][0]), y="%s" % str(names[index][1])))
        # fig_all.show()
    vec = np.stack((ev_re, ev_im, #osc_x.real, osc_x.imag, osc_y.real, osc_y.imag,
                    #oscR_x_re, oscR_x_im, oscR_y_re, oscR_y_im, oscL_x_re, oscL_x_im, oscL_y_re, oscL_y_im,
                    #yellow_re, yellow_im, green_re, green_im,
                    s_re, s_im, p_re, p_im, d_re, d_im, f_re, f_im, g_re, g_im,
                    h_re, h_im, i_re, i_im, j_re, j_im, k_re, k_im, l_re, l_im
                    ), axis=2)
    vec_normalized = np.stack(((ev_re - np.min(ev_re)) / (np.max(ev_re) - np.min(ev_re)),
                               (ev_im - np.min(ev_im)) / (np.max(ev_im) - np.min(ev_im)),
                               #(oscR_x_re - np.min(oscR_x_re)) / (np.max(oscR_x_re) - np.min(oscR_x_re)),
                               #(oscR_x_im - np.min(oscR_x_im)) / (np.max(oscR_x_im) - np.min(oscR_x_im)),
                               #(oscR_y_re - np.min(oscR_y_re)) / (np.max(oscR_y_re) - np.min(oscR_y_re)),
                               #(oscR_y_im - np.min(oscR_y_im)) / (np.max(oscR_y_im) - np.min(oscR_y_im)),
                               #(oscL_x_re - np.min(oscL_x_re)) / (np.max(oscL_x_re) - np.min(oscL_x_re)),
                               #(oscL_x_im - np.min(oscL_x_im)) / (np.max(oscL_x_im) - np.min(oscL_x_im)),
                               #(oscL_y_re - np.min(oscL_y_re)) / (np.max(oscL_y_re) - np.min(oscL_y_re)),
                               #(oscL_y_im - np.min(oscL_y_im)) / (np.max(oscL_y_im) - np.min(oscL_y_im)),
                               #(osc_x.real - np.min(osc_x.real)) / (np.max(osc_x.real) - np.min(osc_x.real)),
                               #(osc_x.imag - np.min(osc_x.imag)) / (np.max(osc_x.imag) - np.min(osc_x.imag)),
                               #(osc_y.real - np.min(osc_y.real)) / (np.max(osc_y.real) - np.min(osc_y.real)),
                               #(osc_y.imag - np.min(osc_y.imag)) / (np.max(osc_y.imag) - np.min(osc_y.imag)),
                               #(yellow_re - np.min(yellow_re)) / (np.max(yellow_re) - np.min(yellow_re)),
                               #(yellow_im - np.min(yellow_im)) / (np.max(yellow_im) - np.min(yellow_im)),
                               #(green_re - np.min(green_re)) / (np.max(green_re) - np.min(green_re)),
                               #(green_im - np.min(green_im)) / (np.max(green_im) - np.min(green_im)),
                               #(s_re - np.min(s_re)) / (np.max(s_re) - np.min(s_re)),
                               #(s_im - np.min(s_im)) / (np.max(s_im) - np.min(s_im)),
                               #(p_re - np.min(p_re)) / (np.max(p_re) - np.min(p_re)),
                               #(p_im - np.min(p_im)) / (np.max(p_im) - np.min(p_im)),
                               #(d_re - np.min(d_re)) / (np.max(d_re) - np.min(d_re)),
                               #(d_im - np.min(d_im)) / (np.max(d_im) - np.min(d_im)),
                               #(f_re - np.min(f_re)) / (np.max(f_re) - np.min(f_re)),
                               #(f_im - np.min(f_im)) / (np.max(f_im) - np.min(f_im)),
                               #(g_re - np.min(g_re)) / (np.max(g_re) - np.min(g_re)),
                               #(g_im - np.min(g_im)) / (np.max(g_im) - np.min(g_im)),
                               #(h_re - np.min(h_re)) / (np.max(h_re) - np.min(h_re)),
                               #(h_im - np.min(h_im)) / (np.max(h_im) - np.min(h_im)),
                               #(i_re - np.min(i_re)) / (np.max(i_re) - np.min(i_re)),
                               #(i_im - np.min(i_im)) / (np.max(i_im) - np.min(i_im)),
                               #(j_re - np.min(j_re)) / (np.max(j_re) - np.min(j_re)),
                               #(j_im - np.min(j_im)) / (np.max(j_im) - np.min(j_im)),
                               #(k_re - np.min(k_re)) / (np.max(k_re) - np.min(k_re)),
                               #(k_im - np.min(k_im)) / (np.max(k_im) - np.min(k_im)),
                               #(l_re - np.min(l_re)) / (np.max(l_re) - np.min(l_re)),
                               #(l_im - np.min(l_im)) / (np.max(l_im) - np.min(l_im)),
                               ), axis=2)
    return kappa[:, 0], ev, phi[:, 0], vec, vec_normalized


def clump(a):
    return np.array([a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))])


def stepwise_grouping(vec: np.ndarray, vec_normalized: np.ndarray = None) -> np.ndarray:
    """Stepwise grouping algorithm

    Stepwise grouping of array with respect to the shortest distance between each step and avoiding multiple selection.
    For more details see master thesis of Patrick Egenlauf, 2023.

    Parameters
    ----------
    vec : np.ndarray
        Array, which should be grouped. Two- or three-dimensional array, depending on the number of variables for which
        the pairwise distance is calculated.
    vec_normalized : np.ndarray, optional
        Array with normalized entries or even more variable than vec, which is used to group vec. If not specified, vec
        is used for grouping.
    Returns
    -------
    np.ndarray
        Grouped array vec
    """
    vec_normalized = vec_normalized if vec_normalized else vec
    vec_grouped = [vec[0]]
    vec_grouped_normalized = [vec_normalized[0]]
    for angle in range(vec.shape[0]-1):
        current_angle = vec_grouped_normalized[angle]
        next_angle = vec_normalized[angle+1]
        nn_distance = pairwise_distances(next_angle, current_angle, metric='cosine')  # , metric='cosine'
        nn_index = np.argsort(nn_distance, axis=0)
        while len(nn_index[0]) > len(set(nn_index[0])):
            u, inverse, counts = np.unique(nn_index[0], return_inverse=True, return_counts=True)
            index_multi_min = u[np.argmax(counts)]
            unique_min = current_angle[np.where(inverse == np.argmax(counts))] - next_angle[index_multi_min]
            index_current_angle = np.where(inverse == np.argmax(counts))[0][
                np.argmin(np.linalg.norm(unique_min, axis=1))]
            nn_index = np.column_stack(
                [np.append(nn_index[:, i][(nn_index[:, i] != index_multi_min)], [index_multi_min])
                 if i != index_current_angle else nn_index[:, index_current_angle] for i in range(nn_index.shape[1])])
        vec_grouped.append(vec[angle+1][nn_index[0]])
        vec_grouped_normalized.append(vec_normalized[angle+1][nn_index[0]])
    vec_grouped = np.array(vec_grouped)
    return vec_grouped


def get_permutations(vec: np.ndarray, vec_normalized: np.ndarray = None, distance: float = 3.e-6):
    """Get permutations

    Returns all eigenvalues which perform a permutation by calculating the difference between the first and last entry.
    Utilizes the stepwise grouping algorithm.

    Parameters
    ----------
    vec : np.ndarray
        Array, which should be grouped. Two- or three-dimensional array, depending on the number of variables for which
        the pairwise distance is calculated.
    vec_normalized : np.ndarray, optional
        Array with normalized entries or even more variable than vec, which is used to group vec. If not specified, vec
        is used for grouping.
    distance : float, optional
        Distance between start and end of a loop, by default 3.e-6.

    Returns
    -------
    np.ndarray
        Usually 2D array containing the all eigenvalues performing a permutation
    """
    vec_grouped = stepwise_grouping(vec, vec_normalized=vec_normalized)
    ev_all_grouped = np.column_stack([vec_grouped[:, k, 0] + vec_grouped[:, k, 1] * 1j for k in
                                     range(np.shape(vec_grouped)[1])])
    ep_ev_index = np.argwhere(abs(ev_all_grouped[0, :] - ev_all_grouped[-1, :]) > distance)
    return np.column_stack([ev_all_grouped[:, n] for n in ep_ev_index])



def initial_dataset_old(vec: np.ndarray, vec_normalized: np.ndarray, distance: float = 3.e-6) -> np.ndarray:
    """Get initial dataset

    Selecting the eigenvalues belonging to the EP by ordering all eigenvalues and comparing the first end last point.
    If it is greater than 0 the eigenvalues exchange their positions and belong to the EP.        

    Parameters
    ----------
    vec
    vec_normalized
    ev : np.ndarray
        All exact complex eigenvalues
    distance : float, optional
        Distance between start and end of a loop, by default 3.e-6

    Returns
    -------
    np.ndarray
        Usually 2D array containing the two eigenvalues belonging to the EP
    """
    nearest_neighbour = np.array(
        [vmap(lambda x, y: (x - y), in_axes=(0, 0), out_axes=0)(vec,
                                                                jnp.roll(np.roll(vec, -1, axis=0),
                                                                         -i, axis=1))
         for i in range(np.shape(vec_normalized)[1])])
    norm = 1 / np.linalg.norm(nearest_neighbour, axis=3)
    print("norm dim: ", norm.shape)
    vec_all = []
    for i in range(np.shape(norm)[2]):
        ev_grouped = [vec[0, i]]
        l = i
        for j in range(np.shape(norm)[1]):
            l = (np.argmax(norm[:, j, l]) + l) % np.shape(vec_normalized)[1]
            if j + 1 != np.shape(vec_normalized)[0]:
                ev_grouped.append(vec[(j + 1), l])
        vec_all.append(ev_grouped)
    vec_all = np.array(vec_all)
    ev_all_grouped = np.column_stack([vec_all[k, :, 0] + vec_all[k, :, 1] * 1j for k in range(np.shape(vec_all)[0])])
    # print(ev_all_grouped)
    ep_ev_index = np.argwhere(abs(ev_all_grouped[0, :] - ev_all_grouped[-1, :]) > distance)
    return np.column_stack([ev_all_grouped[:, n] for n in ep_ev_index])  # ev_all_grouped[:, ep_ev_index])


def initial_dataset_older(ev, distance: float = 3.e-6):
    nearest_neighbour = np.array(
        [vmap(lambda x, y: abs(x - y), in_axes=(0, 0), out_axes=0)(ev, jnp.roll(np.roll(ev, -1, axis=0), -i, axis=1))
         for i in range(np.shape(ev)[1])])
    ev_all = []
    for i in range(np.shape(nearest_neighbour)[2]):
        ev_grouped = [ev[0, i]]
        l = i
        for j in range(np.shape(nearest_neighbour)[1]):
            l = (np.argmin(nearest_neighbour[:, j, l]) + l) % np.shape(ev)[1]
            if j + 1 != np.shape(ev)[0]:
                ev_grouped.append(ev[(j + 1), l])
        ev_all.append(ev_grouped)
    ev_all_grouped = np.column_stack([ev_all[k] for k in range(np.shape(ev_all)[0])])
    ep_ev_index = np.argwhere(abs(ev_all_grouped[0, :] - ev_all_grouped[-1, :]) > distance)
    # print(type(np.array(ev_all_grouped[:, ep_ev_index])))
    return np.column_stack([ev_all_grouped[:, n] for n in ep_ev_index])  # ev_all_grouped[:, ep_ev_index])


def getting_new_ev_of_ep(data: Data, gpflow_model: GPFmc.GPFlowModel, new_calculations: bool = True,
                         eval_plots: bool = True, plot_name: str = "") -> np.ndarray:
    """Getting new eigenvalues belonging to the EP

    Selecting the two eigenvalues of a new point belonging to the EP by comparing it to a GPR model prediction and
    its variance.

    Parameters
    ----------
    data : data.Data
        Class which contains all scale-, kappa- and eigenvalues
    gpflow_model : GPFlow_model_class.GPFlowModel
        Class which contains both 2D GPR models
    new_calculations : bool, optional
        Controls if it is a new calculation and if not the kappa value needs to be read as well and 
        no extra calculation of the eigenvalues has to be started, by default True
    eval_plots : bool, optional
        Controls if compatibility and selected eigenvalues should be plotted or not, by default True
    plot_name : str, optional
        Specifies special name for plot files, by default ""

    Returns
    -------
    np.ndarray
        2D array containing all old and the new eigenvalues belonging to the EP
    """
    if new_calculations:
        start_exact_calculation(data)
    ev_new = read_new_ev(data, new_calculations)
    grid = np.column_stack((data.kappa_new_scaled.real, data.kappa_new_scaled.imag))
    mean_diff, var_diff = gpflow_model.model_diff.predict_f(grid)
    mean_sum, var_sum = gpflow_model.model_sum.predict_f(grid)
    pairs_diff_all = np.empty(0)
    pairs_sum_all = np.empty(0)
    pairs_difference = np.empty(0)
    ev_1 = np.empty(0)
    ev_2 = np.empty(0)
    for i, val in enumerate(ev_new[0, ::]):
        pairs_diff_squared = vmap(lambda a, b: jnp.power((a - b), 2),
                                  in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        pairs_diff_squared = pairs_diff_squared / data.diff_scale
        pairs_sum = vmap(lambda a, b: 0.5 * jnp.add(a, b), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        ev_sum_real = (pairs_sum.real - data.sum_mean_complex.real)
        ev_sum_imag = (pairs_sum.imag - data.sum_mean_complex.imag)
        ev_sum = ev_sum_real + ev_sum_imag * 1j
        # ev_sum = np.column_stack([ev_sum_complex.real, ev_sum_complex.imag])
        pairs_sum = ev_sum / data.sum_scale
        pairs_diff = vmap(lambda a, b: abs(a - b), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        ev_1 = np.concatenate((ev_1, np.array([ev_new[0, i] for _ in range(len(ev_new[0, (i + 1):]))])))
        ev_2 = np.concatenate((ev_2, ev_new[0, (i + 1):]))
        pairs_diff_all = np.concatenate((pairs_diff_all, np.array(pairs_diff_squared)))
        pairs_sum_all = np.concatenate((pairs_sum_all, np.array(pairs_sum)))
        pairs_difference = np.concatenate((pairs_difference, np.array(pairs_diff)))
    compatibility = - np.power(pairs_diff_all.real - mean_diff.numpy()[0, 0], 2) / (2 * var_diff.numpy()[0, 0]) \
                    - np.power(pairs_diff_all.imag - mean_diff.numpy()[0, 1], 2) / (2 * var_diff.numpy()[0, 1]) \
                    - np.power(pairs_sum_all.real - mean_sum.numpy()[0, 0], 2) / (2 * var_sum.numpy()[0, 0]) \
                    - np.power(pairs_sum_all.imag - mean_sum.numpy()[0, 1], 2) / (2 * var_sum.numpy()[0, 1])
    if eval_plots:
        c = np.array([0 for _ in compatibility])
        fig1 = px.scatter(x=c, y=abs(compatibility), log_y=True)
        fig1.write_html(
            os.path.join(data.working_directory, "compatibility_%s%1d.html" % (plot_name, np.shape(data.ev)[0])))
        # fig1.show()
        # a = np.argsort(compatibility)
        # unique_filename = str(uuid.uuid4())
        # df = pd.DataFrame()
        # df['compatibility'] = compatibility.tolist()
        # df.to_csv(unique_filename + '.csv')
        new = np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])
        # new2 = np.array([[ev_1[a[-2]], ev_2[a[-2]]]])
        # new3 = np.array([[ev_1[a[-3]], ev_2[a[-3]]]])
        # new_diff = np.array([[ev_1[np.argmin(pairs_difference)], ev_2[np.argmin(pairs_difference)]]])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers', name="All eigenvalues"))
        fig.add_trace(go.Scatter(x=data.ev.ravel().real, y=data.ev.ravel().imag, mode='markers', name='EP eigenvalues',
                                 marker=dict(color='green')))
        fig.add_trace(go.Scatter(x=new.ravel().real, y=new.ravel().imag, mode='markers', name="Eigenvalues of EP",
                                 marker=dict(color='red')))
        # fig.add_trace(go.Scatter(x=new2.ravel().real, y=new2.ravel().imag, mode='markers', name="Eigenvalues of EP2",
        #                         marker=dict(color='red')))
        # fig.add_trace(go.Scatter(x=new3.ravel().real, y=new3.ravel().imag, mode='markers', name="Eigenvalues of EP3",
        #                         marker=dict(color='red')))
        fig.write_html(
            os.path.join(data.working_directory, "selected_eigenvalues_%s%1d.html" % (plot_name, np.shape(data.ev)[0])))
        # fig.show()
        # fig_diff = go.Figure()
        # fig_diff.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag,
        #                              mode='markers', name="All eigenvalues"))
        # fig_diff.add_trace(go.Scatter(x=new_diff.ravel().real, y=new_diff.ravel().imag, mode='markers',
        #                              name="Eigenvalues of EP", marker=dict(color='red')))
        # fig_diff.write_html(os.path.join(data.working_directory,
        #                                 "selected_eigenvalue_diff_%1d.html" % (plot_name, np.shape(data.ev)[0])))
        # fig_diff.show()
    return np.concatenate((data.ev, np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])))


def start_exact_calculation(data: Data):
    """Star new exact calculation of the eigenvalues

    Parameters
    ----------
    data : Data
        Class which contains all scale-, kappa- and eigenvalues
    """
    inp = "input_%s.inp" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name))
    input_file = os.path.join(data.working_directory, inp)
    out = "out%s.dat" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name))
    gamma_m = str("{:e}".format(data.kappa_new.real[0])).replace("e", "d")
    f_m = str("{:e}".format(data.kappa_new.imag[0])).replace("e", "d")
    with open(input_file, 'w') as f:
        f.write("%s                  ! N_max\n" % data.input_parameters.get("N_max", "30") +
                "%s                   ! F2min\n" % data.input_parameters.get("F2min", "1") +
                "%s                  ! F2max\n" % data.input_parameters.get("F2max", "28") +
                "%s                   ! N_max Green\n" % data.input_parameters.get("N_max_Green", "28") +
                "%s                  ! F2max Green\n" % data.input_parameters.get("F2max_Green", "18") +
                "%s                   ! delta 0: diagonalize with additional read in states 1: write out states\n"
                % data.input_parameters.get("delta", "2") +
                "%s                   ! parity 0: only even, 1: only odd, 2: even and odd basis states\n"
                % data.input_parameters.get("parity", "2") +
                "%s                  ! abs(alpha) (for N: N*7.5*2.8 = N*21) \n"
                % data.input_parameters.get("abs(alpha)", "42") +
                "%s              ! arg(alpha_i)\n" % data.input_parameters.get("arg(alpha_i)", "0.14d0") +
                "%s              ! arg(alpha_f) \n" % data.input_parameters.get("arg(alpha_f)", "0.14d0") +
                "%s              ! arg(alpha_del)\n" % data.input_parameters.get("arg(alpha_del)", "0.02d0") +
                "%s            ! f_m  => Mittelpunkt des Kreises (f-Koordinate)\n" % str(f_m) +
                "0                 ! phi_f  => Endwinkel in Grad               \n" +
                "0                ! r  => Radius als Anteil an der Feldstärke   \n" +
                "%s                ! gamma_m  => Mittelpunkt des Kreises (gamma-Koordinate)\n" % str(gamma_m) +
                "0               ! phi_i  => Startwinkel in Grad           Magnetic field in T\n" +
                "0                 ! n_phi => Anzahl Winkel \n" +
                "1.0d0               ! s_i               \n" +
                "1.0d0               ! s_f                Strength of the band structure\n" +
                "0.02d0              ! s_del\n" +
                "2.00               ! e_real Energie position in the complex plane, around\n" +
                "0.0                 ! e_imag which the eigenvalues are searched, in eV\n" +
                "%s                  ! nev  Number of eigenvalues and eigenvectors to be computed\n"
                % data.input_parameters.get("nev", "70") +
                "%s                 ! statenr number for file containing additional states for delta diagonalization\n"
                % data.input_parameters.get("statenr", "200") +
                "%s                 ! ofnr\n" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name)))
    subprocess.run('cd {0} && ./main < {1} > {2}'.format(data.working_directory, inp, out), shell=True)


def read_new_ev(data: Data, new_calculations=True) -> np.ndarray:
    """Reads the eigenvalues of a new diagonalization

    Parameters
    ----------
    data : Data
        Class which contains all scale-, kappa- and eigenvalues
    new_calculations : bool, optional
        Controls if it is a new calculation and if not the kappa value needs to be read as well, by default True

    Returns
    -------
    np.ndarray
        All eigenvalues of the new diagonalization
    """
    out = "output_%s_1.dat" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name))
    output_file = os.path.join(data.working_directory, out)
    kappa, ev, phi, vec, vec_normalized = load_dat_file(output_file)
    if not new_calculations:
        data.kappa_new = kappa
        data.kappa_new_scaled = np.array((data.kappa_new.real - data.kappa_center.real) / data.kappa_scaling.real +
                                         ((data.kappa_new.imag - data.kappa_center.imag) / data.kappa_scaling.imag)
                                         * 1j)
        data.kappa = np.concatenate((data.kappa, data.kappa_new))
        data.kappa_scaled = np.concatenate((data.kappa_scaled, data.kappa_new_scaled))
    data.output_name += 1
    return ev
