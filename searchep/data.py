import jax.numpy as jnp
import numpy as np
from jax import vmap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matrix
import subprocess
import os


class Data:

    def __init__(self, filename, directory, output_name):
        df = pd.read_csv(filename, header=0, skiprows=0,
                         names=["kappa", "ev1", "ev2", "phi"])
        self.kappa = np.array(df.kappa).astype(complex)
        self.ev = np.column_stack((np.array(df.ev1).astype(complex), np.array(df.ev2).astype(complex)))
        self.phi = np.array(df.phi)
        # self.kappa, self.ev, self.phi = load_dat_file(filename)
        # fig_all = px.scatter(x=self.ev.ravel().real, y=self.ev.ravel().imag,
        #                     labels=dict(x="Re(\\lambda)", y="Im(\\lambda)"))
        # self.kappa, self.phi = matrix.parametrization(kappa_0, r, steps)
        # symmatrix = matrix.matrix_two_close_im(self.kappa)
        # ev_new = matrix.eigenvalues(symmatrix)
        # self.ev = initial_dataset(self.ev)
        # fig_ev = px.scatter(x=self.ev.ravel().real, y=self.ev.ravel().imag, c="EF553B",
        #                    labels=dict(x="Re(\\lambda)", y="Im(\\lambda)"))
        """df = pd.DataFrame()
        df['kappa'] = self.kappa.tolist()
        df = pd.concat([df, pd.DataFrame(self.ev)], axis=1)
        df['phi'] = self.phi.tolist()
        df.columns = ['kappa', 'ev1', 'ev2', 'phi']"""
        # self.ev = initial_dataset(ev_new)
        # self.ev = self.ev[::11]
        # self.kappa = self.kappa[::11]
        # self.ev = initial_dataset(self.ev)
        # a = int(np.ceil(np.shape(self.kappa)[0]/20))
        """self.ev = self.ev[::2]
        self.kappa = self.kappa[::2]
        self.phi = self.phi[::2]"""
        self.training_steps_color = [0 for _ in self.kappa]

        """fig = make_subplots(rows=1, cols=1)
        fig.add_trace(fig_all["data"][0], row=1, col=1)
        fig.add_trace(fig_ev["data"][0], row=1, col=1)
        fig.show()"""

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
        self.working_directory = os.path.normpath(os.path.join(os.getcwd(), directory))
        # df.to_csv(os.path.join(self.working_directory, 'Punkt29_initial_dataset.csv'))

    def update_scaling(self):
        ev_diff_complex = ((self.ev[::, 0] - self.ev[::, 1]) ** 2)
        self.diff_scale = np.max(abs(np.column_stack((ev_diff_complex.real, ev_diff_complex.imag))))
        ev_sum_complex = (0.5 * (self.ev[::, 0] + self.ev[::, 1]))
        self.sum_mean_complex = complex(np.mean(ev_sum_complex.real), np.mean(ev_sum_complex.imag))
        self.sum_scale = np.max(abs(np.column_stack([ev_sum_complex.real - self.sum_mean_complex.real,
                                                           ev_sum_complex.imag - self.sum_mean_complex.imag])))
        self.kappa_center = complex(0.5 * (np.max(self.kappa.real) + np.min(self.kappa.real)),
                                    0.5 * (np.max(self.kappa.imag) + np.min(self.kappa.imag)))
        return self.diff_scale, self.sum_mean_complex, self.sum_scale


def load_dat_file(filename):
    df = pd.read_csv(filename, sep='\s+', skiprows=7, skip_blank_lines=False,
                     names=["Z1", "Z2", "Z3", "Z4", "Z5", "f", "gamma", "Z8", "Z9", "Z10", "Z11", "ev_re", "ev_im",
                            "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", "Z22", "Z23", "Z24", "Z25", "Z26",
                            "Z27", "Z28", "Z29", "Z30", "phi"])

    f = clump(np.array(df.f))
    gamma = clump(np.array(df.gamma))
    kappa = gamma + f * 1j
    ev_re = clump(np.array(df.ev_re))
    ev_im = clump(np.array(df.ev_im))
    ev = ev_re + ev_im * 1j
    phi = clump(np.array(df.phi))
    return kappa[:, 0], ev, phi[:, 0]


def clump(a):
    return np.array([a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))])


def initial_dataset(ev):
    """Get initial dataset

    Selecting the eigenvalues belonging to the EP by ordering all eigenvalues and comparing the first end last point.
    If it is greater than 0 the eigenvalues exchange their positions and belong to the EP.

    Parameters
    ----------
    ev : np.ndarray
        All exact complex eigenvalues

    Returns
    -------
    np.ndarray
        Usually 2D array containing the two eigenvalues belonging to the EP
    """
    nearest_neighbour = np.array(
        [vmap(lambda x, y: abs(x - y), in_axes=(0, 0), out_axes=0)(ev, jnp.roll(np.roll(ev, -1, axis=0), -i, axis=1))
         for i in range(np.shape(ev)[1])])
    ev_all = []
    for i in range(np.shape(nearest_neighbour)[2]):
        ev_sorted = [ev[0, i]]
        l = i
        for j in range(np.shape(nearest_neighbour)[1]):
            l = (np.argmin(nearest_neighbour[:, j, l]) + l) % np.shape(ev)[1]
            if j + 1 != np.shape(ev)[0]:
                ev_sorted.append(ev[(j + 1), l])
        ev_all.append(ev_sorted)
    ev_all_sorted = np.column_stack([ev_all[k] for k in range(np.shape(ev_all)[0])])
    ep_ev_index = np.argwhere(abs(ev_all_sorted[0, :] - ev_all_sorted[-1, :]) > 3.e-6)
    # print(type(np.array(ev_all_sorted[:, ep_ev_index])))
    return np.column_stack([ev_all_sorted[:, n] for n in ep_ev_index])  # ev_all_sorted[:, ep_ev_index])


def getting_new_ev_of_ep_old(kappa, ev, model_diff, model_sum):
    """Getting new eigenvalues belonging to the EP

    Selecting the two eigenvalues of a new point belonging to the EP by comparing it to a GPR model prediction and
    its variance.

    Parameters
    ----------
    kappa : np.ndarray
        All complex kappa values
    ev : np.ndarray
        Containing all old eigenvalues belonging to the EP
    model_diff : gpflow.models.GPR
        2D GPR model for eigenvalue difference squared
    model_sum : gpflow.models.GPR
        2D GPR model for eigenvalue sum

    Returns
    -------
    np.ndarray
        2D array containing all old and the new eigenvalues belonging to the EP
    """
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    xx, yy = np.meshgrid(kappa.real, kappa.imag)
    grid = np.array((xx.ravel(), yy.ravel())).T
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    pairs_diff_all = np.empty(0)
    pairs_sum_all = np.empty(0)
    ev_1 = np.empty(0)
    ev_2 = np.empty(0)
    for i, val in enumerate(ev_new[0, ::]):
        pairs_diff = vmap(lambda a, b: jnp.power((a - b), 2), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])

        pairs_sum = vmap(lambda a, b: 0.5 * jnp.add(a, b), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        ev_1 = np.concatenate((ev_1, np.array([val for _ in range(len(ev_new[0, (i + 1):]))])))
        ev_2 = np.concatenate((ev_2, ev_new[0, (i + 1):]))
        pairs_diff_all = np.concatenate((pairs_sum_all, np.array(pairs_diff)))
        pairs_sum_all = np.concatenate((pairs_sum_all, np.array(pairs_sum)))
    compatibility = - np.power(pairs_diff_all.real - mean_diff.numpy()[0, 0], 2) / (2 * var_diff.numpy()[0, 0]) \
                    - np.power(pairs_diff_all.imag - mean_diff.numpy()[0, 1], 2) / (2 * var_diff.numpy()[0, 1]) \
                    - np.power(pairs_sum_all.real - mean_sum.numpy()[0, 0], 2) / (2 * var_sum.numpy()[0, 0]) \
                    - np.power(pairs_sum_all.imag - mean_sum.numpy()[0, 1], 2) / (2 * var_sum.numpy()[0, 1])
    c = np.array([0 for _ in compatibility])
    fig1 = px.scatter(x=c, y=abs(compatibility), log_y=True)
    # fig1.show()
    new = np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers', name="All eigenvalues"))
    fig.add_trace(go.Scatter(x=new.ravel().real, y=new.ravel().imag, mode='markers', name="Eigenvalues of EP",
                             marker=dict(color='red')))
    fig.show()
    return np.concatenate((ev, np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])))


def getting_new_ev_of_ep(data, gpflow_model):
    """Getting new eigenvalues belonging to the EP

    Selecting the two eigenvalues of a new point belonging to the EP by comparing it to a GPR model prediction and
    its variance.

    Parameters
    ----------
    data : data_preprocessing.Data
        Class which contains all scale-, kappa- and eigenvalues
    gpflow_model : GPFlow_model_class.GPFlowModel
        Class which contains both 2D GPR models
    # kappa : np.ndarray
    #     All complex kappa values
    # ev : np.ndarray
    #     Containing all old eigenvalues belonging to the EP
    # model_diff : gpflow.models.GPR
    #     2D GPR model for eigenvalue difference squared
    # model_sum : gpflow.models.GPR
    #     2D GPR model for eigenvalue sum

    Returns
    -------
    np.ndarray
        2D array containing all old and the new eigenvalues belonging to the EP
    """
    # symmatrix = matrix.matrix_two_close_im(data.kappa_new)
    # ev_new = matrix.eigenvalues(symmatrix)
    start_exact_calculation(data)
    ev_new = read_new_ev(data)
    grid = np.column_stack((data.kappa_new_scaled.real, data.kappa_new_scaled.imag))
    mean_diff, var_diff = gpflow_model.model_diff.predict_f(grid)
    mean_sum, var_sum = gpflow_model.model_sum.predict_f(grid)
    pairs_diff_all = np.empty(0)
    pairs_sum_all = np.empty(0)
    pairs_difference = np.empty(0)
    ev_1 = np.empty(0)
    ev_2 = np.empty(0)
    for i, val in enumerate(ev_new[0, ::]):
        pairs_diff_squared = vmap(lambda a, b: jnp.power((a - b), 2), in_axes=(None, 0), out_axes=0)(val,
                                                                                                    ev_new[0, (i + 1):])
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
    c = np.array([0 for _ in compatibility])
    fig1 = px.scatter(x=c, y=abs(compatibility), log_y=True)
    fig1.write_html(os.path.join(data.working_directory, "compatibility_%1d.html" % np.shape(data.ev)[0]))
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
    fig.write_html(os.path.join(data.working_directory, "selected_eigenvalues_%1d.html" % np.shape(data.ev)[0]))
    # fig.show()
    # fig_diff = go.Figure()
    # fig_diff.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers', name="All eigenvalues"))
    # fig_diff.add_trace(go.Scatter(x=new_diff.ravel().real, y=new_diff.ravel().imag, mode='markers',
    #                              name="Eigenvalues of EP", marker=dict(color='red')))
    # fig_diff.write_html(os.path.join(data.working_directory, "selected_eigenvalue_diff_%1d.html" % np.shape(data.ev)[0]))
    # fig_diff.show()
    return np.concatenate((data.ev, np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])))


def start_exact_calculation(data):
    inp = "input_%s.inp" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name))
    input_file = os.path.join(data.working_directory, inp)
    out = "out%s.dat" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name))
    gamma_m = str("{:e}".format(data.kappa_new.real[0])).replace("e", "d")
    f_m = str("{:e}".format(data.kappa_new.imag[0])).replace("e", "d")
    with open(input_file, 'w') as f:
        f.write("30                  ! N_max\n" +
                "1                   ! F2min\n" +
                "28                  ! F2max\n" +
                "28                   ! N_max Green\n" +
                "18                  ! F2max Green\n" +
                "2                   ! delta 0: diagonalize with additional read in states 1: write out states\n" +
                "2                   ! parity 0: only even, 1: only odd, 2: even and odd basis states\n" +
                "42                  ! abs(alpha) (for N: N*7.5*2.8 = N*21) \n" +
                "0.14d0              ! arg(alpha_i)\n" +
                "0.14d0              ! arg(alpha_f) \n" +
                "0.02d0              ! arg(alpha_del)\n" +
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
                "70                  ! nev  Number of eigenvalues and eigenvectors to be computed\n" +
                "200                 ! statenr number for file containing additional states for delta diagonalization\n" +
                "%s                 ! ofnr  Punkt 35\n" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name)))
    subprocess.run('cd {0} && ./main < {1} > {2}'.format(data.working_directory, inp, out), shell=True)


def read_new_ev(data):
    out = "output_%s_1.dat" % str((3 - len(str(data.output_name))) * "0" + str(data.output_name))
    output_file = os.path.join(data.working_directory, out)
    kappa, ev, phi = load_dat_file(output_file)
    # data.kappa_new = kappa
    # data.kappa_new_scaled = np.array((data.kappa_new.real - data.kappa_center.real) / data.kappa_scaling.real +
    #                                    ((data.kappa_new.imag - data.kappa_center.imag) / data.kappa_scaling.imag) * 1j)
    # data.kappa = np.concatenate((data.kappa, data.kappa_new))
    # data.kappa_scaled = np.concatenate((data.kappa_scaled, data.kappa_new_scaled))
    data.output_name += 1
    return ev
