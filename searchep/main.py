import numpy as np
from . import plots
from . import gpr
from . import zero_search as zs
from . import GPFlow_model_class as GPFmc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from . import data as dpp
import os
from gpflow.utilities import print_summary

ep = np.array([1.185034 + 1.008482j])
ep_two_close = np.array([0.88278093 + 1.09360073j])

INITIAL_DATASET = "../my_calculations/Punkt23/Punkt23_initial_dataset.csv"
POINT = "Punkt23"
DIRECTORY = "../my_calculations/Punkt23/"
EXTRA_TRAININGS_STEP = False
OUTPUT_NAME = 2

if __name__ == '__main__':
    # plots.init_matplotlib()

    n = False
    i = 0
    eps = 1.e-15
    eps_diff = 1.e-8
    data = dpp.Data(INITIAL_DATASET, DIRECTORY, OUTPUT_NAME)
    # data.kappa_new = np.array([kappa_0])
    # model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(data)
    # model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(data)
    # gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
    # data.kappa = np.concatenate((data.kappa, data.kappa_new))
    # data.ev = dpp.getting_new_ev_of_ep(data, gpflow_model)
    # data.update_scaling()
    while not n:
        model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(data)
        print_summary(model_diff)
        model_sum, kernel_ev_sum = gpr.gp_2d_sum_kappa(data)
        print_summary(model_sum)
        if np.any(abs(kernel_ev_diff) < eps):
            print("MODEL FULLY TRAINED!")
        gpflow_model = GPFmc.GPFlowModel(model_diff, model_sum)
        gpflow_function = gpflow_model.get_model_generator()
        data.kappa_new_scaled = zs.zero_search(gpflow_function, 0. + 0.j)
        data.kappa_scaled = np.concatenate((data.kappa_scaled, data.kappa_new_scaled))
        data.kappa_new = np.array([complex(data.kappa_new_scaled.real * data.kappa_scaling.real + data.kappa_center.real,
                                           data.kappa_new_scaled.imag * data.kappa_scaling.imag + data.kappa_center.imag)])
        data.kappa = np.concatenate((data.kappa, data.kappa_new))
        data.ev = dpp.getting_new_ev_of_ep(data, gpflow_model)
        data.update_scaling()
        i += 1
        data.training_steps_color.append(i)
        print("Root: ", data.kappa_new)
        # print("Real EP: ", ep_two_close)
        ev_diff = data.ev[-1, 0] - data.ev[-1, 1]
        print("Eigenvalue difference: ", ev_diff)
        print("Real part of eigenvalue difference: ", ev_diff.real)
        print("Imaginary part of eigenvalue difference: ", ev_diff.imag)
        if EXTRA_TRAININGS_STEP and i == 2:
            data.kappa_new = np.array([2 * data.kappa[-1] - data.kappa[-2]])
            # print(data.kappa_new)
            data.kappa = np.concatenate((data.kappa, data.kappa_new))  # np.array([complex(data.kappa_new.real, data.kappa_new.imag)])
            # print(data.kappa)
            data.ev = dpp.getting_new_ev_of_ep(data, gpflow_model)
            data.update_scaling()
            data.training_steps_color.append(i)
        if abs(ev_diff.real) < eps_diff and abs(ev_diff.imag) < eps_diff:
            print("Could find EP:")
            print(data.kappa_new)
            # print("Real EP: ")
            # print(ep)
            # print(ep_two_close)
            print("Eigenvalue difference:")
            print(ev_diff)
            n = True
        if i == 25:
            print("Could not find EP yet:")
            print(data.kappa_new)
            # print("Real EP: ")
            # print(ep)
            # print(ep_two_close)
            n = True
    fig1 = px.scatter(x=data.kappa.real, y=data.kappa.imag, color=data.training_steps_color,
                      labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    # fig2 = px.scatter(x=kappa_no_noise.real, y=kappa_no_noise.imag, color=training_steps_color,
    #                  labels=dict(x="Re(kappa)", y="Im(kappa)", color="# of training steps"))
    fig_ep = px.scatter(x=ep_two_close.real, y=ep_two_close.imag, color_discrete_sequence=['#00CC96'], color=["EP"],
                        labels=dict(x="Re(EP)", y="Im(EP)"))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(fig1["data"][0], row=1, col=1)
    # fig.add_trace(fig_ep["data"][0], row=1, col=1)
    fig.update_xaxes(title_text="Re(kappa)", row=1, col=1)
    fig.update_yaxes(title_text="Im(kappa)", row=1, col=1)
    fig.update_layout(coloraxis={'colorbar': dict(title="# of training steps", len=0.9)})
    if EXTRA_TRAININGS_STEP:
        fig.write_html(os.path.join(data.working_directory, "{}_kappa_space_with_extra.html".format(POINT)))
    else:
        fig.write_html(os.path.join(data.working_directory, "{}_kappa_space.html".format(POINT)))
    # fig.show()
    df = pd.DataFrame()
    df['kappa'] = data.kappa.tolist()
    df = pd.concat([df, pd.DataFrame(data.ev)], axis=1)
    df['training_steps_color'] = data.training_steps_color
    df.columns = ['kappa', 'ev1', 'ev2', 'training_steps_color']
    if EXTRA_TRAININGS_STEP:
        df.to_csv(os.path.join(data.working_directory, '{}_data_with_extra.csv'.format(POINT)))
    else:
        df.to_csv(os.path.join(data.working_directory, '{}_data.csv'.format(POINT)))
