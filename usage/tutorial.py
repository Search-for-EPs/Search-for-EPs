import searchep as sep
import os
import plotly.express as px
import numpy as np


DIRECTORY = os.path.dirname(os.path.realpath(__file__))
INITIAL_DATASET = "Punkt23_initial_dataset_biggest_radius.csv"
EXTRA_TRAININGS_STEP = False
POINT = "Punkt23"
NEW_CALCULATIONS = False
EPS = 1.e-15
EPS_DIFF = 1.e-8  # default
KAPPA_SPACE_FILENAME = "_kappa_space_test_biggest_radius2.html"
DATFILE_KAPPA_EV_TS_NAME = "_data_test_biggest_radius2.csv"
KERNEL_EVS_PICKLE = "_kernel_evs_biggest_radius.pkl"
NEW_INITIAL_DATASET = "Punkt23_new_initial_dataset.csv"


def training_loop():
    training_data = sep.data.Data(INITIAL_DATASET, DIRECTORY, output_name=101)
    sep.training.train(training_data, new_calculations=NEW_CALCULATIONS, plotname="biggest_radius_")
    sep.eval.kappa_space_training_plotly("{}{}".format(POINT, KAPPA_SPACE_FILENAME), training_data)
    sep.eval.data_kappa_ev_ts_file("{}{}".format(POINT, DATFILE_KAPPA_EV_TS_NAME), training_data)
    sep.eval.save_kernel_evs("{}{}".format(POINT, KERNEL_EVS_PICKLE), training_data)


def training_loop_less_data_points():
    training_data = sep.data.Data(INITIAL_DATASET, DIRECTORY, output_name=201)
    training_data.ev = training_data.ev[::4]
    training_data.kappa = training_data.kappa[::4]
    training_data.kappa_scaled = training_data.kappa_scaled[::4]
    training_data.phi = training_data.phi[::4]
    training_data.training_steps_color = training_data.training_steps_color[::4]
    sep.training.train(training_data, new_calculations=NEW_CALCULATIONS, plotname="less_data_points")
    sep.eval.kappa_space_training_plotly("{}{}".format(POINT, KAPPA_SPACE_FILENAME), training_data)
    sep.eval.data_kappa_ev_ts_file("{}{}".format(POINT, DATFILE_KAPPA_EV_TS_NAME), training_data)
    sep.eval.save_kernel_evs("{}{}".format(POINT, KERNEL_EVS_PICKLE), training_data)
    
    
def initial_training_set():
    init_data = sep.data.Data("output_001_1.dat",
                              directory=DIRECTORY, evs_ep=False)
    init_data.ev = sep.data.get_permutations(init_data.vec)
    #init_data.ev = init_data.ev[:, [0, 1]]  # 0 and 1 are the columns of the array and belong to the first permutation. Change numbers to investigate the respective permutation.
    phi_all = np.sort(np.array([init_data.phi.copy() for _ in range(np.shape(init_data.ev)[1])]).ravel())
    fig_all = px.scatter(x=init_data.ev.ravel().real, y=init_data.ev.ravel().imag, color=phi_all,
                         labels=dict(x="Re(\\lambda)", y="Im(\\lambda)"))
    fig_all.show()
    #sep.eval.write_new_dataset(NEW_INITIAL_DATASET, init_data)  # After selecting a permutation this command can be used to save the data to a csv file.


if __name__ == "__main__":
    #training_loop()
    #training_loop_less_data_points()
    initial_training_set()
