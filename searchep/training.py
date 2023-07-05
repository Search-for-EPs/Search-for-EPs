import numpy as np
from gpflow.utilities import print_summary
from . import gpr
from . import GPFlow_model_class
from . import data
from . import zero_search


def train(training_data: data.Data, new_calculations: bool = True, extra_training_step: bool = False,
          eps_kernel_ev: float = 1.e-15, eps_diff: float = 2.e-8, info: bool = True, eval_plots: bool = True,
          plotname: str = ""):
    n = False
    current_training_step = training_data.training_steps_color[-1]
    while not n:
        model_diff, kernel_ev_diff = gpr.gp_2d_diff_kappa(training_data)
        model_sum, _ = gpr.gp_2d_sum_kappa(training_data)
        training_data.all_kernel_ev[current_training_step] = kernel_ev_diff
        if info:
            print_summary(model_diff)
            print_summary(model_sum)
        if training_data.exception:
            print("Exception occurred. Current EP:")
            print(training_data.kappa_new)
            print("Eigenvalue difference:")
            if 'ev_diff' in locals():
                print(ev_diff)
            n = True
            break
        if np.any(abs(kernel_ev_diff) < eps_kernel_ev):
            print("MODEL FULLY TRAINED!")
        gpflow_model = GPFlow_model_class.GPFlowModel(model_diff, model_sum)
        gpflow_function = gpflow_model.get_model_generator()
        if new_calculations:
            training_data.kappa_new_scaled = zero_search.zero_search(gpflow_function, 0. + 0.j)
            training_data.kappa_scaled = np.concatenate((training_data.kappa_scaled, training_data.kappa_new_scaled))
            training_data.kappa_new = np.array([complex(training_data.kappa_new_scaled.real *
                                                        training_data.kappa_scaling.real +
                                                        training_data.kappa_center.real,
                                                        training_data.kappa_new_scaled.imag *
                                                        training_data.kappa_scaling.imag +
                                                        training_data.kappa_center.imag)])
            training_data.kappa = np.concatenate((training_data.kappa, training_data.kappa_new))
        training_data.ev = data.getting_new_ev_of_ep(training_data, gpflow_model, new_calculations,
                                                     eval_plots=eval_plots, plot_name=plotname)
        training_data.update_scaling()
        current_training_step += 1
        training_data.training_steps_color.append(current_training_step)
        ev_diff = training_data.ev[-1, 0] - training_data.ev[-1, 1]
        if info:
            print("Root: ", training_data.kappa_new)
            print("Eigenvalue difference: ", ev_diff)
            print("Real part of eigenvalue difference: ", ev_diff.real)
            print("Imaginary part of eigenvalue difference: ", ev_diff.imag)
        if extra_training_step and current_training_step == 2:
            training_data.kappa_new = np.array([2 * training_data.kappa[-1] - training_data.kappa[-2]])
            training_data.kappa = np.concatenate((training_data.kappa, training_data.kappa_new))
            training_data.ev = data.getting_new_ev_of_ep(training_data, gpflow_model)
            training_data.update_scaling()
            training_data.training_steps_color.append(current_training_step)
        if abs(ev_diff.real) < eps_diff and abs(ev_diff.imag) < eps_diff:
            print("Could find EP:")
            print(training_data.kappa_new)
            print("Eigenvalue difference:")
            print(ev_diff)
            n = True
        if current_training_step == 25:
            print("Could not find EP yet:")
            print(training_data.kappa_new)
            n = True
