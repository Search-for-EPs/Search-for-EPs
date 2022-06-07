import numpy as np
import matrix
import plots
import time
import gaussian_process_regression as gpr
from scipy import optimize
import zerosearch as zs
import matplotlib.pyplot as plt
import GPFlow_model_class as GPFmc


if __name__ == '__main__':
    # plots.init_matplotlib()

    kappa_0 = 0.5 + 0.5j
    r = 1
    steps = 100
    kappa, phi = matrix.parametrization(kappa_0, r, steps)
    # start1 = time.time()
    symmatrix = matrix.matrix_random(2, kappa)
    # end1 = time.time()
    # print("Matrix created in ", end1-start1)
    # start2 = time.time()
    ev = matrix.eigenvalues(symmatrix)
    # print(ev)
    # end2 = time.time()
    # print("EV calculated in ", end2-start2)
    ev_training = ev[0::11]
    phi_training = phi[0::11]
    kappa_training = kappa[0::11]
    # start3 = time.time()
    m_re, m_im = gpr.gp_diff_kappa(ev, kappa)
    # end3 = time.time()
    # print("Model optimized in ", end3-start3)
    gpflow_model = GPFmc.GPFlowModel(m_re, m_im)
    gpflow_function = gpflow_model.get_model_generator()
    sol = zs.zero_search(gpflow_function, kappa_0)
    print(sol.x)
    # plots.three_d_eigenvalue_kappa_plotly(kappa_0, r, m_re, m_im)
    # plots.parameter_plane_plotly(kappa, phi)
    # plots.energy_plane_plotly(ev, phi)
    # plots.eigenvalues_angle_plotly(ev, phi, m_re, m_im)
    # plots.eigenvalues_angle_matplotlib(ev, phi, m_re, m_im)
    # plt.show()
