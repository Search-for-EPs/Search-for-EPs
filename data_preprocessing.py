import jax.numpy as np
import numpy
from jax import vmap
import plotly.express as px
import plotly.graph_objects as go
import matrix


def getting_new_ev_of_ep(kappa, ev, model_diff, model_sum):
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev_new = matrix.eigenvalues(symmatrix)
    xx, yy = numpy.meshgrid(kappa.real, kappa.imag)
    grid = numpy.array((xx.ravel(), yy.ravel())).T
    mean_diff, var_diff = model_diff.predict_f(grid)
    mean_sum, var_sum = model_sum.predict_f(grid)
    pairs_diff_all = np.empty(0)
    pairs_sum_all = np.empty(0)
    ev_1 = np.empty(0)
    ev_2 = np.empty(0)
    for i, val in enumerate(ev_new[0, ::]):
        pairs_diff = vmap(lambda a, b: np.power((a - b), 2), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
        pairs_sum = vmap(lambda a, b: 0.5 * np.add(a, b), in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers', name="All eigenvalues"))
    # fig.add_trace(go.Scatter(x=x, y=mean_re.numpy().ravel(), mode='makers', name="Eigenvalues of EP",
    #                         marker=dict(color='red')))
    # fig.show()
    return numpy.concatenate((ev, np.array([[ev_1[np.argmax(compatibility)], ev_2[np.argmax(compatibility)]]])))
