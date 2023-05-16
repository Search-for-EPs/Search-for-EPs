import jax.numpy as jnp
from jax import vmap
import numpy as np
from . import matrix
import gpflow


class GPFlowModel:
    """
    Class for the deployment of a GPFlow
    model.
    """

    def __init__(
            self,
            model_diff: gpflow.models.GPModel,
            model_sum: gpflow.models.GPModel
    ):
        """
        Constructor for the model class.

        Parameters
        ----------
        model_diff : gpflow.models.GPFlow
                Optimized model from GPFlow for eigenvalue difference squared
        model_sum : gpflow.models.GPFlow
                Optimized model from GPFlow for eigenvalue sum
        """
        self.model_sum = model_sum
        self.model_diff = model_diff

    def get_model_generator(self):
        """
        Get the model generator for a specific model.
        """

        def _model_generator(x_data):
            """
            Function to compute values at x_data

            Parameters
            ----------
            x_data : np.ndarray
                    x_data for which you want a new function value.
            """
            xx, yy = np.meshgrid(x_data[0], x_data[1])
            grid = np.array((xx.ravel(), yy.ravel())).T
            values, _ = self.model_diff.predict_f(grid)

            return [np.float_(values.numpy().ravel()[0]), np.float_(values.numpy().ravel()[1])]

        return _model_generator

    def get_ep_evs(self):
        """
        Get the eigenvalues belonging to the EP.
        """

        def _ep_evs(kappa):
            """
            Function to compute and select  eigenvalues belonging to the EP.

            Parameters
            ----------
            kappa : np.ndarray
                    kappa value for which you want the eigenvalues
            """
            symmatrix = matrix.matrix_one_close_re(np.array([complex(kappa[0], kappa[1])]))
            ev_new = matrix.eigenvalues(symmatrix)
            xx, yy = np.meshgrid(kappa[0], kappa[1])
            grid = np.array((xx.ravel(), yy.ravel())).T
            mean_diff, var_diff = self.model_diff.predict_f(grid)
            mean_sum, var_sum = self.model_sum.predict_f(grid)
            pairs_diff_all = np.empty(0)
            pairs_sum_all = np.empty(0)
            ev_1 = np.empty(0)
            ev_2 = np.empty(0)
            for i, val in enumerate(ev_new[0, ::]):
                pairs_diff = vmap(lambda a, b: jnp.power((a - b), 2),
                                  in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
                pairs_sum = vmap(lambda a, b: 0.5 * jnp.add(a, b),
                                 in_axes=(None, 0), out_axes=0)(val, ev_new[0, (i + 1):])
                ev_1 = np.concatenate((ev_1, np.array([val for _ in range(len(ev_new[0, (i + 1):]))])))
                ev_2 = np.concatenate((ev_2, ev_new[0, (i + 1):]))
                pairs_diff_all = np.concatenate((pairs_sum_all, np.array(pairs_diff)))
                pairs_sum_all = np.concatenate((pairs_sum_all, np.array(pairs_sum)))
            compatibility = - np.power(pairs_diff_all.real - mean_diff.numpy()[0, 0], 2) / (2 * var_diff.numpy()[0, 0]) \
                            - np.power(pairs_diff_all.imag - mean_diff.numpy()[0, 1], 2) / (2 * var_diff.numpy()[0, 1]) \
                            - np.power(pairs_sum_all.real - mean_sum.numpy()[0, 0], 2) / (2 * var_sum.numpy()[0, 0]) \
                            - np.power(pairs_sum_all.imag - mean_sum.numpy()[0, 1], 2) / (2 * var_sum.numpy()[0, 1])
            # c = np.array([0 for _ in compatibility])
            # fig1 = px.scatter(x=c, y=abs(compatibility), log_y=True)
            # fig1.show()
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=ev_new.ravel().real, y=ev_new.ravel().imag, mode='markers',
            #                         name="All eigenvalues"))
            # fig.add_trace(go.Scatter(x=x, y=mean_re.numpy().ravel(), mode='makers', name="Eigenvalues of EP",
            #                         marker=dict(color='red')))
            # fig.show()
            diff = ev_1[np.argmax(compatibility)] - ev_2[np.argmax(compatibility)]
            return np.array([np.float_(diff.real), np.float_(diff.imag)])

        return _ep_evs
