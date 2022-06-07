import numpy as np
import gpflow


class GPFlowModel:
    """
    Class for the deployment of a GPFlow
    model.
    """

    def __init__(
            self,
            real_model: gpflow.models.GPModel,
            imag_model: gpflow.models.GPModel
    ):
        """
        Constructor for the model class.

        Parameters
        ----------
        real_model : gpflow.models.GPFlow
                Optimized real model from GPFlow
        imag_model : gpflow.models.GPFlow
                Optimized imaginary model from GPFlow
        """
        self.real_model = real_model
        self.imag_model = imag_model

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
            real_values, _ = self.real_model.predict_f(grid)
            imag_values, _ = self.imag_model.predict_f(grid)

            return [np.float_(real_values.numpy()), np.float_(imag_values.numpy())]  # (2, n_points)

        return _model_generator
