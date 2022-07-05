import numpy as np
import gpflow


class GPFlowModel:
    """
    Class for the deployment of a GPFlow
    model.
    """

    def __init__(
            self,
            model: gpflow.models.GPModel
    ):
        """
        Constructor for the model class.

        Parameters
        ----------
        model : gpflow.models.GPFlow
                Optimized model from GPFlow
        """
        self.model = model

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
            values, _ = self.model.predict_f(grid)

            return [np.float_(values.numpy().ravel()[0]), np.float_(values.numpy().ravel()[1])]

        return _model_generator
