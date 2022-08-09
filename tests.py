import numpy as np
import matrix
import plots
import time
import gaussian_process_regression as gpr
import zero_search as zs
import matplotlib.pyplot as plt
import GPFlow_model_class as GPFmc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import jax.numpy as jnp
from jax import vmap
import data_preprocessing as dpp
import ast

if __name__ == '__main__':
    kappa, phi = matrix.parametrization(1.1850345+1.0084825j, 1e-7, 200)
    symmatrix = matrix.matrix_one_close_re(kappa)
    ev = matrix.eigenvalues(symmatrix)
    plots.energy_plane_plotly(ev, phi)
    ev = dpp.initial_dataset(ev)
    plots.energy_plane_plotly(ev, phi)
