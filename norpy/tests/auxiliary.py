"""This module contains some auxiliary functions for testing."""
import numpy as np

from norpy import get_model_obj
from norpy import simulate


def run_regression_test(init_dict):
    """This function evaluates a single regression test."""
    df = simulate(get_model_obj(init_dict))
    return np.sum(df.sum())
