import os
import pickle
import sys 



import numpy as np
#import pytest

from norpy import simulate
from norpy import TEST_RESOURCES_DIR
from norpy import get_model_obj, get_random_model_specification


def test1():
    """This test runs a random selection of five regression tests from
    our regression test battery.
    """

    fname = TEST_RESOURCES_DIR / "regression_vault.pickle"
    tests = pickle.load(open(fname, "rb"))
    random_choice = np.random.choice(range(len(tests)), 5)
    tests = [tests[i] for i in random_choice]

    for test in tests:

        stat, init_dict = test

        model_object = get_model_obj(init_dict)

        df = simulate(model_object)

        stat_new = np.sum(df.sum())

    np.testing.assert_array_equal(stat_new, stat)
