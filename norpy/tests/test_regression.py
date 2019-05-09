import pickle as pkl

import numpy as np

from norpy.tests.auxiliary import run_regression_test
from norpy.norpy_config import TEST_RESOURCES_DIR


def test1():
    """This test runs the first five regression tests from our regression vault.
    """
    for test in pkl.load(open(TEST_RESOURCES_DIR / "regression_vault.pkl", "rb"))[:5]:
        init_dict, rslt = test
        np.testing.assert_array_equal(run_regression_test(init_dict), rslt)
