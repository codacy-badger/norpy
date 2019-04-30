"""




"""
import os
import argparse
import pickle

import numpy as np

from norpy.simulate.simulate import simulate
from norpy.model_spec import get_random_model_specification, get_model_obj
from norpy.norpy_config import TEST_RESOURCES_DIR


def process_arguments(parser):
    """This function parses the input arguments."""
    args = parser.parse_args()

    # Distribute input arguments
    request = args.request
    num_test = args.num_test
    seed = args.seed
    # Test validity of input arguments
    assert request in ["check", "create"]

    if num_test is None:
        num_test = 100

    if seed is None:
        seed = 123456

    return request, num_test, seed


def create_vault(num_test=100, seed=123456):
    """This function creates our regression vault."""
    np.random.seed(seed)
    seeds = np.random.randint(0, 1000, size=num_test)
    file_dir = os.path.join(TEST_RESOURCES_DIR, "regression_vault.pickle")
    tests = []

    for counter, seed in enumerate(seeds):

        np.random.seed(seed)

        init_dict = get_random_model_specification()
        # I sthat fine like this or should I first save to yaml and then give the input ?
        # I dont see how that is better
        model_object = get_model_obj(init_dict)

        df = simulate(model_object)

        stat = np.sum(df.sum())

        tests += [(stat, init_dict)]

    pickle.dump(tests, open(file_dir, str("wb")))


def check_vault():
    """This function runs another simulation for each init file in our regression vault.
    """
    file_dir = os.path.join(TEST_RESOURCES_DIR, "regression_vault.pickle")

    tests = pickle.load(open(file_dir, "rb"))
    for test in tests:

        stat, init_dict = test
        print(init_dict)
        df = simulate(get_model_obj(init_dict))

        stat_new = np.sum(df.sum())

        np.testing.assert_array_almost_equal(stat, stat_new)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Work with regression tests for package."
    )

    parser.add_argument(
        "--request",
        action="store",
        dest="request",
        required=True,
        choices=["check", "create"],
        help="request",
    )

    parser.add_argument(
        "--num", action="store", dest="num_test", type=int, help="number of init files"
    )

    parser.add_argument(
        "--seed", action="store", dest="seed", type=int, help="seed value"
    )

    request, num_test, seed = process_arguments(parser)

    if request == "check":
        check_vault()

    elif request == "create":

        create_vault(num_test, seed)
