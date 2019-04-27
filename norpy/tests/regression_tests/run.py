'''




'''
import os

import numpy as np

from norpy.simulate_model import simulate
from norpy.model_spec import get_random_model_specification, get_model_obj



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
    file_dir = os.path.join(TEST_RESOURCES_DIR, "regression_vault.norpy.json")
    tests = []

    for counter, seed in enumerate(seeds):

        np.random.seed(seed)


        init_dict = get_random_model_specification()

        model_object = get_model_obj(init_dict)

        df = simulate(model_object)

        stat = np.sum(df.sum())

        tests += [(stat, init_dict)]



    json.dump(tests, open(file_dir, "w"))






