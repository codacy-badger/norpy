#!/usr/bin/env python
"""This script allows to run the regression tests."""
import pickle as pkl
import argparse

from ose_utils.testing import create_regression_vault, check_regression_vault
from norpy.model_spec import get_random_model_specification
from norpy.tests.auxiliary import run_regression_test
from norpy.norpy_config import PACKAGE_DIR


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run regression tests')

    parser.add_argument('-t', '--tests', type=int, help='number of tests to analyze',
                        default=1, dest='num_tests', required=True)

    parser.add_argument('--create', action='store_true', dest='is_create',
                        help='create vault (instead of checking)')

    args = parser.parse_args()

    if args.is_create:
        input_ = (run_regression_test, get_random_model_specification, args.num_tests)
        vault = create_regression_vault(*input_)
        pkl.dump(vault, open(PACKAGE_DIR / 'tests/regression_vault.pkl', 'wb'))

    vault = pkl.load(open(PACKAGE_DIR / 'tests/regression_vault.pkl', 'rb'))
    check_regression_vault(run_regression_test, args.num_tests, vault)
