#!/usr/bin/env python
"""This script allows us to run some more extensive testing for our pull requests."""
import subprocess
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_pr_testing(robustness, property_, regression):

    print(" \n ... running robustness tests")
    cmd = "./run.py --hours {:}".format(robustness)
    subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR + "/robustness")

    print(" \n ... running regression tests")
    cmd = ""
    cmd += "./run.py --tests {:}".format(regression)
    subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR + "/regression")

    print(" \n ... running property tests")
    cmd = "./run.py --request run --hours {:}".format(property_)
    subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR + "/property")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run regression tests")

    parser.add_argument(
        "-p",
        "--property",
        type=float,
        help="hours to run for property tests",
        default=0.001,
        dest="property",
    )

    parser.add_argument(
        "-rob",
        "--robustness",
        type=float,
        help="hours to run for robustness tests",
        default=0.001,
        dest="robustness",
    )

    parser.add_argument(
        "-reg",
        "--regression",
        type=float,
        help="number of regression tests to run",
        default=1,
        dest="regression",
    )

    args = parser.parse_args()

    run_pr_testing(args.robustness, args.property, args.regression)
