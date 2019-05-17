#!/usr/bin/env python
"""This script checks whether the package performs properly for random requests."""
import argparse

from ose_utils.testing import run_property_tests
import norpy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run regression tests")

    parser.add_argument(
        "-hrs", "--hours", type=float, help="hours to run", default=1, dest="hours"
    )

    args = parser.parse_args()

    run_property_tests(norpy, args.hours)
