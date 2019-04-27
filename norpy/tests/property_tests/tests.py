#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametize the test functions 

TODO: Is that one just the same as test_grid_rewards??
Check if so then erase it !

"""

import os
import unittest as ut

import pytest
from numpy import f2py
import numpy as np
import math

# If you need to compile the F2PY interface again.

FLAGS_DEBUG = []
FLAGS_DEBUG += ["-O", "-Wall", "-Wline-truncation", "-Wsurprising", "-Waliasing"]
FLAGS_DEBUG += ["-Wunused-parameter", "-fwhole-file", "-fcheck=all"]
FLAGS_DEBUG += ["-fbacktrace", "-g", "-fmax-errors=1", "-ffree-line-length-0"]
FLAGS_DEBUG += ["-cpp", "-Wcharacter-truncation", "-Wimplicit-interface"]

if True:

    cmd = "git clean -df"
    os.system(cmd)

    # We need to process in two steps. First we compile a library and then use it in a special
    # F2PY interface for Python.
    os.chdir("norpy/src")

    cmd = "gfortran -c -fPIC lib_norpy.f90"
    os.system(cmd)

    cmd = "ar crs libnorpy.a lib_norpy.o"
    os.system(cmd)

    args = ""
    args += "--f90exec=gfortran --f90flags=" + '"' + " ".join(FLAGS_DEBUG) + '" '
    args += "-L. -lnorpy -llapack"

    src = open("norpy_hatchery.f90", "rb").read()
    f2py.compile(src, "norpy_hatchery", args, extension=".f90")

    os.chdir("../")

from norpy.solve.norpy_hatchery import f2py_calculate_immediate_rewards
from norpy.solve.norpy_hatchery import f2py_create_state_space


def set_up_state_space(boolean):
    args = {}
    args["num_periods"] = np.random.randint(1, 10)
    args["num_types"] = np.random.randint(1, 5)
    num_edu_start = np.random.randint(1, 5)
    args["edu_spec_start"] = np.random.choice(
        range(1, 10), size=num_edu_start, replace=False
    )
    args["edu_spec_max"] = np.random.randint(15, 25)
    args["min_idx_int"] = args["edu_spec_max"] + 1
    args["test_indication_optional"] = boolean
    states_all, states_number_period, mapping_state_idx, max_states_period = f2py_create_state_space(
        **args
    )
    return (
        args,
        states_all,
        states_number_period,
        mapping_state_idx,
        max_states_period,
        num_edu_start,
    )


def get_immediate_rewards():
    args, states_all, states_number_period, mapping_state_idx, max_states_period, num_edu_start = set_up_state_space(
        False
    )
    coeffs_common = np.random.uniform(size=2)
    coeffs_home = np.random.uniform(size=3)
    coeffs_edu = np.random.uniform(size=7)
    coeffs_a = np.random.uniform(size=13)

    # TODO: more flexible
    type_spec_shifts = np.tile(0.0, (args["num_types"], 4))

    args_2 = list()
    args_2 += [args["num_periods"], states_number_period, states_all, max_states_period]
    args_2 += [coeffs_common, coeffs_a, coeffs_edu, coeffs_home, type_spec_shifts]

    immediate_rewards = f2py_calculate_immediate_rewards(*args_2)
    return (
        immediate_rewards,
        args,
        states_all,
        states_number_period,
        mapping_state_idx,
        max_states_period,
        coeffs_common,
        coeffs_home,
        coeffs_edu,
        coeffs_a,
    )


@pytest.fixture(params=[str(x) for x in list(range(10))])
def input_output_state_space():
    args, states_all, states_number_period, mapping_state_idx, max_states_period, num_edu_start = set_up_state_space(
        False
    )

    period = np.random.randint(1, args["num_periods"] + 1)
    # Split up different cases
    type_ = np.random.randint(1, args["num_types"] + 1)
    edu_start = args["edu_spec_start"][np.random.randint(len(args["edu_spec_start"]))]
    exp_a = np.random.randint(period)
    edu_add = np.random.randint(min(period - exp_a, args["edu_spec_max"] - edu_start))
    print(edu_start)

    if exp_a == 0:
        if period == 1:
            lagged_choice = np.random.randint(2, 4)
        elif edu_add == period - 1 and period > 1:
            lagged_choice = 2
        elif edu_add == 0:
            lagged_choice = 3
        else:
            lagged_choice = np.random.randint(2, 4)

    elif edu_add == 0:
        if exp_a == period - 1:
            lagged_choice = 1
        else:

            lagged_choice = [1, 3][np.random.randint(2)]

    elif edu_add + exp_a == period - 1 and period > 1:
        lagged_choice = np.random.randint(1, 3)
    else:
        lagged_choice = np.random.randint(1, 4)

    manual = np.array([exp_a, edu_start + edu_add, lagged_choice, type_])

    states_all = states_all[:, :max_states_period, :]
    return states_all, manual, period


def test_state_space_1(input_output_state_space):
    assert np.any(
        np.all(
            input_output_state_space[0][input_output_state_space[2] - 1]
            == input_output_state_space[1],
            axis=1,
        )
    )


@pytest.fixture(params=[str(x) for x in list(range(10))])
def input_not_output_state_space():
    args, states_all, states_number_period, mapping_state_idx, max_states_period, num_edu_start = set_up_state_space(
        False
    )
    period = np.random.randint(1, args["num_periods"] + 1)
    # Split up different cases
    type_ = np.random.randint(1, args["num_types"] + 1)
    edu_start = args["edu_spec_start"][np.random.randint(len(args["edu_spec_start"]))]
    exp_a = np.random.randint(period)
    edu_add = np.random.randint(period - exp_a)
    exp_a == 0
    if exp_a == 0:
        lagged_choice = 1
    elif edu_add == 0 and period >= 2:
        lagged_choice = 2
        edu_start = min(
            args["edu_spec_start"]
        )  # Here we have to be careful because only this case is not allowed
    else:
        lagged_choice = 300000

    states_all = states_all[:, :max_states_period, :]
    manual = np.array([exp_a, edu_start + edu_add, lagged_choice, type_])
    return states_all, manual, period, max_states_period


def test_state_space_2(input_not_output_state_space):
    assert (
        np.any(
            np.all(
                input_not_output_state_space[0][
                    input_not_output_state_space[2] - 1
                ].reshape(input_not_output_state_space[3], 4)
                == input_not_output_state_space[1],
                axis=1,
            )
        )
        == False
    )


@pytest.fixture(params=[str(x) for x in list(range(10))])
def input_output_size():
    args, states_all, states_number_period, mapping_state_idx, max_states_period, num_edu_start = set_up_state_space(
        False
    )
    states_all = states_all[:, :max_states_period, :]

    max_period = np.where(states_number_period == max_states_period)
    return states_all, max_period, max_states_period, states_number_period


def test_state_space_3(input_output_size):
    assert input_output_size[0][input_output_size[1]] != np.array(
        [-99, -99, -99, -99, -99]
    )
    assert input_output_size[3].max() == input_output_size[2]


@pytest.fixture(params=[str(x) for x in list(range(10))])
def input_output_dimension():
    args, states_all, states_number_period, mapping_state_idx, max_states_period, num_edu_start = set_up_state_space(
        True
    )
    states_all = states_all[:, : max_states_period + 1, :]
    period = np.random.randint(1, args["num_periods"] + 1)

    if period > 1:
        dim_period = int(
            ((((period) ** 2 + period) / 2) - (period))
            * 3
            * num_edu_start
            * args["num_types"]
        )
    else:
        dim_period = int(2 * num_edu_start * args["num_types"])

    return (
        states_all,
        max_states_period,
        states_number_period,
        period,
        dim_period,
        num_edu_start,
    )


def test_state_space_dimension(input_output_dimension):
    assert (
        input_output_dimension[2][input_output_dimension[3] - 1]
        == input_output_dimension[4]
    )


###############################################################################
######################Test immediate rewards###################################
###############################################################################


@pytest.fixture(params=[str(x) for x in list(range(10))])
def input_output_immediate_rewards_home():

    immediate_rewards, args, states_all, states_number_period, mapping_state_idx, max_states_period, coeffs_common, coeffs_home, coeffs_edu, coeffs_a = (
        get_immediate_rewards()
    )
    # Randomly draw a position on the state space
    period_to_check = np.random.randint(1, args["num_periods"] + 1)
    k_to_check = np.random.randint(states_number_period[period_to_check - 1])
    # - one to modify fortran indexing to python indexing

    states_to_check = states_all[period_to_check - 1, k_to_check]
    # initialize manual_result
    manually_calculated_result = 0
    # Calculate common part of home rewards
    if states_to_check[1] < 12:
        manually_calculated_result = 0
    elif states_to_check[1] < 15:
        manually_calculated_result = coeffs_common[0]
    else:
        manually_calculated_result = coeffs_common[0] + coeffs_common[1]
    # Calculate specific part of home rewards
    if 3 <= period_to_check < 6:

        manually_calculated_result = (
            manually_calculated_result + coeffs_home[0] + coeffs_home[1]
        )
    elif period_to_check >= 6:
        manually_calculated_result = (
            manually_calculated_result + coeffs_home[0] + coeffs_home[2]
        )
    else:
        manually_calculated_result = manually_calculated_result + coeffs_home[0]

    return immediate_rewards, period_to_check, manually_calculated_result, k_to_check


def test_immediate_rewards_home(input_output_immediate_rewards_home):
    assert (
        input_output_immediate_rewards_home[0][
            input_output_immediate_rewards_home[1] - 1,
            input_output_immediate_rewards_home[3],
            2,
        ]
        == input_output_immediate_rewards_home[2]
    )


@pytest.fixture(params=[str(x) for x in list(range(10))])
def input_output_immediate_rewards_educ():
    immediate_rewards, args, states_all, states_number_period, mapping_state_idx, max_states_period, coeffs_common, coeffs_home, coeffs_edu, Scoeffs_a = (
        get_immediate_rewards()
    )

    period_to_check = np.random.randint(1, args["num_periods"] + 1)
    k_to_check = np.random.randint(states_number_period[period_to_check - 1])
    # - one to modify fortran indexing to python indexing

    states_to_check = states_all[period_to_check - 1, k_to_check]
    # initialize manual_result
    manually_calculated_result = coeffs_edu[0] + coeffs_edu[5] * (period_to_check - 1)
    # Calculate specific part of edu rewards
    if states_to_check[1] < 9:
        if states_to_check[2] != 2:
            manually_calculated_result = (
                coeffs_edu[6] + coeffs_edu[3] + manually_calculated_result
            )
        else:
            manually_calculated_result = coeffs_edu[6] + manually_calculated_result

    elif states_to_check[1] < 12:
        if states_to_check[2] != 2:
            manually_calculated_result = coeffs_edu[3] + manually_calculated_result
        else:
            manually_calculated_result = manually_calculated_result

    elif states_to_check[1] < 15:
        if states_to_check[2] != 2:
            manually_calculated_result = (
                coeffs_edu[4] + coeffs_edu[1] + manually_calculated_result
            )
        else:
            manually_calculated_result = manually_calculated_result + coeffs_edu[1]

    else:
        if states_to_check[2] != 2:
            manually_calculated_result = (
                manually_calculated_result
                + coeffs_edu[2]
                + coeffs_edu[1]
                + coeffs_edu[4]
            )
        else:
            manually_calculated_result = (
                manually_calculated_result + coeffs_edu[1] + coeffs_edu[2]
            )
    # Calculate common part of home rewards

    if states_to_check[1] < 12:
        manually_calculated_result = manually_calculated_result
    elif states_to_check[1] < 15:
        manually_calculated_result = coeffs_common[0] + manually_calculated_result
    else:
        manually_calculated_result = (
            coeffs_common[0] + coeffs_common[1] + manually_calculated_result
        )

    return immediate_rewards, period_to_check, k_to_check, manually_calculated_result


def test_immediate_rewards_educ(input_output_immediate_rewards_educ):
    np.testing.assert_array_almost_equal(
        np.array(
            [
                input_output_immediate_rewards_educ[0][
                    input_output_immediate_rewards_educ[1] - 1,
                    input_output_immediate_rewards_educ[2],
                    1,
                ]
            ]
        ),
        np.array([input_output_immediate_rewards_educ[3]]),
    )


@pytest.fixture(params=[str(x) for x in list(range(10))])
def input_output_immediate_rewards_occupation():

    immediate_rewards, args, states_all, states_number_period, mapping_state_idx, max_states_period, coeffs_common, coeffs_home, coeffs_edu, coeffs_a = (
        get_immediate_rewards()
    )

    # Randomly draw a position on the state space
    period_to_check = np.random.randint(1, args["num_periods"] + 1)

    k_to_check = np.random.randint(states_number_period[period_to_check - 1])
    # - one to modify fortran indexing to python indexing

    states_to_check = states_all[period_to_check - 1, k_to_check]
    # initialize manual_result
    manually_calculated_result_exponent = (
        coeffs_a[0]
        + coeffs_a[1] * states_to_check[1]
        + coeffs_a[2] * states_to_check[0]
        + coeffs_a[6] * (period_to_check - 1)
        + coeffs_a[3] * (states_to_check[0] ** 2 / 100)
    )

    if states_to_check[1] >= 12:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + coeffs_a[4]
        )

    if states_to_check[1] >= 15:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + coeffs_a[5]
        )

    if period_to_check < 3:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + coeffs_a[7]
        )
    if states_to_check[0] > 0:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + coeffs_a[8]
        )
    if states_to_check[2] == 1:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + coeffs_a[9]
        )

    manually_calculated_result = (
        math.exp(manually_calculated_result_exponent) + coeffs_a[10]
    )
    if states_to_check[2] != 1:
        if states_to_check[0] == 0:
            manually_calculated_result = manually_calculated_result + coeffs_a[12]

        else:
            manually_calculated_result = manually_calculated_result + coeffs_a[11]

    # Calculate specific part of edu rewards
    # Calculate common part of home rewards

    if states_to_check[1] < 12:
        manually_calculated_result = manually_calculated_result
    elif states_to_check[1] < 15:
        manually_calculated_result = coeffs_common[0] + manually_calculated_result
    else:
        manually_calculated_result = (
            coeffs_common[0] + coeffs_common[1] + manually_calculated_result
        )
    # Calculate specific part of home rewards

    return immediate_rewards, period_to_check, k_to_check, manually_calculated_result


def test_immediate_rewards_occupation(input_output_immediate_rewards_occupation):
    np.testing.assert_array_almost_equal(
        np.array(
            [
                input_output_immediate_rewards_occupation[0][
                    input_output_immediate_rewards_occupation[1] - 1,
                    input_output_immediate_rewards_occupation[2],
                    0,
                ]
            ]
        ),
        np.array([input_output_immediate_rewards_occupation[3]]),
    )
