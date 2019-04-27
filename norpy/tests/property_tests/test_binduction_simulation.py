#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:18:35 2019

@author: moritz
"""

import os

from scipy.stats import invwishart
from numpy import f2py
import pandas as pd
import numpy as np

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

    os.chdir("../../")

from norpy.solve.norpy_hatchery import f2py_calculate_immediate_rewards
from norpy.solve.norpy_hatchery import f2py_create_state_space
from norpy.solve.norpy_hatchery import f2py_backward_induction
from norpy.solve.norpy_hatchery import f2py_simulate


from norpy.tests.auxiliary import DATA_FORMATS_SIM
from norpy.tests.auxiliary import DATA_LABELS_SIM
from norpy.tests.auxiliary import MISSING_FLOAT
from norpy.tests.auxiliary import MISSING_INT
from norpy.tests.auxiliary import HUGE_FLOAT

# np.random.seed(123)

for _ in range(10):
    print("succesfully completed {}th trial ".format(_))
    # We want to set up a basic testing infrastructure for the state space creation.
    num_periods = np.random.randint(1, 10)
    num_types = np.random.randint(1, 5)
    num_draws_emax = np.random.randint(10, 200)
    delta = np.random.uniform(0.01, 0.99)

    shocks_cov = invwishart.rvs(df=3, scale=np.identity(3))

    num_edu_start = np.random.randint(1, 5)
    edu_spec_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
    edu_spec_max = np.random.randint(15, 25)

    type_spec_shifts = np.zeros(num_types * 3).reshape((num_types, 3))
    coeffs_common = np.random.uniform(size=2)
    coeffs_home = np.random.uniform(size=3)
    coeffs_edu = np.random.uniform(size=7)
    coeffs_work = np.random.uniform(size=13)

    num_agents_sim = np.random.randint(10, 100)

    args = (num_periods, num_types, edu_spec_start, edu_spec_max, edu_spec_max + 1)
    states_all, states_number_period, mapping_state_idx, max_states_period = f2py_create_state_space(
        *args
    )
    states_all = states_all[:, : max(states_number_period), :]
    state_space_info = [
        states_all,
        states_number_period,
        mapping_state_idx,
        max_states_period,
    ]

    args = (np.zeros(3), shocks_cov, (num_periods, num_draws_emax))
    periods_draws_emax = np.random.multivariate_normal(*args)
    periods_draws_emax[:, :, :2] = np.clip(
        np.exp(periods_draws_emax[:, :, :2]), 0.0, 1000
    )

    args = list()

    args += [
        num_periods,
        states_number_period,
        states_all,
        max_states_period,
        coeffs_common,
        coeffs_work,
    ]
    args += [coeffs_edu, coeffs_home, type_spec_shifts]
    periods_rewards_systematic = f2py_calculate_immediate_rewards(*args)

    args = list()
    args += state_space_info[0:3]
    args += [num_periods, max_states_period, periods_draws_emax, num_draws_emax]
    args += [periods_rewards_systematic, edu_spec_max, delta]
    args += [coeffs_common, coeffs_work]

    periods_emax = f2py_backward_induction(*args)

    k_to_check = np.random.randint(1, states_number_period[-1])

    draws_to_check = periods_draws_emax[num_periods - 1]

    immediate_rewards_last_period = periods_rewards_systematic[
        num_periods - 1, k_to_check
    ]
    state_to_check = states_all[num_periods - 1, k_to_check]
    aux = np.zeros(num_draws_emax * 3).reshape(num_draws_emax, 3)
    # des is der common anteil
    common = 0
    if state_to_check[1] >= 12:
        common = coeffs_common[0] + common
    else:
        common = common

    if state_to_check[1] >= 15:
        common = coeffs_common[1] + common
    else:
        common = common

    # des is der genral teil
    general = coeffs_work[10]

    print(coeffs_work)
    print(state_to_check[0] == 0)
    if state_to_check[0] == 0:
        general = general + coeffs_work[12]

    elif state_to_check[2] != 1:
        general = general + coeffs_work[11]
    else:
        general = general
    print(general)

    for x in range(num_draws_emax):
        aux[x, 0] = (
            (immediate_rewards_last_period[0] - common - general) * draws_to_check[x, 0]
            + common
            + general
        )
        aux[x, 1] = immediate_rewards_last_period[1] + draws_to_check[x, 1]
        aux[x, 2] = immediate_rewards_last_period[2] + draws_to_check[x, 2]

    out = np.zeros(num_draws_emax)

    for i in range(num_draws_emax):
        out[i] = aux[i, :].max()

    manual_result = out.sum() / num_draws_emax

    print(state_to_check)
    print(edu_spec_max)
    print(coeffs_work[11])
    print(manual_result)
    print(periods_emax[num_periods - 1, k_to_check])

    np.testing.assert_array_almost_equal(
        np.array([manual_result]),
        np.array([periods_emax[num_periods - 1, k_to_check]]),
        decimal=1,
    )


#####Now do the same for any period!


for _ in range(10):
    print("succesfully completed {}th trial ".format(_))
    # We want to set up a basic testing infrastructure for the state space creation.
    num_periods = np.random.randint(2, 10)
    num_types = np.random.randint(1, 5)
    num_draws_emax = np.random.randint(10, 200)
    delta = np.random.uniform(0.01, 0.99)

    shocks_cov = invwishart.rvs(df=3, scale=np.identity(3))

    num_edu_start = np.random.randint(1, 5)
    edu_spec_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
    edu_spec_max = np.random.randint(15, 25)

    type_spec_shifts = np.zeros(num_types * 3).reshape((num_types, 3))
    coeffs_common = np.random.uniform(size=2)
    coeffs_home = np.random.uniform(size=3)
    coeffs_edu = np.random.uniform(size=7)
    coeffs_work = np.random.uniform(size=13)
    test_indication_optional = False

    num_agents_sim = np.random.randint(10, 100)

    args = (
        num_periods,
        num_types,
        edu_spec_start,
        edu_spec_max,
        edu_spec_max + 1,
        test_indication_optional,
    )
    states_all, states_number_period, mapping_state_idx, max_states_period = f2py_create_state_space(
        *args
    )
    states_all = states_all[:, : max(states_number_period), :]
    state_space_info = [
        states_all,
        states_number_period,
        mapping_state_idx,
        max_states_period,
    ]
    print(mapping_state_idx.shape)
    args = (np.zeros(3), shocks_cov, (num_periods, num_draws_emax))
    periods_draws_emax = np.random.multivariate_normal(*args)
    periods_draws_emax[:, :, :2] = np.clip(np.exp(periods_draws_emax[:, :, :2]), 0.0, 5)

    args = list()

    args += [
        num_periods,
        states_number_period,
        states_all,
        max_states_period,
        coeffs_common,
        coeffs_work,
    ]
    args += [coeffs_edu, coeffs_home, type_spec_shifts]
    periods_rewards_systematic = f2py_calculate_immediate_rewards(*args)

    args = list()
    args += state_space_info[0:3]
    args += [num_periods, max_states_period, periods_draws_emax, num_draws_emax]
    args += [periods_rewards_systematic, edu_spec_max, delta]
    args += [coeffs_common, coeffs_work]

    periods_emax = f2py_backward_induction(*args)
    print(periods_emax.shape)

    period_to_check = np.random.randint(0, num_periods - 1)
    k_to_check = np.random.randint(0, states_number_period[period_to_check] - 1)

    draws_to_check = periods_draws_emax[period_to_check]

    immediate_rewards_last_period = periods_rewards_systematic[
        period_to_check, k_to_check
    ]
    state_to_check = states_all[period_to_check, k_to_check]
    aux = np.zeros(num_draws_emax * 3).reshape(num_draws_emax, 3)
    # des is der common anteil
    common = 0
    if state_to_check[1] >= 12:
        common = coeffs_common[0] + common
    else:
        common = common

    if state_to_check[1] >= 15:
        common = coeffs_common[1] + common
    else:
        common = common

    # des is der genral teil
    general = coeffs_work[10]

    if state_to_check[0] == 0:
        general = general + coeffs_work[12]

    elif state_to_check[2] != 1:
        general = general + coeffs_work[11]
    else:
        general = general

    ##Obtain next periods states
    next_period_1 = mapping_state_idx[
        period_to_check + 1,
        state_to_check[0] + 1,
        state_to_check[1],
        0,
        state_to_check[3] - 1,
    ]
    next_period_2 = mapping_state_idx[
        period_to_check + 1,
        state_to_check[0],
        state_to_check[1] + 1,
        1,
        state_to_check[3] - 1,
    ]
    next_period_3 = mapping_state_idx[
        period_to_check + 1,
        state_to_check[0],
        state_to_check[1],
        2,
        state_to_check[3] - 1,
    ]

    for x in range(num_draws_emax):
        aux[x, 0] = (
            (immediate_rewards_last_period[0] - common - general) * draws_to_check[x, 0]
            + common
            + general
            + delta * periods_emax[period_to_check + 1, next_period_1]
        )
        aux[x, 1] = (
            immediate_rewards_last_period[1]
            + draws_to_check[x, 1]
            + delta * periods_emax[period_to_check + 1, next_period_2]
        )
        aux[x, 2] = (
            immediate_rewards_last_period[2]
            + draws_to_check[x, 2]
            + delta * periods_emax[period_to_check + 1, next_period_3]
        )

    out = np.zeros(num_draws_emax)

    for i in range(num_draws_emax):
        out[i] = aux[i, :].max()

    manual_result = out.sum() / num_draws_emax

    np.testing.assert_array_almost_equal(
        np.array([manual_result]),
        np.array([periods_emax[period_to_check, k_to_check]]),
        decimal=1,
    )


for _ in range(10):
    print("succesfully completed {}th trial ".format(_))
    # We want to set up a basic testing infrastructure for the state space creation.
    num_periods = np.random.randint(2, 10)
    num_types = np.random.randint(1, 5)
    num_draws_emax = np.random.randint(10, 200)
    delta = np.random.uniform(0.01, 0.99)

    shocks_cov = invwishart.rvs(df=3, scale=np.identity(3))

    num_edu_start = np.random.randint(1, 5)
    edu_spec_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
    edu_spec_max = np.random.randint(15, 25)

    type_spec_shifts = np.random.normal(size=num_types * 3).reshape((num_types, 3))
    coeffs_common = np.random.uniform(size=2)
    coeffs_home = np.random.uniform(size=3)
    coeffs_edu = np.random.uniform(size=7)
    coeffs_work = np.random.uniform(size=13)

    num_agents_sim = np.random.randint(10, 100)

    args = (num_periods, num_types, edu_spec_start, edu_spec_max, edu_spec_max + 1)
    states_all, states_number_period, mapping_state_idx, max_states_period = f2py_create_state_space(
        *args
    )
    states_all = states_all[:, : max(states_number_period), :]
    state_space_info = [
        states_all,
        states_number_period,
        mapping_state_idx,
        max_states_period,
    ]

    args = (np.zeros(3), shocks_cov, (num_periods, num_draws_emax))
    periods_draws_emax = np.random.multivariate_normal(*args)
    periods_draws_emax[:, :, :2] = np.clip(
        np.exp(periods_draws_emax[:, :, :2]), 0.0, HUGE_FLOAT
    )

    args = list()

    args += [
        num_periods,
        states_number_period,
        states_all,
        max_states_period,
        coeffs_common,
        coeffs_work,
    ]
    args += [coeffs_edu, coeffs_home, type_spec_shifts]
    periods_rewards_systematic = f2py_calculate_immediate_rewards(*args)

    args = list()
    args += state_space_info[0:3]
    args += [num_periods, max_states_period, periods_draws_emax, num_draws_emax]
    args += [periods_rewards_systematic, edu_spec_max, delta]
    args += [coeffs_common, coeffs_work]

    periods_emax = f2py_backward_induction(*args)

    # We need to simulate the initial conditions of the individuals for the simulation. Here we
    # take a couple of shortcuts that need to be improved upon later: (1) the sampling of the
    # lagged choice in the beginning is specified by education level, (2) the sampling of types
    # depends on the initial education level, ...
    args = (np.zeros(3), shocks_cov, (num_periods, num_agents_sim))
    periods_draws_sims = np.random.multivariate_normal(*args)
    periods_draws_sims[:, :, :2] = np.clip(
        np.exp(periods_draws_sims[:, :, :2]), 0.0, HUGE_FLOAT
    )

    sample_lagged_start = np.random.choice([3, 3], p=[0.1, 0.9], size=num_agents_sim)
    sample_edu_start = np.random.choice(edu_spec_start, size=num_agents_sim)
    sample_types = np.random.choice(range(num_types), size=num_agents_sim)

    args = [
        states_all,
        mapping_state_idx,
        periods_rewards_systematic,
        periods_emax,
        num_periods,
        num_agents_sim,
        periods_draws_sims,
        edu_spec_max,
        coeffs_common,
        coeffs_work,
        delta,
        sample_edu_start,
        sample_types,
        sample_lagged_start,
    ]

    dat = f2py_simulate(*args)
    agent_to_check = np.random.randint(0, num_agents_sim)
    period_to_check = np.random.randint(0, num_periods - 1)

    # print(dat[num_periods*(agent_to_check-1)+period_to_check,2])
    assert dat.shape == (num_periods * num_agents_sim, 23)
    assert dat[(agent_to_check - 1) * num_periods, 4] == 0
    assert dat[(agent_to_check-1)*num_periods+period_to_check,2] in [1,2,3]
    assert dat[(agent_to_check-1)*num_periods+period_to_check,17] == delta 
    
    # assert dat[num_periods*(agent_to_check-1)+period_to_check,2]==1


# Check for arr


for _ in range(10):
    print("succesfully completed {}th trial ".format(_))
    # We want to set up a basic testing infrastructure for the state space creation.
    num_periods = np.random.randint(2, 10)
    num_types = np.random.randint(1, 5)
    num_draws_emax = np.random.randint(10, 200)
    delta = np.random.uniform(0.01, 0.99)

    shocks_cov = invwishart.rvs(df=3, scale=np.identity(3))

    num_edu_start = np.random.randint(1, 5)
    edu_spec_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
    edu_spec_max = np.random.randint(15, 25)

    type_spec_shifts = np.zeros(num_types * 3).reshape((num_types, 3))
    coeffs_common = np.random.uniform(size=2)
    coeffs_home = np.random.uniform(size=3)
    coeffs_edu = np.random.uniform(size=7)
    coeffs_work = np.concatenate(
        (np.random.uniform(size=10), np.array([2000000000000000, 0, 0]))
    )

    num_agents_sim = np.random.randint(10, 100)

    args = (num_periods, num_types, edu_spec_start, edu_spec_max, edu_spec_max + 1)
    states_all, states_number_period, mapping_state_idx, max_states_period = f2py_create_state_space(
        *args
    )
    states_all = states_all[:, : max(states_number_period), :]
    state_space_info = [
        states_all,
        states_number_period,
        mapping_state_idx,
        max_states_period,
    ]

    args = (np.zeros(3), shocks_cov, (num_periods, num_draws_emax))
    periods_draws_emax = np.random.multivariate_normal(*args)
    periods_draws_emax[:, :, :2] = np.clip(
        np.exp(periods_draws_emax[:, :, :2]), 0.0, 10000
    )

    args = list()

    args += [
        num_periods,
        states_number_period,
        states_all,
        max_states_period,
        coeffs_common,
        coeffs_work,
    ]
    args += [coeffs_edu, coeffs_home, type_spec_shifts]
    periods_rewards_systematic = f2py_calculate_immediate_rewards(*args)

    args = list()
    args += state_space_info[0:3]
    args += [num_periods, max_states_period, periods_draws_emax, num_draws_emax]
    args += [periods_rewards_systematic, edu_spec_max, delta]
    args += [coeffs_common, coeffs_work]

    periods_emax = f2py_backward_induction(*args)

    # We need to simulate the initial conditions of the individuals for the simulation. Here we
    # take a couple of shortcuts that need to be improved upon later: (1) the sampling of the
    # lagged choice in the beginning is specified by education level, (2) the sampling of types
    # depends on the initial education level, ...
    args = (np.zeros(3), shocks_cov, (num_periods, num_agents_sim))
    periods_draws_sims = np.random.multivariate_normal(*args)
    periods_draws_sims[:, :, :2] = np.clip(
        np.exp(periods_draws_sims[:, :, :2]), 0.0, 20000
    )
    

    sample_lagged_start = np.random.choice([3, 3], p=[0.1, 0.9], size=num_agents_sim)
    sample_edu_start = np.random.choice(edu_spec_start, size=num_agents_sim)
    sample_types = np.random.choice(range(num_types), size=num_agents_sim)

    args = [
        states_all,
        mapping_state_idx,
        periods_rewards_systematic,
        periods_emax,
        num_periods,
        num_agents_sim,
        periods_draws_sims,
        edu_spec_max,
        coeffs_common,
        coeffs_work,
        delta,
        sample_edu_start,
        sample_types,
        sample_lagged_start,
    ]

    dat = f2py_simulate(*args)
    agent_to_check = np.random.randint(0, num_agents_sim)
    period_to_check = np.random.randint(0, num_periods - 1)

    print(dat[num_periods * (agent_to_check - 1) + period_to_check, 3])
    print(dat[num_periods * (agent_to_check - 1) + period_to_check, 8:11])
    print(dat[num_periods * (agent_to_check - 1) + period_to_check, 14:17])
    print(dat[num_periods * (agent_to_check - 1) + period_to_check, 11:14])
    print(dat[num_periods*(agent_to_check-1)+period_to_check,2])
    assert dat[(agent_to_check-1)*num_periods+period_to_check,2] == 1
    # Todo:check for high common, check agent identifier, check period ident
    
    
