'''
This file simulates a model object and retruns the result
TODO:
auxiliary.py file should become part of model specs !
'''

import os
from numpy import f2py
import numpy as np
import pandas as pd
from scipy.stats import invwishart


FLAGS_DEBUG = []

FLAGS_DEBUG += ['-O', '-Wall', '-Wline-truncation', '-Wsurprising', '-Waliasing']
FLAGS_DEBUG += ['-Wunused-parameter', '-fwhole-file', '-fcheck=all']
FLAGS_DEBUG += ['-fbacktrace', '-g', '-fmax-errors=1', '-ffree-line-length-0']
FLAGS_DEBUG += ['-cpp', '-Wcharacter-truncation', '-Wimplicit-interface']


if True:

    cmd = "git clean -df"
    os.system(cmd)

    # We need to process in two steps. First we compile a library and then use it in a special
    # F2PY interface for Python.
    os.chdir("norpy/solve")

    cmd = "gfortran -c -fPIC lib_norpy.f90"
    os.system(cmd)

    cmd = "ar crs libnorpy.a lib_norpy.o"
    os.system(cmd)

    args = ""
    args += "--f90exec=gfortran --f90flags=" + '"' + " ".join(FLAGS_DEBUG) + '" '
    args += "-L. -lnorpy -llapack"

    src = open("norpy_hatchery.f90", "rb").read()
    f2py.compile(src, "norpy_hatchery", args, extension=".f90")

    os.chdir('../../')

from norpy.solve.norpy_hatchery import f2py_calculate_immediate_rewards
from norpy.solve.norpy_hatchery import f2py_create_state_space
from norpy.solve.norpy_hatchery import f2py_backward_induction
from norpy.solve.norpy_hatchery import f2py_simulate

from norpy.tests.auxiliary import DATA_FORMATS_SIM
from norpy.tests.auxiliary import DATA_LABELS_SIM
from norpy.tests.auxiliary import MISSING_FLOAT
from norpy.tests.auxiliary import MISSING_INT
from norpy.tests.auxiliary import HUGE_FLOAT



def create_state_space(model_object):
    args = (model_object.num_periods, model_object.num_types, model_object.edu_spec_start,
            model_object.edu_spec_max, model_spec.edu_spec_max + 1)
    states_all, states_number_period, mapping_state_idx, max_states_period = f2py_create_state_space(
        *args
    )
    states_all = states_all[:, : max(states_number_period), :]
    state_space_info = [
        states_all,
        states_number_period,
        mapping_state_idx,
        max_states_period
    ]
    return state_space_info





def return_immediate_rewards(model_object,state_space_info):
    args = list()
    args += [model_object.num_periods, state_space_info["states_number_period"],
             state_space_info["states_all"], state_space_info["max_states_period"], model_object.coeffs_common,
             model_object.coeffs_work]
    args += [model_object.coeffs_edu, model_object.coeffs_home, model_object.type_spec_shifts]

    periods_rewards_systematic = f2py_calculate_immediate_rewards(*args)

    return periods_rewards_systematic






def return_simulated_shocks(model_object,simulation=False):
    if simulation == True:
        args = (np.zeros(3), model_object.shocks_cov, (model_object.num_periods, model_object.num_agents_sim))
    else:
        args = (np.zeros(3), model_object.shocks_cov, (model_object.num_periods, model_object.num_agents_sim))

    periods_draws_emax = np.random.multivariate_normal(*args)
    periods_draws_emax[:, :, :2] = np.clip(np.exp(periods_draws_emax[:, :, :2]), 0.0, HUGE_FLOAT)

    return periods_draws_emax




def backward_induction_procedure(model_object, state_space_info,periods_rewards_systematic, periods_draws_emax ):
    '''
    Performs backward induction procedure on the whole state space:
    ARGS:


    RETURNS:

'''
    args = list()
    args += [state_space_info["states_all"],
             state_space_info["states_number_period"],
             state_space_info["mapping_state_idx"]
             ]
    args += [model_object.num_periods,
             state_space_info["max_states_period"],
             periods_draws_emax,
             model_object.num_draws_emax
             ]

    args += [periods_rewards_systematic, model_object.edu_spec_max, model_object.delta]
    args += [model_object.coeffs_common, model_object.coeffs_work]


    periods_emax = f2py_backward_induction(*args)
    return periods_emax









def simulate(model_object):
    '''
    Simulate the full model and return relevant results:
    All intermediate results are saved as local variables and all inputs are stored in a
    named tuple!

    Woher kommen diese ganzen sample variablen ?



    '''
    state_space_info = create_state_space(model_object)
    periods_rewards_systematic = return_immediate_rewards(model_object,state_space_info)
    periods_draws_emax = return_simulated_shocks(model_object)
    periods_draws_sims = return_simulated_shocks(model_object,True)
    periods_emax = backward_induction_procedure(model_object,state_space_info,periods_rewards_systematic, periods_draws_emax)

    sample_lagged_start = np.random.choice([3, 3], p=[0.1, 0.9], size=model_object.num_agents_sim)
    sample_edu_start = np.random.choice(edu_spec_start, size=model_object.num_agents_sim)
    sample_types = np.random.choice(range(num_types), size=model_object.num_agents_sim)

    args = [state_space_info["states_all"], state_space_info["mapping_state_idx"],
            periods_rewards_systematic, periods_emax,
            model_object.num_periods, model_object.num_agents_sim,
            periods_draws_sims, model_object.edu_spec_max,
            model_object.coeffs_common,
            model_object.coeffs_work, delta,
            sample_edu_start, sample_types, sample_lagged_start]

    dat = f2py_simulate(*args)


    return dat