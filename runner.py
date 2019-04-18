"""This module allows to run the NORPY hatchery.

    TODO:
        * We need to remove all traces of Occupation B. This is not relevant for Norway as we do
        not have occupational information in the dataset. This should be the very first step and
        affects the state space creation but also the calculation of the immediate rewards.
        -- We can use this as a regression test setup for the following step.
        --
        * We want to be able to include ordered categories of cognitive skills into the model.
        For example, in the data we will have nine ranked groups available. Again, both functions
        are affected.
        ---
        NOTES:
            * Set up a separate PRs for both so that I can review separately. We want to use
            checklists.




Ok irgendwie funktioniert die immediate rewards funktion nicht debug it !
"""
import os

from scipy.stats import invwishart
from numpy import f2py
import pandas as pd
import numpy as np

# If you need to compile the F2PY interface again.
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

    os.chdir('../../')


from norpy.src.norpy_hatchery import f2py_calculate_immediate_rewards
from norpy.src.norpy_hatchery import f2py_create_state_space
from norpy.src.norpy_hatchery import f2py_backward_induction
from norpy.src.norpy_hatchery import f2py_simulate

from auxiliary import DATA_FORMATS_SIM
from auxiliary import DATA_LABELS_SIM
from auxiliary import MISSING_FLOAT
from auxiliary import MISSING_INT
from auxiliary import HUGE_FLOAT

np.random.seed(123)

for _ in range(10):
    print('succesfully completed {}th trial '.format(_))

    # We want to set up a basic testing infrastructure for the state space creation.
    num_periods = np.random.randint(1, 10)
    num_types = np.random.randint(1, 5)
    num_draws_emax = np.random.randint(10, 200)
    delta = np.random.uniform(0.01, 0.99)
    
    shocks_cov = invwishart.rvs(df=3, scale=np.identity(3))
    
    num_edu_start = np.random.randint(1, 5)
    edu_spec_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
    edu_spec_max = np.random.randint(15, 25)

    type_spec_shifts = np.random.normal(size=num_types*3).reshape((num_types, 3))
    coeffs_common = np.random.uniform(size=2)
    coeffs_home = np.random.uniform(size=3)
    coeffs_edu = np.random.uniform(size=7)
    coeffs_work = np.random.uniform(size=13)
    
    num_agents_sim = np.random.randint(10, 100)


    args = (num_periods, num_types, edu_spec_start, edu_spec_max, edu_spec_max + 1)
    states_all, states_number_period, mapping_state_idx, max_states_period= f2py_create_state_space(*args)
    states_all = states_all[:, :max(states_number_period), :]
    state_space_info = [states_all, states_number_period, mapping_state_idx, max_states_period]

    args = (np.zeros(3), shocks_cov, (num_periods, num_draws_emax))
    periods_draws_emax = np.random.multivariate_normal(*args)
    periods_draws_emax[:, :, :2] = np.clip(np.exp(periods_draws_emax[:, :, :2]), 0.0, HUGE_FLOAT)

    args = list() 
    
    args += [num_periods,states_number_period, states_all, max_states_period, coeffs_common, coeffs_work]
    args += [coeffs_edu, coeffs_home, type_spec_shifts]
    periods_rewards_systematic = f2py_calculate_immediate_rewards(*args)
    print(periods_rewards_systematic)
    
    args = list()
    args += state_space_info[0:3]
    args += [num_periods,max_states_period, periods_draws_emax, num_draws_emax]
    args += [periods_rewards_systematic, edu_spec_max, delta]
    args += [coeffs_common, coeffs_work]
    
    periods_emax = f2py_backward_induction(*args)   
    
    # We need to simulate the initial conditions of the individuals for the simulation. Here we
    # take a couple of shortcuts that need to be improved upon later: (1) the sampling of the
    # lagged choice in the beginning is specified by education level, (2) the sampling of types
    # depends on the initial education level, ...
    args = (np.zeros(3), shocks_cov, (num_periods, num_agents_sim))
    periods_draws_sims = np.random.multivariate_normal(*args)
    periods_draws_sims[:, :, :2] = np.clip(np.exp(periods_draws_sims[:, :, :2]), 0.0, HUGE_FLOAT)

    sample_lagged_start = np.random.choice([3, 3], p=[0.1, 0.9], size=num_agents_sim)
    sample_edu_start = np.random.choice(edu_spec_start, size=num_agents_sim)
    sample_types = np.random.choice(range(num_types), size=num_agents_sim)
    
    args = [states_all, mapping_state_idx, periods_rewards_systematic, periods_emax,
            num_periods, num_agents_sim, periods_draws_sims, edu_spec_max, coeffs_common,
            coeffs_work,  delta, sample_edu_start, sample_types, sample_lagged_start]
    
    dat = f2py_simulate(*args)

    # We can now set up an easily accessible data frame for exploratory testing.
    print('is dieses zeug des problem ?')
    df = pd.DataFrame(data=dat, columns=DATA_LABELS_SIM).astype(DATA_FORMATS_SIM).\
        sort_values(["Identifier", "Period"])
    print('eha ned')            
    df.replace([MISSING_FLOAT, MISSING_INT], np.nan)
    df.to_pickle('data.norpy.pkl')

    # Some crude regression tests to catch deviations quickly.
    
    #if _ == 0:
       # np.testing.assert_equal(periods_emax.sum(), 1063396613.0722561)
       # np.testing.assert_equal(dat.sum(), 44852497673.30828)



