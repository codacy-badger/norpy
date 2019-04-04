"""This module allows to run the NORPY hatchery."""
import os

from numpy import f2py
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

    os.chdir("../")

from norpy.src.norpy_hatchery import f2py_calculate_immediate_rewards
from norpy.src.norpy_hatchery import f2py_create_state_space

for _ in range(1000):
    print("\n .. random request", _)

    # We want to set up a basic testing infrastructure for the state space creation.
    num_periods = np.random.randint(1, 10)
    num_types = np.random.randint(1, 5)

    # TODO: Here we need a draw for the random number of cognitive skill ranks.
    # num_skills = np.random.randint(1, 5)

    num_edu_start = np.random.randint(1, 5)
    edu_spec_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
    edu_spec_max = np.random.randint(15, 25)

    args = (num_periods, num_types, edu_spec_start, edu_spec_max, edu_spec_max + 1)
    states_all, states_number_period, mapping_state_idx, max_states_period = f2py_create_state_space(
        *args
    )

    # If we want to continue, then we need to cut the states_all container down to size.
    states_all = states_all[:, :max_states_period, :]
    print(states_all)
    print(states_all[-1][-1])

    # TODO: The coefficient groups now need to be amended in spirit of Belzil & al. (2017, QE)
    #  and then incorporated in the calculation of the immediate rewards.
    coeffs_common = np.random.uniform(size=2)
    coeffs_home = np.random.uniform(size=3)
    coeffs_edu = np.random.uniform(size=7)
    coeffs_a = np.random.uniform(size=13)

    # TODO: more flexible
    type_spec_shifts = np.tile(0.0, (num_types, 4))
    # type_spec_shifts = np.concatenate((np.tile(0.0, (num_types, 4)),np.array([1,1,1,1]).reshape(1,4)))

    args = list()
    args += [num_periods, states_number_period, states_all, max_states_period]
    args += [coeffs_common, coeffs_a, coeffs_edu, coeffs_home, type_spec_shifts]

    f2py_calculate_immediate_rewards(*args)

