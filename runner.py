"""This module allows to run the hatchery."""
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

    os.chdir('norpy/src')

    args = ''
    args += '--f90exec=gfortran --f90flags=' + '"' + ' '.join(FLAGS_DEBUG) + '" '
    args += ' -llapack'

    src = open('norpy_hatchery.f90', 'rb').read()
    f2py.compile(src, 'norpy_hatchery', args, extension='.f90')

    os.chdir('../')

from norpy.src.norpy_hatchery  import f2py_calculate_immediate_rewards
from norpy.src.norpy_hatchery  import f2py_create_state_space


for _ in range(100):

    # We want to set up a basic testing infrastructure for the state space creation.
    num_periods = np.random.randint(1, 10)
    num_types = np.random.randint(1, 5)

    num_edu_start = np.random.randint(1, 5)
    edu_spec_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
    edu_spec_max = np.random.randint(15, 25)

    args = (num_periods, num_types, edu_spec_start, edu_spec_max, edu_spec_max + 1)
    f2py = f2py_create_state_space(*args)
