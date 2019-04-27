'''
This file simulates a model object and retruns the result
'''

import os

import numpy as np
import pandas as pd
from scipy.stats import invwishart


from create_model_object import
from norpy.src.norpy_hatchery import f2py_calculate_immediate_rewards
from norpy.src.norpy_hatchery import f2py_create_state_space
from norpy.src.norpy_hatchery import f2py_backward_induction
from norpy.src.norpy_hatchery import f2py_simulate



def simulate(model_object):

