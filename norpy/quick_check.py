import os

import numpy as np

from norpy.simulate_model import simulate
from norpy.model_spec import get_random_model_specification, get_model_obj

model_object = get_model_obj(get_random_model_specification())

dat = simulate(model_object)

print(dat.sum())