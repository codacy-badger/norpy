"""
Initial sketch of the adapter class.
"""
from functools import partial
import os

import pandas as pd
import numpy as np


from respy_smm.auxiliary import is_valid_covariance_matrix
from respy_smm.clsEstimation import EstimationCls
from respy_smm.auxiliary import smm_sample_f2py, get_initial_conditions
from respy_smm.clsLogging import logger_obj
from respy_smm import HUGE_FLOAT
from respy_smm import HUGE_INT
from norpy.adapter.smm_auxiliary import OPTIM_PARAS

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import dist_class_attributes
from norpy.python.shared.shared_constants import DATA_FORMATS_SIM
from respy.python.shared.shared_constants import DATA_LABELS_SIM
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import MISSING_INT
from norpy.model_spec import get_model_obj
import f2py_interface as respy_f2py


class SimulationBasedEstimationCls(EstimationCls):
    """This class manages the distribution of the use requests throughout the toolbox."""
    def __init__(self, init_file, moments_obs, weighing_matrix, get_moments, max_evals=HUGE_INT):

        super().__init__()

        self.model_obj = get_model_obj(init_file)

        # Creating a random data array also for the SMM routine allows to align a lot of the
        # designs across the two different estimation strategies.
        self.data_array = np.random.rand(8, 8)
        self.weighing_matrix = weighing_matrix
        self.get_moments = get_moments
        self.moments_obs = moments_obs
        self.max_evals = max_evals

        self.simulate_sample = None

        self.set_derived_attributes()


    def evaluate(self, free_params):
        """This method evaluates the criterion function for a candidate parametrization proposed
        by the optimizer."""
        self.model_obj = update_model_spec(free_params,self.model_obj,OPTIM_PARAS)
        #Do we need this assert statement ?
        assert np.all(np.isfinite(free_params))
        #Here we check our covariance matrix ?
        if not is_valid_covariance_matrix(x_all_econ_current[43:53]):
            msg = 'invalid evaluation due to lack of proper covariance matrix'
            logger_obj.record_abort_eval(msg)
            return HUGE_FLOAT

        array_sim = simulate(self.model_obj)
        moments_sim = self.get_moments(array_sim)
        stats_obs, stats_sim = [], []
        #Check whether this goes through in our model !
        for group in self.moments_sim.keys():
            for period in range(max(self.moments_sim[group].keys()) + 1):
                if period not in moments_sim[group].keys():
                    continue
                if period not in self.moments_obs[group].keys():
                    continue
                stats_obs.extend(self.moments_obs[group][period])
                stats_sim.extend(moments_sim[group][period])

        # We need to deal with the special case where it might happen that some moments for the
        # wage distribution are available in the observed dataset but not the simulated dataset.
        is_valid = len(stats_obs) == len(stats_sim) == len(np.diag(self.weighing_matrix))
        if is_valid:
            stats_diff = np.array(stats_obs) - np.array(stats_sim)
            fval = float(np.dot(np.dot(stats_diff, self.weighing_matrix), stats_diff))
        else:
            msg = 'invalid evaluation due to missing moments'
            logger_obj.record_abort_eval(msg)
            self.check_termination()
            fval = HUGE_FLOAT
        return fval

