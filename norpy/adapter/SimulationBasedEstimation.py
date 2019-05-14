"""
Initial sketch of the adapter class.
"""
import numpy as np

from norpy.model_spec import get_model_obj

HUGE_INT = 1000000


class SimulationBasedEstimationCls:
    """This class manages the distribution of the use requests throughout the toolbox."""

    def __init__(
        self,
        init_file,
        moments_obs,
        weighing_matrix,
        get_moments,
        update_model_spec,
        max_evals=HUGE_INT,
    ):

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

    def evaluate(self, free_params):
        """This method evaluates the criterion function for a candidate parametrization proposed
        by the optimizer."""
        self.model_obj = update_model_spec(free_params, self.model_obj, OPTIM_PARAS)
        # Do we need this assert statement ?
        assert np.all(np.isfinite(free_params))
        # Here we check our covariance matrix ?
        # Just deleted a check for valid covariance matrix
        array_sim = simulate(self.model_obj)
        moments_sim = self.get_moments(array_sim)
        stats_obs, stats_sim = [], []
        # Check whether this goes through in our model !
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
        is_valid = (
            len(stats_obs) == len(stats_sim) == len(np.diag(self.weighing_matrix))
        )
        if is_valid:
            stats_diff = np.array(stats_obs) - np.array(stats_sim)
            fval = float(np.dot(np.dot(stats_diff, self.weighing_matrix), stats_diff))
        else:
            msg = "invalid evaluation due to missing moments"
            logger_obj.record_abort_eval(msg)
            self.check_termination()
            fval = HUGE_FLOAT
        return fval
