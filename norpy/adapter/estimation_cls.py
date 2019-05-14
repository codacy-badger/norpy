"""
What should this fiel contain ?
We want to specify a base class that can be used for 
estimation! 
"""
from collections import OrderedDict
import numpy as np


class EstimationCls(object):
    def __init__(self):

        # We have several attributes that are shared across the children classes. These are
        # declared here for convenience.
        self.x_all_econ = OrderedDict()
        self.criterion_function = None
        self.logging_container = None
        self.x_free_econ_start = None
        self.fval = OrderedDict()
        self.data_array = None
        self.paras_free = None
        self.num_paras = None
        self.mpi_setup = None
        self.is_start = True
        self.num_evals = 0
        self.num_steps = 0

        # We need to set up containers for logging our progress.
        self.info = OrderedDict()
        self.info["is_step"] = np.empty(0, dtype=np.bool)
        self.info["x_econ_all"] = np.empty(0)
        self.info["fval"] = np.empty(0)

    def construct_complete_parameters(self, x_free_econ):
        """
        This we just need to initialize ?
        """
        x_all_econ_current = self.x_all_econ["start"].copy()
        x_all_econ_current[self.paras_free] = x_free_econ
        return x_all_econ_current

    def wrapping_up_evaluation(self, x_all_econ_current, fval):
        """
        Is this for paramter updating ???
        That should stay equal
        """
        self.num_evals += 1

        is_step = self.fval["step"] > fval
        is_current = True

        if self.is_start:
            self.fval["start"] = fval
            self.is_start = False

        if is_current:
            self.x_all_econ["current"] = x_all_econ_current
            self.fval["current"] = fval
        # Was sagt uns dieses is_step hier ?
        if is_step:
            self.x_all_econ["step"] = x_all_econ_current
            self.fval["step"] = fval
            self.num_steps += 1

    def check_termination(self):
        is_termination = self.num_evals >= self.max_evals
        if is_termination:
            self.terminate()

    def terminate(self, is_gentle=False):
        if hasattr(self.mpi_setup, "Bcast"):
            MPI = get_mpi()
            cmd = np.array(1, dtype="int32")
            self.mpi_setup.Bcast([cmd, MPI.INT], root=MPI.ROOT)
        if not is_gentle:
            raise StopIteration

    def set_derived_attributes(self):

        optim_paras = self.respy_base.get_attr("optim_paras")
        for k in ["start", "step", "current"]:
            self.x_all_econ[k] = respy_spec_old_to_new(optim_paras)
            self.fval[k] = HUGE_FLOAT

        self.num_paras = len(self.x_all_econ["start"])

        self.logging_container = np.tile(np.nan, (self.num_paras + 1, 3))
        self.paras_free = ~np.array(optim_paras["paras_fixed"])

        self.x_free_econ_start = self.x_all_econ["start"][self.paras_free]

    def set_up_baseline(self, periods_draws_emax, periods_draws_prob):
        """This method distributes the basic information to the slave processes."""
        num_periods, num_draws_emax, num_draws_sim = dist_class_attributes(
            self.respy_base, "num_periods", "num_draws_emax", "num_draws_sim"
        )

        # In the case of SMM these are not required and we simply create random numbers.
        if periods_draws_prob is None:
            periods_draws_prob = np.random.randn(num_periods, num_draws_prob, 4)

        for i in range(num_periods):
            for j in range(num_draws_emax):
                self.mpi_setup.Bcast(
                    [periods_draws_emax[i, j, :], MPI.DOUBLE], root=MPI.ROOT
                )

        for i in range(num_periods):
            for j in range(num_draws_sim):
                self.mpi_setup.Bcast(
                    [periods_draws_prob[i, j, :], MPI.DOUBLE], root=MPI.ROOT
                )

        # This is relevant for the SMM routine.
        for i in range(self.data_array.shape[0]):
            self.mpi_setup.Bcast([self.data_array[i, :], MPI.DOUBLE], root=MPI.ROOT)
