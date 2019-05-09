import os

import pytest

from norpy.simulate.simulate import simulate, create_state_space,  return_immediate_rewards, backward_induction_procedure,simulate,return_simulated_shocks

from norpy.model_spec import get_model_obj, get_random_model_specification 

from norpy.norpy_config import PACKAGE_DIR


def test():
    """The function allows to run the tests from inside the interpreter."""
    current_directory = os.getcwd()
    os.chdir(PACKAGE_DIR)
    pytest.main()
    os.chdir(current_directory)
