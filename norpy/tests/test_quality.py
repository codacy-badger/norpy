"""This module contains test that check the code quality of the package."""
import subprocess

import numpy as np
import pytest

from norpy.norpy_config import PACKAGE_DIR


@pytest.mark.skip(reason="requires initial cleanup")
def test1():
    """This test runs flake8 to ensure the code quality."""
    np.testing.assert_equal(subprocess.call(["flake8"], shell=True, cwd=PACKAGE_DIR), 0)
