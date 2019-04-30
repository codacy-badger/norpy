"""
I clipped shocks to be less than 10 mio for testing purposes.
There might be a m ore elgenat way to do this!


"""

import pathlib
import yaml
import copy

import numpy as np
from scipy.stats import invwishart

# The typing module requires the use of
# Python 3.6 or higher.
import typing

# Question how would I proceed with the intermediate model objects?
class ModelSpec(typing.NamedTuple):
    """Model specification.

    Class that contains all the information required to simulate
    a specified AR(1) process. It is the one an central place
    that contains this information throughout. It is an extended
    version of a namedtuple and thus ensures that model specification
    remains immutable.

    Attributes:
        rho: a float indicating the degree of serial correlation
        periods: an integer for the length of the time horizon
    """

    # We need to define all fields and their type right at the
    # beginning. These cannot be changed after initialization
    # and no other fields added dynamically without raising
    # an error.
    # I keep the variable names in the format we used so far in the runner.py files !

    num_periods: int
    num_types: int
    num_draws_emax: int
    num_edu_start: int
    edu_spec_start: np.ndarray
    edu_spec_max: int
    shocks_cov: np.ndarray
    type_spec_shifts: np.ndarray
    coeffs_common: np.ndarray
    coeffs_work: np.ndarray
    coeffs_home: np.ndarray
    coeffs_edu: np.ndarray
    num_agents_sim: int
    delta: float
    # We make some of the private methods of the base class
    # public.
    def as_dict(self):
        return self._asdict()

    def replace(self, *args, **kwargs):
        return self._replace(*args, **kwargs)

    # We write wrappers for common use cases.
    def copy(self):
        return copy.deepcopy(self)

    # We specify some methods that have no counterpart in the
    # base class.
    def to_yaml(self, fname="test.yml"):
        with open(fname, "w") as out_file:
            yaml.dump(self._asdict(), out_file)

    def validate(self):
        """Validation of model specification.

        All validation is done here and no further checks are
        necessary later in the program for the immutable
        parameters describing the model. The for-loop ensures that
        all fields require exlicit checks.
        """
        for field in self._fields:
            attr = getattr(self, field)
            if field in [
                "num_periods",
                "num_types",
                "num_draws_emax",
                "num_edu_start",
                "edu_spec_max",
                "num_agents_sim",
            ]:
                assert isinstance(attr, int)
                assert attr > 0
            elif field == "delta":
                assert isinstance(attr, float)
            elif field in [
                "coeffs_common",
                "coeffs_work",
                "coeffs_home",
                "coeffs_edu",
                "type_spec_shifts",
                "edu_spec_start",
                "shocks_cov"
            ]:
                assert isinstance(attr, np.ndarray)
            else:
                raise NotImplementedError("validation of {:} missing".format(field))

    # We ovewrite some of the intrinsic __dunder__ methods to increase
    # usability of our class.
    def __repr__(self):
        """Provides a string representation of the model speficiation for
        quick visual inspection.
        """
        str_ = ""
        for field in self._fields:
            str_ += "{:}: {:}\n".format(field, getattr(self, field))
        return str_

    def __eq__(self, other):
        """Check the equality of two model specifications.

        Returns true if two model specifications have the same fields defined
        and all have the same value.

        Args:
            other: A ModelSpec class instance.

        Returns:
            A boolean corresponding to equality of specifications.
        """
        assert isinstance(other, type(self))
        assert set(spec_1._fields) == set(spec_2._fields)
        for field in self._fields:
            if getattr(self, field) != getattr(other, field):
                return False
        return True

    def __ne__(self, other):
        """Check the inequality of two model specification."""
        return not self.__eq__(other)


def get_random_model_specification(constr=None):
    """Create a random model specification

    Creates a random specification of the model which is useful
    for testing the robustness of implementation and testing
    in general.

    Args:
        constr: A dictionary that contains the requested constrains.
            The keys correspond to the field that is set to the value
            field.

            {'periods': 4, 'rho': 0.4}
    """

    def process_constraints(constr):
        """Impose a constraint on initialization dictionary.

        This function processes all constraints passed in by the user
        for the random model specification.

        Args:
            constr: A dictionary which contains the constraints.
        """
        if constr is None:
            constr = dict()

        list_of_var = [
            "num_types",
            "num_periods",
            "num_agents_sim",
            "edu_spec_max",
            "edu_spec_start",
            "num_edu_start",
            "num_draws_emax",
            "delta",
            "coeffs_common",
            "coeffs_home",
            "coeffs_edu",
            "coeffs_work",
            "type_spec_shifts",
            "shocks_cov"

        ]
        for x in list_of_var:
            if x in list(constr.keys()):
                init_dict[x] = constr[x]

    init_dict = dict()
    init_dict["num_types"] = np.random.randint(1, 5)
    init_dict["num_periods"] = np.random.randint(2, 10)
    init_dict["num_edu_start"] = np.random.randint(1, 4)
    init_dict["edu_spec_max"] = np.random.randint(15, 25)
    init_dict["edu_spec_start"] = np.random.choice(
        range(1, 10), size=init_dict["num_edu_start"], replace=False
    )
    init_dict["num_agents_sim"] = np.random.randint(1, 50)

    init_dict["num_draws_emax"] = np.random.randint(1, 50)
    init_dict["delta"] = np.random.uniform(0.01, 0.99)
    init_dict["coeffs_common"] = np.random.uniform(size=2)
    init_dict["coeffs_home"] = np.random.uniform(size=3)
    init_dict["coeffs_edu"] = np.random.uniform(size=7)
    init_dict["coeffs_work"] = np.random.uniform(size=13)
    init_dict["type_spec_shifts"] = np.random.normal(
        size=init_dict["num_types"] * 3
    ).reshape((init_dict["num_types"], 3))
    init_dict["shocks_cov"] = invwishart.rvs(df=3, scale=np.identity(3))
    args = (
        np.zeros(3),
        init_dict["shocks_cov"],
        (init_dict["num_periods"], init_dict["num_draws_emax"]),
    )

    process_constraints(constr)

    return init_dict


def get_model_obj(source=None, constr=None):
    """Get model specification.

    This is a factory method to create a model spefication from
    a variety of differnt input types.

    Args:
        input: str, dictionary, None specifying the input for
            for the model specification.
        constr: A dictionary with the constraints imposed
            on a random initialization file.

    Returns:
        An instance of the ModelSpec class with the model
        specification.
    """
    # We want to enforce the use of Path objects going forward.
    if isinstance(source, str):
        source = pathlib.Path(source)

    if isinstance(source, dict):
        model_spec = ModelSpec(**source)
    elif isinstance(source, pathlib.PosixPath):
        model_spec = ModelSpec(**yaml.load(open(source, "r"), Loader=yaml.FullLoader))
    elif source is None:
        model_spec = ModelSpec(**get_random_model_specification(constr))
    else:
        raise NotImplementedError

    # We validate our model specification once and for all.
    # Unfortunately, there is no way to do so at class
    # initialization as we cannot override the __new__
    # method.
    model_spec.validate()

    return model_spec
