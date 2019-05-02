"""
This module contains the second property testing battery
"""
import os
import sys



from numpy import f2py
import pandas as pd
import numpy as np
import pytest
import math
import pytest

from norpy.simulate.simulate import (
    create_state_space,
    return_immediate_rewards,
    backward_induction_procedure,
    simulate,
)
from norpy.model_spec import get_random_model_specification, get_model_obj

def random_model_object():
    model_object = get_model_obj(get_random_model_specification())
    return model_object


@pytest.fixture(params=[str(x) for x in list(range(10))])
def set_up_last_period():

    # We want to set up a basic testing infrastructure for the state space creation.
    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    state_space = create_state_space(model_object)
    immediate_rewards = return_immediate_rewards(model_object, state_space)
    
    periods_emax = backward_induction_procedure(
        model_object, state_space, immediate_rewards,periods_draws_emax
    )

    k_to_check = np.random.randint(1, state_space["states_number_period"][-1])

    draws_to_check = model_object.periods_draws_emax[model_object.num_periods - 1]

    immediate_rewards_last_period = immediate_rewards[
        model_object.num_periods - 1, k_to_check
    ]
    state_to_check = state_space["states_all"][model_object.num_periods - 1, k_to_check]
    aux = np.zeros(model_object.num_draws_emax * 3).reshape(
        model_object.num_draws_emax, 3
    )
    # des is der common anteil
    common = 0
    if state_to_check[1] >= 12:
        common = model_object.coeffs_common[0] + common
    else:
        common = common

    if state_to_check[1] >= 15:
        common = model_object.coeffs_common[1] + common
    else:
        common = common

    # des is der genral teil
    general = model_object.coeffs_work[10]

    if state_to_check[0] == 0:
        general = general + model_object.coeffs_work[12]

    elif state_to_check[2] != 1:
        general = general + model_object.coeffs_work[11]
    else:
        general = general

    for x in range(model_object.num_draws_emax):
        aux[x, 0] = (
            (immediate_rewards_last_period[0] - common - general) * draws_to_check[x, 0]
            + common
            + general
        )
        aux[x, 1] = immediate_rewards_last_period[1] + draws_to_check[x, 1]
        aux[x, 2] = immediate_rewards_last_period[2] + draws_to_check[x, 2]

    out = np.zeros(model_object.num_draws_emax)

    for i in range(model_object.num_draws_emax):
        out[i] = aux[i, :].max()

    manual_result = out.sum() / model_object.num_draws_emax

    return manual_result, periods_emax, model_object.num_periods, k_to_check


def test_last_period_value_func(set_up_last_period):

    np.testing.assert_array_almost_equal(
        np.array([set_up_last_period[0]]),
        np.array(
            [set_up_last_period[1][set_up_last_period[2] - 1, set_up_last_period[3]]]
        ),
        decimal=1,
    )


#####Now do the same for any period!


@pytest.fixture(params=[str(x) for x in list(range(10))])
def set_up_any_period():
    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    state_space = create_state_space(model_object)
    immediate_rewards = return_immediate_rewards(model_object, state_space)
    periods_emax = backward_induction_procedure(
        model_object, state_space, immediate_rewards
    )
    period_to_check = np.random.randint(0, model_object.num_periods - 1)
    k_to_check = np.random.randint(
        0, state_space["states_number_period"][period_to_check] - 1
    )

    draws_to_check = model_object.periods_draws_emax[period_to_check]

    immediate_rewards_last_period = immediate_rewards[period_to_check, k_to_check]
    state_to_check = state_space["states_all"][period_to_check, k_to_check]
    aux = np.zeros(model_object.num_draws_emax * 3).reshape(
        model_object.num_draws_emax, 3
    )
    # des is der common anteil
    common = 0
    if state_to_check[1] >= 12:
        common = model_object.coeffs_common[0] + common
    else:
        common = common

    if state_to_check[1] >= 15:
        common = model_object.coeffs_common[1] + common
    else:
        common = common

    # des is der genral teil
    general = model_object.coeffs_work[10]

    if state_to_check[0] == 0:
        general = general + model_object.coeffs_work[12]

    elif state_to_check[2] != 1:
        general = general + model_object.coeffs_work[11]
    else:
        general = general

    ##Obtain next periods states
    next_period_1 = state_space["mapping_state_idx"][
        period_to_check + 1,
        state_to_check[0] + 1,
        state_to_check[1],
        0,
        state_to_check[3] - 1,
    ]
    next_period_2 = state_space["mapping_state_idx"][
        period_to_check + 1,
        state_to_check[0],
        state_to_check[1] + 1,
        1,
        state_to_check[3] - 1,
    ]
    next_period_3 = state_space["mapping_state_idx"][
        period_to_check + 1,
        state_to_check[0],
        state_to_check[1],
        2,
        state_to_check[3] - 1,
    ]

    for x in range(model_object.num_draws_emax):
        aux[x, 0] = (
            (immediate_rewards_last_period[0] - common - general) * draws_to_check[x, 0]
            + common
            + general
            + model_object.delta * periods_emax[period_to_check + 1, next_period_1]
        )
        aux[x, 1] = (
            immediate_rewards_last_period[1]
            + draws_to_check[x, 1]
            + model_object.delta * periods_emax[period_to_check + 1, next_period_2]
        )
        aux[x, 2] = (
            immediate_rewards_last_period[2]
            + draws_to_check[x, 2]
            + model_object.delta * periods_emax[period_to_check + 1, next_period_3]
        )

    out = np.zeros(model_object.num_draws_emax)

    for i in range(model_object.num_draws_emax):
        out[i] = aux[i, :].max()

    manual_result = out.sum() / model_object.num_draws_emax

    return manual_result, periods_emax, period_to_check, k_to_check


def test_value_func_general(set_up_any_period):

    np.testing.assert_array_almost_equal(
        np.array([set_up_any_period[0]]),
        np.array([set_up_any_period[1][set_up_any_period[2], set_up_any_period[3]]]),
        decimal=1,
    )


@pytest.fixture(params=[str(x) for x in list(range(10))])
def init_simulation(constr=False):
    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    dat = simulate(model_object)
    agent_to_check = np.random.randint(0, model_object.num_agents_sim)
    period_to_check = np.random.randint(0, model_object.num_periods - 1)

    # print(dat[num_periods*(agent_to_check-1)+period_to_check,2])
    return (
        dat,
        model_object.num_periods,
        model_object.num_agents_sim,
        period_to_check,
        model_object.delta,
        agent_to_check,
    )


def test_simulation_descriptives(init_simulation):
    assert init_simulation[0].shape == (init_simulation[1] * init_simulation[2], 23)
    assert init_simulation[0][(init_simulation[5] - 1) * init_simulation[1], 4] == 0
    assert init_simulation[0][
        (init_simulation[5] - 1) * init_simulation[1] + init_simulation[3], 2
    ] in [1, 2, 3]
    assert (
        init_simulation[0][
            (init_simulation[5] - 1) * init_simulation[1] + init_simulation[3], 17
        ]
        == init_simulation[4]
    )

    # assert dat[num_periods*(agent_to_check-1)+period_to_check,2]==1


# Check for arr


@pytest.fixture(params=[str(x) for x in list(range(10))])
def init_simulation_huge_rewards():
    model_object = get_model_obj(
        get_random_model_specification(
            constr={
                "num_types": 4,
                "type_spec_shifts": np.zeros(12).reshape(4, 3),
                "coeffs_work": np.concatenate(
                    (np.random.uniform(size=10), np.array([2000000000000000, 0, 0]))
                ),
            }
        )
    )
    dat = simulate(model_object)
    agent_to_check = np.random.randint(0, model_object.num_agents_sim)
    period_to_check = np.random.randint(0, model_object.num_periods - 1)
    # print(dat[num_periods*(agent_to_check-1)+period_to_check,2])
    return dat, model_object.num_periods, period_to_check, agent_to_check


def test_simulation_with_high_work_rewards(init_simulation_huge_rewards):
    assert (
        init_simulation_huge_rewards[0][
            (init_simulation_huge_rewards[3] - 1) * init_simulation_huge_rewards[1]
            + init_simulation_huge_rewards[2],
            2,
        ]
        == 1
    )
    # Todo:check for high common, check agent identifier, check period ident
