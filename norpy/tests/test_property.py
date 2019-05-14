#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test the full module!!

Important note:
The grid tests are written under the constraint that the type shifts are zero !
I might want to change that in the medium run !
Also there is still considerable repetition in the selection procedure which has to be changed at some point !


"""

import numpy as np
import math


from norpy import (
    create_state_space,
    return_immediate_rewards,
    backward_induction_procedure,
    simulate,
    return_simulated_shocks,
)
from norpy import get_random_model_specification, get_model_obj
from norpy import create_state_space, return_immediate_rewards
from norpy import get_random_model_specification, get_model_obj


def random_model_object():
    model_object = get_model_obj(get_random_model_specification())
    return model_object


def input_output_state_space():

    model_object = random_model_object()
    state_space = create_state_space(model_object)

    period = np.random.randint(1, model_object.num_periods + 1)
    # Split up different cases
    type_ = np.random.randint(1, model_object.num_types + 1)
    edu_start = model_object.edu_range_start[
        np.random.randint(len(model_object.edu_range_start))
    ]
    exp_a = np.random.randint(period)
    edu_add = np.random.randint(min(period - exp_a, model_object.edu_max - edu_start))

    if exp_a == 0:
        if period == 1:
            lagged_choice = np.random.randint(2, 4)
        elif edu_add == period - 1 and period > 1:
            lagged_choice = 2
        elif edu_add == 0:
            lagged_choice = 3
        else:
            lagged_choice = np.random.randint(2, 4)

    elif edu_add == 0:
        if exp_a == period - 1:
            lagged_choice = 1
        else:

            lagged_choice = [1, 3][np.random.randint(2)]

    elif edu_add + exp_a == period - 1 and period > 1:
        lagged_choice = np.random.randint(1, 3)
    else:
        lagged_choice = np.random.randint(1, 4)

    manual = np.array([exp_a, edu_start + edu_add, lagged_choice, type_])

    state_space["states_all"] = state_space["states_all"][
        :, : state_space["max_states_period"], :
    ]
    return state_space["states_all"], manual, period


def test_state_space_1(input_output_state_space=input_output_state_space()):
    assert np.any(
        np.all(
            input_output_state_space[0][input_output_state_space[2] - 1]
            == input_output_state_space[1],
            axis=1,
        )
    )


def input_not_output_state_space():
    model_object = random_model_object()

    state_space = create_state_space(model_object)
    period = np.random.randint(1, model_object.num_periods + 1)
    # Split up different cases
    type_ = np.random.randint(1, model_object.num_types + 1)
    edu_start = model_object.edu_range_start[
        np.random.randint(len(model_object.edu_range_start))
    ]
    exp_a = np.random.randint(period)
    edu_add = np.random.randint(period - exp_a)
    exp_a == 0
    if exp_a == 0:
        lagged_choice = 1
    elif edu_add == 0 and period >= 2:
        lagged_choice = 2
        edu_start = min(
            model_object.edu_range_start
        )  # Here we have to be careful because only this case is not allowed
    else:
        lagged_choice = 300000

    state_space["states_all"] = state_space["states_all"][
        :, : state_space["max_states_period"], :
    ]
    manual = np.array([exp_a, edu_start + edu_add, lagged_choice, type_])
    return state_space["states_all"], manual, period, state_space["max_states_period"]


def test_state_space_2(input_not_output_state_space=input_not_output_state_space()):
    assert (
        np.any(
            np.all(
                input_not_output_state_space[0][
                    input_not_output_state_space[2] - 1
                ].reshape(input_not_output_state_space[3], 4)
                == input_not_output_state_space[1],
                axis=1,
            )
        )
        == False
    )


def input_output_size():
    model_object = random_model_object()

    state_space = create_state_space(model_object)

    state_space["states_all"] = state_space["states_all"][
        :, : state_space["max_states_period"], :
    ]

    max_period = np.where(
        state_space["states_number_period"] == state_space["max_states_period"]
    )
    return (
        state_space["states_all"],
        max_period,
        state_space["max_states_period"],
        state_space["states_number_period"],
    )


def test_state_space_3(input_output_size=input_output_size()):
    assert input_output_size[3].max() == input_output_size[2]


def input_output_dimension():
    model_object = get_model_obj(get_random_model_specification())
    state_space = create_state_space(model_object, True)

    state_space["states_all"] = state_space["states_all"][
        :, : state_space["max_states_period"] + 1, :
    ]
    period = np.random.randint(1, model_object.num_periods + 1)

    if period > 1:
        dim_period = int(
            ((((period) ** 2 + period) / 2) - (period))
            * 3
            * model_object.num_edu_start
            * model_object.num_types
        )
    else:
        dim_period = int(2 * model_object.num_edu_start * model_object.num_types)

    return (
        state_space["states_all"],
        state_space["max_states_period"],
        state_space["states_number_period"],
        period,
        dim_period,
        model_object.num_edu_start,
    )


def test_state_space_dimension(input_output_dimension=input_output_dimension()):
    np.testing.assert_array_almost_equal(
        input_output_dimension[2][input_output_dimension[3] - 1],
        input_output_dimension[4],
    )


def input_output_immediate_rewards_home():

    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    state_space = create_state_space(model_object)
    immediate_rewards = return_immediate_rewards(model_object, state_space)

    # Randomly draw a position on the state space
    period_to_check = np.random.randint(1, model_object.num_periods + 1)
    k_to_check = np.random.randint(
        state_space["states_number_period"][period_to_check - 1]
    )
    # - one to modify fortran indexing to python indexing

    states_to_check = state_space["states_all"][period_to_check - 1, k_to_check]
    # initialize manual_result
    manually_calculated_result = 0
    # Calculate common part of home rewards
    if states_to_check[1] < 12:
        manually_calculated_result = 0
    elif states_to_check[1] < 15:
        manually_calculated_result = model_object.coeffs_common[0]
    else:
        manually_calculated_result = (
            model_object.coeffs_common[0] + model_object.coeffs_common[1]
        )
    # Calculate specific part of home rewards
    if 3 <= period_to_check < 6:

        manually_calculated_result = (
            manually_calculated_result
            + model_object.coeffs_home[0]
            + model_object.coeffs_home[1]
        )
    elif period_to_check >= 6:
        manually_calculated_result = (
            manually_calculated_result
            + model_object.coeffs_home[0]
            + model_object.coeffs_home[2]
        )
    else:
        manually_calculated_result = (
            manually_calculated_result + model_object.coeffs_home[0]
        )

    return immediate_rewards, period_to_check, manually_calculated_result, k_to_check


def test_immediate_rewards_home(
    input_output_immediate_rewards_home=input_output_immediate_rewards_home()
):
    np.testing.assert_array_almost_equal(
        input_output_immediate_rewards_home[0][
            input_output_immediate_rewards_home[1] - 1,
            input_output_immediate_rewards_home[3],
            2,
        ],
        input_output_immediate_rewards_home[2],
    )


def input_output_immediate_rewards_educ():

    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    state_space = create_state_space(model_object)
    immediate_rewards = return_immediate_rewards(model_object, state_space)

    # Randomly draw a position on the state space
    period_to_check = np.random.randint(1, model_object.num_periods + 1)
    k_to_check = np.random.randint(
        state_space["states_number_period"][period_to_check - 1]
    )
    # - one to modify fortran indexing to python indexing

    states_to_check = state_space["states_all"][period_to_check - 1, k_to_check]
    # initialize manual_result
    manually_calculated_result = model_object.coeffs_edu[0] + model_object.coeffs_edu[
        5
    ] * (period_to_check - 1)
    # Calculate specific part of edu rewards
    if states_to_check[1] < 9:
        if states_to_check[2] != 2:
            manually_calculated_result = (
                model_object.coeffs_edu[6]
                + model_object.coeffs_edu[3]
                + manually_calculated_result
            )
        else:
            manually_calculated_result = (
                model_object.coeffs_edu[6] + manually_calculated_result
            )

    elif states_to_check[1] < 12:
        if states_to_check[2] != 2:
            manually_calculated_result = (
                model_object.coeffs_edu[3] + manually_calculated_result
            )
        else:
            manually_calculated_result = manually_calculated_result

    elif states_to_check[1] < 15:
        if states_to_check[2] != 2:
            manually_calculated_result = (
                model_object.coeffs_edu[4]
                + model_object.coeffs_edu[1]
                + manually_calculated_result
            )
        else:
            manually_calculated_result = (
                manually_calculated_result + model_object.coeffs_edu[1]
            )

    else:
        if states_to_check[2] != 2:
            manually_calculated_result = (
                manually_calculated_result
                + model_object.coeffs_edu[2]
                + model_object.coeffs_edu[1]
                + model_object.coeffs_edu[4]
            )
        else:
            manually_calculated_result = (
                manually_calculated_result
                + model_object.coeffs_edu[1]
                + model_object.coeffs_edu[2]
            )
    # Calculate common part of home rewards

    if states_to_check[1] < 12:
        manually_calculated_result = manually_calculated_result
    elif states_to_check[1] < 15:
        manually_calculated_result = (
            model_object.coeffs_common[0] + manually_calculated_result
        )
    else:
        manually_calculated_result = (
            model_object.coeffs_common[0]
            + model_object.coeffs_common[1]
            + manually_calculated_result
        )

    return immediate_rewards, period_to_check, k_to_check, manually_calculated_result


def test_immediate_rewards_educ(
    input_output_immediate_rewards_educ=input_output_immediate_rewards_educ()
):
    np.testing.assert_array_almost_equal(
        np.array(
            [
                input_output_immediate_rewards_educ[0][
                    input_output_immediate_rewards_educ[1] - 1,
                    input_output_immediate_rewards_educ[2],
                    1,
                ]
            ]
        ),
        np.array([input_output_immediate_rewards_educ[3]]),
    )


def input_output_immediate_rewards_occupation():

    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    state_space = create_state_space(model_object)
    immediate_rewards = return_immediate_rewards(model_object, state_space)

    # Randomly draw a position on the state space
    period_to_check = np.random.randint(1, model_object.num_periods + 1)

    k_to_check = np.random.randint(
        state_space["states_number_period"][period_to_check - 1]
    )
    # - one to modify fortran indexing to python indexing

    states_to_check = state_space["states_all"][period_to_check - 1, k_to_check]
    # initialize manual_result
    manually_calculated_result_exponent = (
        model_object.coeffs_work[0]
        + model_object.coeffs_work[1] * states_to_check[1]
        + model_object.coeffs_work[2] * states_to_check[0]
        + model_object.coeffs_work[6] * (period_to_check - 1)
        + model_object.coeffs_work[3] * (states_to_check[0] ** 2 / 100)
    )

    if states_to_check[1] >= 12:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + model_object.coeffs_work[4]
        )

    if states_to_check[1] >= 15:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + model_object.coeffs_work[5]
        )

    if period_to_check < 3:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + model_object.coeffs_work[7]
        )
    if states_to_check[0] > 0:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + model_object.coeffs_work[8]
        )
    if states_to_check[2] == 1:
        manually_calculated_result_exponent = (
            manually_calculated_result_exponent + model_object.coeffs_work[9]
        )

    manually_calculated_result = (
        math.exp(manually_calculated_result_exponent) + model_object.coeffs_work[10]
    )
    if states_to_check[2] != 1:
        if states_to_check[0] == 0:
            manually_calculated_result = (
                manually_calculated_result + model_object.coeffs_work[12]
            )

        else:
            manually_calculated_result = (
                manually_calculated_result + model_object.coeffs_work[11]
            )

    # Calculate specific part of edu rewards
    # Calculate common part of home rewards

    if states_to_check[1] < 12:
        manually_calculated_result = manually_calculated_result
    elif states_to_check[1] < 15:
        manually_calculated_result = (
            model_object.coeffs_common[0] + manually_calculated_result
        )
    else:
        manually_calculated_result = (
            model_object.coeffs_common[0]
            + model_object.coeffs_common[1]
            + manually_calculated_result
        )
    # Calculate specific part of home rewards

    return immediate_rewards, period_to_check, k_to_check, manually_calculated_result


def test_immediate_rewards_occupation(
    input_output_immediate_rewards_occupation=input_output_immediate_rewards_occupation()
):

    np.testing.assert_array_almost_equal(
        np.array(
            [
                input_output_immediate_rewards_occupation[0][
                    input_output_immediate_rewards_occupation[1] - 1,
                    input_output_immediate_rewards_occupation[2],
                    0,
                ]
            ]
        ),
        np.array([input_output_immediate_rewards_occupation[3]]),
    )


def random_model_object():
    model_object = get_model_obj(get_random_model_specification())
    return model_object


def set_up_last_period():

    # We want to set up a basic testing infrastructure for the state space creation.
    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    state_space = create_state_space(model_object)
    immediate_rewards = return_immediate_rewards(model_object, state_space)

    periods_draws_emax = return_simulated_shocks(model_object, simulation=False)
    periods_emax = backward_induction_procedure(
        model_object, state_space, immediate_rewards, periods_draws_emax
    )

    k_to_check = np.random.randint(1, state_space["states_number_period"][-1])

    draws_to_check = periods_draws_emax[model_object.num_periods - 1]

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


def test_last_period_value_func(set_up_last_period=set_up_last_period()):
    np.testing.assert_array_almost_equal(
        np.array([set_up_last_period[0]]),
        np.array(
            [set_up_last_period[1][set_up_last_period[2] - 1, set_up_last_period[3]]]
        ),
        decimal=1,
    )


def set_up_any_period():
    model_object = get_model_obj(
        get_random_model_specification(
            constr={"num_types": 4, "type_spec_shifts": np.zeros(12).reshape(4, 3)}
        )
    )
    state_space = create_state_space(model_object)
    immediate_rewards = return_immediate_rewards(model_object, state_space)
    periods_draws_emax = return_simulated_shocks(model_object, simulation=False)
    periods_emax = backward_induction_procedure(
        model_object, state_space, immediate_rewards, periods_draws_emax
    )
    period_to_check = np.random.randint(0, model_object.num_periods - 1)
    k_to_check = np.random.randint(
        0, state_space["states_number_period"][period_to_check] - 1
    )

    draws_to_check = periods_draws_emax[period_to_check]

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


def test_value_func_general(set_up_any_period=set_up_any_period()):

    np.testing.assert_array_almost_equal(
        np.array([set_up_any_period[0]]),
        np.array([set_up_any_period[1][set_up_any_period[2], set_up_any_period[3]]]),
        decimal=1,
    )


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


def test_simulation_descriptives(init_simulation=init_simulation()):
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


def test_simulation_with_high_work_rewards(
    init_simulation_huge_rewards=init_simulation_huge_rewards()
):
    assert (
        init_simulation_huge_rewards[0][
            (init_simulation_huge_rewards[3] - 1) * init_simulation_huge_rewards[1]
            + init_simulation_huge_rewards[2],
            2,
        ]
        == 1
    )
    # Todo:check for high common, check agent identifier, check period ident
