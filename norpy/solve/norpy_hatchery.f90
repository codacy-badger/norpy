!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_create_state_space(states_all, states_number_period, mapping_state_idx, &
        max_states_period, num_periods, num_types, edu_range_start, edu_max, min_idx_int, &
        test_indication_optional)

    USE lib_norpy

    IMPLICIT NONE

    !/* dummy arguments        */

    INTEGER, INTENT(OUT) :: mapping_state_idx(num_periods, num_periods, min_idx_int, 3, num_types)
    INTEGER, INTENT(OUT) :: states_all(num_periods, 500000, 4)
    INTEGER, INTENT(OUT) :: states_number_period(num_periods)
    INTEGER, INTENT(OUT) :: max_states_period

    INTEGER, INTENT(IN) :: edu_range_start(:)
    INTEGER, INTENT(IN) :: min_idx_int
    INTEGER, INTENT(IN) :: num_periods
    INTEGER, INTENT(IN) :: num_types
    INTEGER, INTENT(IN) :: edu_max

    LOGICAL, INTENT(IN), OPTIONAL :: test_indication_optional

    !/* local variables        */

    LOGICAL :: test_indication

    INTEGER :: choice_lagged
    INTEGER :: num_edu_start
    INTEGER :: edu_start
    INTEGER :: edu_add
    INTEGER :: period
    INTEGER :: type_
    INTEGER :: exp
    INTEGER :: k
    INTEGER :: j

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------

    ! TODO: We always want the first call in the F2PY wrappers to go to the
    ! construction of the model_spec which whose attributes are then accessed
    ! downstram only by %, eg. model_spec%edu_start, etc.

    !Initialize test indication variable
    IF (PRESENT(test_indication_optional) .EQV. .TRUE.) THEN
        test_indication = test_indication_optional
    ELSE
        test_indication = .FALSE.
    END IF

    ! Construct auxiliary objects
    num_edu_start = SIZE(edu_range_start)

    ! Initialize output
    states_number_period = MISSING_INT
    mapping_state_idx = MISSING_INT
    states_all = MISSING_INT

    ! ! Construct state space by periods
    DO period = 1, (num_periods)

        ! Count admissible realizations of state space by period
        k = 0

        ! Loop over all types.
        DO type_ = 1, num_types

            ! Loop over all initial level of schooling
            DO j = 1, num_edu_start

                edu_start = edu_range_start(j)

                ! Loop over all admissible work experiences for Occupation A
                DO exp = 0, num_periods

                    ! Loop over all admissible additional education levels
                    DO edu_add = 0, num_periods

                        IF (edu_add + exp  .GT. period - 1) CYCLE

                        ! Agent cannot attain more additional education than (EDU_MAX - EDU_START).
                        IF (edu_add .GT. (edu_max - edu_start)) CYCLE

                        DO choice_lagged = 1, 3

                            IF (period .GT. one_int) THEN

                                ! (0, 1) Whenever an agent has only worked in Occupation A, then choice_lagged cannot be anything other than one.
                                IF ((choice_lagged .NE. one_int) .AND. (exp .EQ. period - 1)) CYCLE

                                ! (0, 3) Whenever an agent has only acquired additional education, then choice_lagged cannot be  anything other than two.
                                IF ((choice_lagged .NE. two_int) .AND. (edu_add .EQ. period - 1)) CYCLE

                                ! (0, 4) Whenever an agent has not acquired any additional education and we are not in the first period, then lagged education cannot take a value of three.
                                IF ((choice_lagged .EQ. two_int) .AND. (edu_add .EQ. zero_int)) CYCLE

                                !Whenever an agent has only worked or has only stayed in school he cannot have home as lagged choice
                                IF ((choice_lagged .EQ. three_int) .AND. (edu_add + exp  .EQ. period - 1)) CYCLE

                            END IF

                            ! (1, 1) In the first period individual either were in school the previous period as well or at home. The cannot have any work experience.
                            IF (period .EQ. one_int) THEN
                                IF (choice_lagged .EQ. one_int)  CYCLE

                            END IF

                            ! (2, 1) An individual that has never worked in Occupation A cannot have a that lagged activity.
                            IF ((choice_lagged .EQ. one_int) .AND. (exp .EQ. zero_int)) CYCLE

                            ! (3, 1) An individual that has never worked in Occupation B cannot have a that lagged activity.

                            IF (test_indication .EQV. .FALSE.) THEN
                                IF (mapping_state_idx(period, exp + 1, edu_start + edu_add + 1, choice_lagged, type_) &
                                        .NE. MISSING_INT) CYCLE

                            END IF

                            mapping_state_idx(period, exp + 1, edu_start + edu_add + 1, choice_lagged, type_) = k
                            states_all(period, k + 1, :) = (/ exp, edu_start + edu_add, choice_lagged, type_ /)

                            ! Update count
                            k = k + 1

                        END DO

                    END DO

                END DO

            END DO

        END DO

        states_number_period(period) = k

    END DO

    max_states_period = MAXVAL(states_number_period)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_calculate_immediate_rewards(periods_rewards_systematic, num_periods, &
        states_number_period, states_all, max_states_period, coeffs_common, coeffs_work, &
        coeffs_edu, coeffs_home, type_spec_shifts)

    USE lib_norpy

    IMPLICIT NONE

    !/* dummy arguments   */

    DOUBLE PRECISION, INTENT(OUT) :: periods_rewards_systematic(num_periods, max_states_period, 3)

    DOUBLE PRECISION, INTENT(IN) :: type_spec_shifts(:, :)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_home(3)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_edu(7)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_work(13)

    INTEGER, INTENT(IN) :: states_number_period(:)
    INTEGER, INTENT(IN) :: states_all(:, :, :)
    INTEGER, INTENT(IN) :: max_states_period
    INTEGER, INTENT(IN) :: num_periods

    !/* local variables        */

    INTEGER :: covars_home(3)
    INTEGER :: covars_edu(7)
    INTEGER :: choice_lagged
    INTEGER :: period
    INTEGER :: type_
    INTEGER :: exp
    INTEGER :: edu
    INTEGER :: k
    INTEGER :: i

    DOUBLE PRECISION :: rewards_general
    DOUBLE PRECISION :: rewards_common
    DOUBLE PRECISION :: rewards(3)
    DOUBLE PRECISION :: wages

    TYPE(MODEL_SPECIFICATION) :: model_spec
    TYPE(COVARIATES_DICT) :: covariates

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------

    ! TODO: This is the placeholder for the more involved complete setup of the
    ! model object to come.

    model_spec%num_periods = num_periods
    model_spec%coeffs_common = coeffs_common
    model_spec%coeffs_work = coeffs_work
    model_spec%coeffs_home = coeffs_home
    model_spec%coeffs_edu = coeffs_edu

    ! TODO: A selected few need to be allocated first.
    ALLOCATE(model_spec%type_shifts(SIZE(type_spec_shifts, 1), 4))
    model_spec%type_shifts = type_spec_shifts

    periods_rewards_systematic = MISSING_FLOAT

    ! Calculate systematic instantaneous rewards
    DO period = model_spec%num_periods, 1, -1

        DO k = 1, (states_number_period(period))

            ! Distribute state space
            exp = states_all(period, k, 1)
            edu = states_all(period, k, 2)
            choice_lagged = states_all(period, k, 3)
            type_ = states_all(period, k, 4)

            ! Construct auxiliary information
            covariates = construct_covariates(exp, edu, choice_lagged, type_, period)

            ! Calculate common and general rewards component.
            rewards_general = calculate_rewards_general(covariates, model_spec%coeffs_work(11:13))
            rewards_common = calculate_rewards_common(covariates, model_spec)

            ! Calculate the systematic part of working.
            wages = calculate_wages_systematic(covariates, model_spec)
            rewards(1) = wages + rewards_general

            ! Calculate systematic part of schooling utility
            covars_edu(1) = one_int
            covars_edu(2) = covariates%hs_graduate
            covars_edu(3) = covariates%co_graduate
            covars_edu(4) = covariates%is_return_not_high_school
            covars_edu(5) = covariates%is_return_high_school
            covars_edu(6) = covariates%period - one_int
            covars_edu(7) = covariates%is_mandatory

            rewards(2) = DOT_PRODUCT(covars_edu, model_spec%coeffs_edu)

            ! Calculate systematic part of HOME
            covars_home(1) = one_int
            covars_home(2) = covariates%is_young_adult
            covars_home(3) = covariates%is_adult

            rewards(3) = DOT_PRODUCT(covars_home, model_spec%coeffs_home)

            ! Now we add the type-specific deviation.
            DO i = 2, 3
                rewards(i) = rewards(i) + type_spec_shifts(type_, i)
            END DO

            ! We can now also added the common component of rewards.
            DO i = 1, 3
                rewards(i) = rewards(i) + rewards_common
            END DO

            periods_rewards_systematic(period, k, :) = rewards

        END DO

    END DO

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_backward_induction(periods_emax, states_all, states_number_period, &
        mapping_state_idx, num_periods, max_states_period, periods_draws_emax, num_draws_emax, &
        periods_rewards_systematic, edu_max, delta, coeffs_common, coeffs_work)

    USE lib_norpy

    IMPLICIT NONE

    !/* dummy arguments        */

    DOUBLE PRECISION, INTENT(OUT) :: periods_emax(num_periods, max_states_period)

    DOUBLE PRECISION, INTENT(IN) :: periods_rewards_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN) :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_work(13)
    DOUBLE PRECISION, INTENT(IN) :: delta

    INTEGER, INTENT(IN) :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN) :: states_number_period(:)
    INTEGER, INTENT(IN) :: states_all(:, :, :)
    INTEGER, INTENT(IN) :: max_states_period
    INTEGER, INTENT(IN) :: num_draws_emax
    INTEGER, INTENT(IN) :: num_periods
    INTEGER, INTENT(IN) :: edu_max

    !/* internal arguments*/

    TYPE(MODEL_SPECIFICATION) :: model_spec

    INTEGER(our_int) :: choice_lagged
    INTEGER(our_int) :: period
    INTEGER(our_int) :: type_
    INTEGER(our_int) :: edu
    INTEGER(our_int) :: exp
    INTEGER(our_int) :: k

    REAL(our_dble) :: draws_emax_risk(num_draws_emax, 3)
    REAL(our_dble) :: continuation_value(3)
    REAL(our_dble) :: rewards_systematic(3)
    REAL(our_dble) :: wages_systematic
    REAL(our_dble) :: emax

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------

    model_spec%coeffs_common = coeffs_common
    model_spec%edu_max = edu_max
    model_spec%coeffs_work = coeffs_work
    model_spec%delta = delta
    model_spec%num_draws_emax = num_draws_emax
    model_spec%num_periods = num_periods

    periods_emax = MISSING_FLOAT

    DO period = (model_spec%num_periods - 1), 0, -1

        draws_emax_risk = periods_draws_emax(period + 1, :, :)

        DO k = 0, (states_number_period(period + 1) - 1)

            rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)
            exp = states_all(period + 1, k + 1, 1)
            edu = states_all(period + 1, k + 1, 2)
            choice_lagged = states_all(period + 1, k + 1, 3)
            type_ = states_all(period + 1, k + 1, 4)

            wages_systematic = back_out_systematic_wages(rewards_systematic, exp, edu, &
                    choice_lagged, model_spec)

            IF (period .NE. (num_periods - one_int)) THEN
                continuation_value = get_emaxs(mapping_state_idx, period, periods_emax, &
                        model_spec, exp, edu, type_)
            ELSE
                continuation_value = zero_dble
            END IF

            emax = construct_emax_risk(draws_emax_risk, rewards_systematic, model_spec, edu, &
                    wages_systematic, continuation_value)
            periods_emax(period + 1, k + 1) = emax

        END DO

    END DO

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_simulate(data_sim, states_all, mapping_state_idx, periods_rewards_systematic, &
        periods_emax, num_periods, num_agents_sim, periods_draws_sims, edu_max, coeffs_common, &
        coeffs_work, delta, sample_edu_start, sample_types, sample_lagged_start)

    USE lib_norpy

    IMPLICIT NONE

    !/* dummy arguments        */

    DOUBLE PRECISION, INTENT(OUT) :: data_sim(num_agents_sim * num_periods, 23)

    DOUBLE PRECISION, INTENT(IN) :: periods_rewards_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN) :: periods_draws_sims(:, :, :)
    DOUBLE PRECISION, INTENT(IN) :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN) :: coeffs_work(13)
    DOUBLE PRECISION, INTENT(IN) :: delta

    INTEGER, INTENT(IN) :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN) :: sample_lagged_start(:)
    INTEGER, INTENT(IN) :: sample_edu_start(:)
    INTEGER, INTENT(IN) :: states_all(:, :, :)
    INTEGER, INTENT(IN) :: sample_types(:)
    INTEGER, INTENT(IN) :: num_agents_sim
    INTEGER, INTENT(IN) :: num_periods
    INTEGER, INTENT(IN) :: edu_max

    !/* internal arguments*/

    TYPE(MODEL_SPECIFICATION) :: model_spec
    TYPE(COVARIATES_DICT) :: covariates

    REAL(our_dble) :: rewards_systematic(3)
    REAL(our_dble) :: continuation_value(3)
    REAL(our_dble) :: rewards_ex_post(3)
    REAL(our_dble) :: wages_systematic
    REAL(our_dble) :: total_values(3)
    REAL(our_dble) :: draws(3)

    INTEGER(our_int) :: current_state(4)
    INTEGER(our_int) :: choice_lagged
    INTEGER(our_int) :: choice
    INTEGER(our_int) :: period
    INTEGER(our_int) :: count
    INTEGER(our_int) :: type_
    INTEGER(our_int) :: exp
    INTEGER(our_int) :: edu
    INTEGER(our_int) :: i
    INTEGER(our_int) :: k

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------

    ! Construct derived types
    model_spec%coeffs_common = coeffs_common
    model_spec%edu_max = edu_max
    model_spec%coeffs_work = coeffs_work
    model_spec%delta = delta
    model_spec%num_periods = num_periods
    model_spec%num_agents_sim = num_agents_sim

    data_sim = MISSING_FLOAT

    ! Iterate over agents and periods
    count = 0

    DO i = 0, (num_agents_sim - 1)

        current_state = states_all(1, 1, :)

        current_state(2) = sample_edu_start(i + 1)
        current_state(3) = sample_lagged_start(i + 1)
        current_state(4) = sample_types(i + 1)

        DO period = 0, (model_spec%num_periods - 1)

            ! Distribute state space
            exp = current_state(1)
            edu = current_state(2)
            choice_lagged = current_state(3)
            type_ = current_state(4)

            ! Getting state index
            k = mapping_state_idx(period + 1, exp + 1, edu + 1, choice_lagged, type_ + 1)

            ! TODO: THere is the +1 again
            IF (period .NE. (num_periods - one_int)) THEN
                continuation_value = get_emaxs(mapping_state_idx, period, periods_emax, &
                        model_spec, exp, edu, type_ + 1)
            ELSE
                continuation_value = zero_dble
            END IF

            ! Write agent identifier and current period to data frame
            data_sim(count + 1, 1) = DBLE(i)
            data_sim(count + 1, 2) = DBLE(period)

            ! Calculate ex post rewards
            rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)
            draws = periods_draws_sims(period + 1, i + 1, :)

            wages_systematic = back_out_systematic_wages(rewards_systematic, exp, edu, &
                    choice_lagged, model_spec)

            ! Calculate total utilities
            CALL get_total_values(total_values, rewards_ex_post, rewards_systematic, draws, &
                    model_spec, edu, wages_systematic, continuation_value)

            ! We need to ensure that no individual chooses an inadmissible state. This cannot be done directly in the get_total_values function as the penalty otherwise dominates the interpolation equation. The parameter INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very constructed cases.
            IF (edu >= model_spec%edu_max) total_values(2) = -HUGE_FLOAT

            ! Determine and record optimal choice
            choice = MAXLOC(total_values, DIM = one_int)
            data_sim(count + 1, 3) = DBLE(choice)

            ! Record wages
            IF ((choice .EQ. one_int)) THEN
                data_sim(count + 1, 4) = wages_systematic * draws(1)
            END IF

            ! Write relevant state space for period to data frame
            data_sim(count + 1, 5:8) = current_state(:4)

            ! As we are working with a simulated dataset, we can also output additional information that is not available in an observed dataset. The discount rate is included as this allows to construct the EMAX with the information provided in the simulation output.
            data_sim(count + 1, 9:11) = total_values
            data_sim(count + 1, 12:14) = rewards_systematic
            data_sim(count + 1, 15:17) = draws
            data_sim(count + 1, 18:18) = model_spec%delta

            ! For testing purposes, we also explicitly include the general reward component and the common component.
            covariates = construct_covariates(exp, edu, choice_lagged, type_, period)
            data_sim(count + 1, 19) = calculate_rewards_general(covariates, model_spec%coeffs_work)
            data_sim(count + 1, 20) = calculate_rewards_common(covariates, model_spec)
            data_sim(count + 1, 21:23) = rewards_ex_post

            !# Update work experiences or education
            IF ((choice .EQ. one_int) .OR. (choice .EQ. two_int)) THEN
                current_state(choice) = current_state(choice) + 1
            END IF

            !# Update lagged activity variable.
            current_state(3) = choice

            ! Update row indicator
            count = count + 1

        END DO

    END DO

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
