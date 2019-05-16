!***************************************************************************************************
!***************************************************************************************************
MODULE lib_norpy

    IMPLICIT NONE

    !***********************************************************************************************
    !***********************************************************************************************

    INTEGER, PARAMETER :: our_int = selected_int_kind(9)
    INTEGER, PARAMETER :: our_dble = selected_real_kind(15, 307)

    INTEGER(our_int), PARAMETER :: MISSING_INT = -99_our_int
    INTEGER(our_int), PARAMETER :: three_int = 3_our_int
    INTEGER(our_int), PARAMETER :: zero_int = 0_our_int
    INTEGER(our_int), PARAMETER :: one_int = 1_our_int
    INTEGER(our_int), PARAMETER :: two_int = 2_our_int

    REAL(our_dble), PARAMETER :: INADMISSIBILITY_PENALTY = -400000.00_our_dble
    REAL(our_dble), PARAMETER :: one_hundred_dble = 100.00_our_dble
    REAL(our_dble), PARAMETER :: MISSING_FLOAT = -99.0_our_dble
    REAL(our_dble), PARAMETER :: HUGE_FLOAT = 1.0e20_our_dble
    REAL(our_dble), PARAMETER :: three_dble = 3.00_our_dble
    REAL(our_dble), PARAMETER :: zero_dble = 0.00_our_dble
    REAL(our_dble), PARAMETER :: half_dble = 0.50_our_dble
    REAL(our_dble), PARAMETER :: one_dble = 1.00_our_dble
    REAL(our_dble), PARAMETER :: two_dble = 2.00_our_dble

    TYPE COVARIATES_DICT

        INTEGER(our_int) :: is_return_not_high_school
        INTEGER(our_int) :: is_return_high_school
        INTEGER(our_int) :: not_exp_lagged
        INTEGER(our_int) :: is_young_adult
        INTEGER(our_int) :: choice_lagged
        INTEGER(our_int) :: is_mandatory
        INTEGER(our_int) :: work_lagged
        INTEGER(our_int) :: not_any_exp
        INTEGER(our_int) :: hs_graduate
        INTEGER(our_int) :: co_graduate
        INTEGER(our_int) :: edu_lagged
        INTEGER(our_int) :: any_exp
        INTEGER(our_int) :: is_minor
        INTEGER(our_int) :: is_adult
        INTEGER(our_int) :: period
        INTEGER(our_int) :: type_
        INTEGER(our_int) :: exp
        INTEGER(our_int) :: edu

    END TYPE

    TYPE MODEL_SPECIFICATION

        REAL(our_dble), ALLOCATABLE :: type_shifts(:, :)
        REAL(our_dble) :: coeffs_common(2)
        REAL(our_dble) :: coeffs_work(13)
        REAL(our_dble) :: coeffs_home(3)
        REAL(our_dble) :: shocks_cov(3)
        REAL(our_dble) :: coeffs_edu(7)
        REAL(our_dble) :: delta(1)

        INTEGER(our_int), ALLOCATABLE :: edu_range_start(:)
        INTEGER(our_int) :: num_draws_emax
        INTEGER(our_int) :: num_agents_sim
        INTEGER(our_int) :: num_edu_start
        INTEGER(our_int) :: num_periods
        INTEGER(our_int) :: seed_emax
        INTEGER(our_int) :: num_types
        INTEGER(our_int) :: seed_sim
        INTEGER(our_int) :: edu_max

    END TYPE

CONTAINS

    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION calculate_wages_systematic(covariates, model_spec) RESULT(wages)

        !/* dummy arguments        */

        REAL(our_dble) :: wages

        TYPE(MODEL_SPECIFICATION), INTENT(IN) :: model_spec
        TYPE(COVARIATES_DICT), INTENT(IN) :: covariates

        !/* local variables        */

        REAL(our_dble) :: covars_wages(10)

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        covars_wages(1) = one_dble
        covars_wages(2) = covariates%edu
        covars_wages(3) = covariates%exp
        covars_wages(4) = (covariates%exp ** 2) / one_hundred_dble
        covars_wages(5) = covariates%hs_graduate
        covars_wages(6) = covariates%co_graduate
        covars_wages(7) = covariates%period - one_dble
        covars_wages(8) = covariates%is_minor
        covars_wages(9:) = (/ covariates%any_exp, covariates%work_lagged /)

        wages = EXP(DOT_PRODUCT(covars_wages, model_spec%coeffs_work(:10)))
        wages = wages * EXP(model_spec%type_shifts(covariates%type_, 1))

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION calculate_rewards_common(covariates, model_spec) RESULT(rewards_common)

        !/* dummy arguments       */

        REAL(our_dble) :: rewards_common

        TYPE(MODEL_SPECIFICATION), INTENT(IN) :: model_spec
        TYPE(COVARIATES_DICT), INTENT(IN) :: covariates

        !/* local variables        */

        REAL(our_dble) :: covars_common(2)

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        covars_common = (/ covariates%hs_graduate, covariates%co_graduate /)
        rewards_common = DOT_PRODUCT(model_spec%coeffs_common, covars_common)

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION calculate_rewards_general(covariates, coeffs_general) RESULT(rewards_general)

        !/* dummy arguments        */

        REAL(our_dble) :: rewards_general

        REAL(our_dble), INTENT(IN) :: coeffs_general(3)

        TYPE(COVARIATES_DICT), INTENT(IN) :: covariates

        !/* local variables       */

        REAL(our_dble) :: covars_general(3)

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        covars_general = (/ one_int, covariates%not_exp_lagged, covariates%not_any_exp /)
        rewards_general = DOT_PRODUCT(covars_general, coeffs_general)

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION to_boolean(input) RESULT(output)

        !/* dummy arguments    */

        INTEGER(our_int), INTENT(IN) :: input

        LOGICAL(our_dble) :: output

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        IF (input .EQ. one_int) THEN
            output = .TRUE.
        ELSEIF (input .EQ. zero_int) THEN
            output = .FALSE.
        ELSE
            STOP 'Misspecified request'
        END IF

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION construct_covariates(exp, edu, choice_lagged, type_, period) RESULT(covariates)

        !/* dummy arguments    */

        TYPE(COVARIATES_DICT) :: covariates

        INTEGER(our_int), INTENT(IN) :: choice_lagged
        INTEGER(our_int), INTENT(IN) :: period
        INTEGER(our_int), INTENT(IN) :: type_
        INTEGER(our_int), INTENT(IN) :: exp
        INTEGER(our_int), INTENT(IN) :: edu

        !/* local variables        */

        INTEGER(our_int) :: hs_graduate
        INTEGER(our_int) :: edu_lagged

        LOGICAL :: cond

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        edu_lagged = TRANSFER(choice_lagged .EQ. two_int, our_int)

        cond = ((exp .GT. 0) .AND. choice_lagged .NE. one_int)
        covariates%not_exp_lagged = TRANSFER(cond, our_int)
        covariates%work_lagged = TRANSFER(choice_lagged .EQ. one_int, our_int)
        covariates%edu_lagged = TRANSFER(choice_lagged .EQ. two_int, our_int)
        covariates%not_any_exp = TRANSFER(exp .EQ. 0, our_int)
        covariates%any_exp = TRANSFER(exp .GT. 0, our_int)
        covariates%is_minor = TRANSFER(period .LT. 3, our_int)
        covariates%is_young_adult = TRANSFER(((period .GE. 3) .AND. (period .LT. 6)), our_int)
        covariates%is_adult = TRANSFER(period .GE. 6, our_int)
        covariates%is_mandatory = TRANSFER(edu .LT. 9, our_int)
        covariates%co_graduate = TRANSFER(edu .GE. 15, our_int)
        covariates%hs_graduate = TRANSFER(edu .GE. 12, our_int)

        hs_graduate = covariates%hs_graduate

        covariates%is_return_not_high_school = &
                TRANSFER((.NOT. to_boolean(edu_lagged)) .AND. (.NOT. to_boolean(hs_graduate)), our_int)
        covariates%is_return_high_school = &
                TRANSFER((.NOT. to_boolean(edu_lagged)) .AND. to_boolean(hs_graduate), our_int)
        covariates%choice_lagged = choice_lagged
        covariates%period = period
        covariates%exp = exp
        covariates%type_ = type_
        covariates%edu = edu

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION construct_emax_risk(draws_emax_risk, rewards_systematic, model_spec, edu, &
            wages_systematic, continuation_value) RESULT(emax)

        !/* dummy arguments    */

        REAL(our_dble) :: emax

        TYPE(MODEL_SPECIFICATION), INTENT(IN) :: model_spec

        REAL(our_dble), INTENT(IN) :: draws_emax_risk(:, :)
        REAL(our_dble), INTENT(IN) :: continuation_value(3)
        REAL(our_dble), INTENT(IN) :: rewards_systematic(3)
        REAL(our_dble), INTENT(IN) :: wages_systematic

        INTEGER(our_int), INTENT(IN) :: edu

        !/* internals objects    */

        INTEGER(our_int) :: i

        REAL(our_dble) :: rewards_ex_post(3)
        REAL(our_dble) :: total_values(3)
        REAL(our_dble) :: draws(3)

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        emax = zero_dble

        DO i = 1, model_spec%num_draws_emax

            draws = draws_emax_risk(i, :)

            CALL get_total_values(total_values, rewards_ex_post, rewards_systematic, draws, &
                    model_spec, edu, wages_systematic, continuation_value)

            emax = emax + MAXVAL(total_values)

        END DO

        emax = emax / model_spec%num_draws_emax

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION back_out_systematic_wages(rewards_systematic, exp, edu, choice_lagged, model_spec) &
        RESULT(wages_systematic)

        !/* dummy arguments        */

        REAL(our_dble) :: wages_systematic

        TYPE(MODEL_SPECIFICATION), INTENT(IN) :: model_spec

        INTEGER(our_int), INTENT(IN) :: choice_lagged
        INTEGER(our_int), INTENT(IN) :: exp
        INTEGER(our_int), INTENT(IN) :: edu

        REAL(our_dble), INTENT(IN) :: rewards_systematic(3)

        !/* internal arguments        */

        TYPE(COVARIATES_DICT) :: covariates

        INTEGER(our_int) :: covars_general(3)
        INTEGER(our_int) :: covars_common(2)

        REAL(our_dble) :: rewards_common
        REAL(our_dble) :: general

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        covariates = construct_covariates(exp, edu, choice_lagged, MISSING_INT, MISSING_INT)
        covars_general = (/ one_int, covariates%not_exp_lagged, covariates%not_any_exp /)
        general = DOT_PRODUCT(covars_general, model_spec%coeffs_work(11:13))

        covars_common = (/ covariates%hs_graduate, covariates%co_graduate /)
        rewards_common = DOT_PRODUCT(covars_common, model_spec%coeffs_common)
        wages_systematic = rewards_systematic(1) - general - rewards_common

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    SUBROUTINE get_total_values(total_values, rewards_ex_post, rewards_systematic, draws, &
            model_spec, edu, wages_systematic, continuation_value)

        !/* dummy arguments        */

        REAL(our_dble), INTENT(OUT) :: rewards_ex_post(3)
        REAL(our_dble), INTENT(OUT) :: total_values(3)

        TYPE(MODEL_SPECIFICATION), INTENT(IN) :: model_spec

        INTEGER(our_int), INTENT(IN) :: edu

        REAL(our_dble), INTENT(IN) :: rewards_systematic(3)
        REAL(our_dble), INTENT(IN) :: continuation_value(3)
        REAL(our_dble), INTENT(IN) :: wages_systematic
        REAL(our_dble), INTENT(IN) :: draws(3)

        !/* internal arguments        */

        REAL(our_dble) :: total_increment

        INTEGER(our_int) :: i

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        rewards_ex_post = zero_dble

        total_increment = rewards_systematic(1) - wages_systematic
        rewards_ex_post(1) = wages_systematic * draws(1) + total_increment

        Do i = 2, 3
            rewards_ex_post(i) = rewards_systematic(i) + draws(i)
        END DO

        total_values = rewards_ex_post + model_spec%delta(1) * continuation_value

        IF (edu >= model_spec%edu_max) THEN
            total_values(3) = total_values(3) + INADMISSIBILITY_PENALTY
        END IF

    END SUBROUTINE
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION get_emaxs(mapping_state_idx, period, periods_emax, model_spec, exp, edu, type_) &
        RESULT(emaxs)

        !/* dummy arguments        */

        TYPE(MODEL_SPECIFICATION), INTENT(IN) :: model_spec

        REAL(our_dble) :: emaxs(3)

        INTEGER(our_int), INTENT(IN) :: mapping_state_idx(:, :, :, :, :)
        INTEGER(our_int), INTENT(IN) :: period
        INTEGER(our_int), INTENT(IN) :: type_
        INTEGER(our_int), INTENT(IN) :: exp
        INTEGER(our_int), INTENT(IN) :: edu

        REAL(our_dble), INTENT(IN) :: periods_emax(:, :)

        !/* internal arguments       */

        INTEGER(our_int) :: future_idx

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        future_idx = mapping_state_idx(period + 1 , exp + 1 + 1, edu + 1, 1, type_)
        emaxs(1) = periods_emax(period + 1 , future_idx)

        IF(edu .GE. model_spec%edu_max) THEN
            emaxs(2) = zero_dble
        ELSE
            future_idx = mapping_state_idx(period + 1 , exp + 1, edu + 1 + 1, 2, type_)
            emaxs(2) = periods_emax(period + 1, future_idx)
        END IF

        future_idx = mapping_state_idx(period + 1, exp + 1, edu + 1, 3, type_)
        emaxs(3) = periods_emax(period + 1 , future_idx)

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************

END MODULE
