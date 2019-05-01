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
        INTEGER(our_int) :: work_lagged
        INTEGER(our_int) :: not_any_exp
        INTEGER(our_int) :: hs_graduate
        INTEGER(our_int) :: co_graduate
        INTEGER(our_int) :: edu_lagged
        INTEGER(our_int) :: any_exp
        INTEGER(our_int) :: is_minor
        INTEGER(our_int) :: is_adult
        INTEGER(our_int) :: period
        INTEGER(our_int) :: exp
        INTEGER(our_int) :: type_
        INTEGER(our_int) :: edu
        INTEGER(our_int) :: is_mandatory

    END TYPE

    TYPE OPTIMPARAS_DICT

        REAL(our_dble), ALLOCATABLE :: type_shifts(:, :)
        REAL(our_dble), ALLOCATABLE :: typeshares(:)
        REAL(our_dble) :: coeffs_common(2)
        REAL(our_dble) :: coeffs_home(3)
        REAL(our_dble) :: coeffs_edu(7)
        REAL(our_dble) :: coeffs_work(13)
        REAL(our_dble) :: delta(1)

    END TYPE

    TYPE EDU_DICT

        INTEGER(our_int), ALLOCATABLE :: start(:)
        INTEGER(our_int) :: max

        REAL(our_dble), ALLOCATABLE :: lagged(:)
        REAL(our_dble), ALLOCATABLE :: share(:)

    END TYPE

CONTAINS

    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION calculate_wages_systematic(covariates,coeffs_work, type_shifts) RESULT(wages)

        !/* dummy arguments        */
	REAL(our_dble) :: wages 
        REAL(our_dble), INTENT(IN) :: type_shifts(:, :)
        REAL(our_dble), INTENT(IN) :: coeffs_work(13)
        
        TYPE(COVARIATES_DICT), INTENT(IN) :: covariates

        !/* local variables        */

        INTEGER(our_int) :: i

        REAL(our_dble) :: covars_wages(10)
        
        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        ! Auxiliary objects
        covars_wages(1) = one_dble
        covars_wages(2) = covariates%edu
        covars_wages(3) = covariates%exp
        covars_wages(4) = (covariates%exp ** 2) / one_hundred_dble
	covars_wages(5) = covariates%hs_graduate
        covars_wages(6) = covariates%co_graduate
        covars_wages(7) = covariates%period - one_dble
        covars_wages(8) = covariates%is_minor
        covars_wages(9:) = (/ covariates%any_exp, covariates%work_lagged/)
        
        wages = EXP(DOT_PRODUCT(covars_wages, coeffs_work(:10)))
        wages = wages * EXP(type_shifts(covariates%type_ + 1,1))
        
    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION calculate_rewards_common(covariates, coeffs_common) RESULT(rewards_common)

        !/* dummy arguments       */
	
	REAL(our_dble) :: rewards_common

        TYPE(COVARIATES_DICT), INTENT(IN) :: covariates

        REAL(our_dble), INTENT(IN) :: coeffs_common(2)

        !/* local variables        */

        REAL(our_dble) :: covars_common(2)

        !------------------------------------------------------------------------------
        ! Algorithm
        !------------------------------------------------------------------------------

        covars_common = (/ covariates%hs_graduate, covariates%co_graduate /)
        rewards_common = DOT_PRODUCT(coeffs_common, covars_common)

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

        ! Auxiliary objects
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
    FUNCTION construct_emax_risk(period, k, draws_emax_risk, rewards_systematic, &
            periods_emax, states_all, mapping_state_idx, edu_spec, optim_paras, &
            num_draws_emax, num_periods) RESULT(emax)

        !/* external objects    */
        
        REAL(our_dble) :: emax

        TYPE(OPTIMPARAS_DICT), INTENT(IN) :: optim_paras
        TYPE(EDU_DICT), INTENT(IN) :: edu_spec

        INTEGER(our_int), INTENT(IN) :: mapping_state_idx(:, :, :, :, :)
        INTEGER(our_int), INTENT(IN) :: states_all(:, :, :)
        INTEGER(our_int), INTENT(IN) :: num_draws_emax
        INTEGER(our_int), INTENT(IN) :: num_periods
        INTEGER(our_int), INTENT(IN) :: period
        INTEGER(our_int), INTENT(IN) :: k

        REAL(our_dble), INTENT(IN) :: periods_emax(:, :)
        REAL(our_dble), INTENT(IN) :: draws_emax_risk(:, :)
        REAL(our_dble), INTENT(IN) :: rewards_systematic(3)

        !/* internals objects    */

        INTEGER(our_int) :: i

        REAL(our_dble) :: rewards_ex_post(3)
        REAL(our_dble) :: total_values(3)
        REAL(our_dble) :: draws(3)
        REAL(our_dble) :: maximum

        !-------------------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------------------

        ! Iterate over Monte Carlo draws
        emax = zero_dble
        DO i = 1, num_draws_emax

            ! Select draws for this draw
            draws = draws_emax_risk(i, :)

            ! Calculate total value
            
            CALL get_total_values(total_values, rewards_ex_post, period, num_periods, &
                    rewards_systematic, draws, mapping_state_idx, periods_emax, k, states_all, &
                    optim_paras, edu_spec)

            ! Determine optimal choice
            maximum = MAXVAL(total_values)

            ! Recording expected future value
            emax = emax + maximum

        END DO

        ! Scaling
        emax = emax / num_draws_emax

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION back_out_systematic_wages(rewards_systematic, exp, edu, choice_lagged, &
            optim_paras) RESULT(wages_systematic)

        !/* external objects        */

        REAL(our_dble) :: wages_systematic

        TYPE(OPTIMPARAS_DICT), INTENT(IN) :: optim_paras

        INTEGER(our_int), INTENT(IN) :: choice_lagged
        INTEGER(our_int), INTENT(IN) :: exp

        INTEGER(our_int), INTENT(IN) :: edu

        REAL(our_dble), INTENT(IN) :: rewards_systematic(3)

        !/* internal objects        */

        TYPE(COVARIATES_DICT) :: covariates

        INTEGER(our_int) :: covars_general(3)
        INTEGER(our_int) :: covars_common(2)
        INTEGER(our_int) :: i
        REAL(our_dble) :: rewards_common
        REAL(our_dble) :: general
        !-------------------------------------------------------------------------------
        ! Algorithm
        !-------------------------------------------------------------------------------
        
        covariates = construct_covariates(exp, edu, choice_lagged, MISSING_INT, MISSING_INT)
	covars_general = (/ one_int, covariates%not_exp_lagged, covariates%not_any_exp /)
        general = DOT_PRODUCT(covars_general, optim_paras%coeffs_work(11:13))
        
        ! Second we do the same with the common component.
        covars_common = (/ covariates%hs_graduate, covariates%co_graduate /)
        rewards_common = DOT_PRODUCT(covars_common,optim_paras%coeffs_common)
	wages_systematic = rewards_systematic(1) - general - rewards_common

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
    SUBROUTINE get_total_values(total_values, rewards_ex_post, period, num_periods, &
            rewards_systematic, draws, mapping_state_idx, periods_emax, k, states_all, &
            optim_paras, edu_spec)		

        !/* external objects        */

        REAL(our_dble), INTENT(OUT) :: rewards_ex_post(3)
        REAL(our_dble), INTENT(OUT) :: total_values(3)

        TYPE(OPTIMPARAS_DICT), INTENT(IN) :: optim_paras
        TYPE(EDU_DICT), INTENT(IN) :: edu_spec

        INTEGER(our_int), INTENT(IN) :: num_periods
        INTEGER(our_int), INTENT(IN) :: mapping_state_idx(:, :, :, :, :)
        INTEGER(our_int), INTENT(IN) :: states_all(:, :, :)
        INTEGER(our_int), INTENT(IN) :: period
        INTEGER(our_int), INTENT(IN) :: k

        REAL(our_dble), INTENT(IN) :: periods_emax(:, :)
        REAL(our_dble), INTENT(IN) :: rewards_systematic(3)
        REAL(our_dble), INTENT(IN) :: draws(3)

        !/* internal objects        */

        REAL(our_dble) :: wages_systematic
	REAL(our_dble) :: total_increment
        REAL(our_dble) :: emaxs(3)

        INTEGER(our_int) :: choice_lagged
        INTEGER(our_int) :: exp
	INTEGER(our_int) :: edu
        INTEGER(our_int) :: i
        
        !------------------------------------------------------------------------------
        ! Algorithm
        !------------------------------------------------------------------------------
        
        ! We need to back out the wages from the total systematic rewards to working in the labor market to add the shock properly.
        exp = states_all(period + 1, k + 1, 1)
	edu = states_all(period + 1, k + 1, 2)
        choice_lagged = states_all(period + 1, k + 1, 3)
        wages_systematic = back_out_systematic_wages(rewards_systematic, exp, edu, choice_lagged, optim_paras)
        
        ! Initialize containers
        rewards_ex_post = zero_dble

        ! Calculate ex post rewards
         total_increment = rewards_systematic(1) - wages_systematic
         rewards_ex_post(1) = wages_systematic * draws(1) + total_increment
         
        Do i = 2, 3
            rewards_ex_post(i) = rewards_systematic(i) + draws(i)
        END DO
        
	! Get future values
        
        IF (period .NE. (num_periods - one_int)) THEN
            emaxs = get_emaxs( mapping_state_idx, period, periods_emax, k, states_all, edu_spec)
        ELSE
            emaxs = zero_dble
        END IF
        
        ! Calculate total utilities
        total_values = rewards_ex_post + optim_paras%delta(1) * emaxs

        ! This is required to ensure that the agent does not choose any inadmissible states. If the state is inadmissible emaxs takes value zero.
        IF (states_all(period + 1, k + 1, 3) >= edu_spec%max) THEN
            total_values(3) = total_values(3) + INADMISSIBILITY_PENALTY
        END IF

    END SUBROUTINE
    !***********************************************************************************************
    !***********************************************************************************************
    FUNCTION get_emaxs(mapping_state_idx, period, periods_emax, k, states_all, edu_spec) RESULT(emaxs)

	!/* external objects        */

	TYPE(EDU_DICT), INTENT(IN) :: edu_spec

	REAL(our_dble) :: emaxs(3)

	INTEGER(our_int), INTENT(IN) :: mapping_state_idx(:, :, :, :, :)
	INTEGER(our_int), INTENT(IN) :: states_all(:, :, :)
	INTEGER(our_int), INTENT(IN) :: period
	INTEGER(our_int), INTENT(IN) :: k

	REAL(our_dble), INTENT(IN) :: periods_emax(:, :)

	!/* internals objects       */

	INTEGER(our_int) :: future_idx
	INTEGER(our_int) :: exp
	INTEGER(our_int) :: type_
	INTEGER(our_int) :: edu

	!------------------------------------------------------------------------------
	! Algorithm
	!------------------------------------------------------------------------------

	! Distribute state space
	exp = states_all(period + 1, k + 1, 1)
	edu = states_all(period + 1, k + 1, 2)
	type_ = states_all(period + 1, k + 1, 4)

	! Working in Occupation A
	future_idx = mapping_state_idx(period + 1 + 1, exp + 1 + 1, edu + 1, 1, type_)
	emaxs(1) = periods_emax(period + 1 + 1, future_idx + 1)

	! Increasing schooling. Note that adding an additional year of schooling is only possible for those that have strictly less than the maximum level of additional education allowed.
	IF(edu .GE. edu_spec%max) THEN
	    emaxs(2) = zero_dble
	ELSE
	    future_idx = mapping_state_idx(period + 1 + 1, exp + 1, edu + 1 + 1, 2, type_)
	    emaxs(2) = periods_emax(period + 1 + 1, future_idx + 1)
	END IF

	! Staying at home
	future_idx = mapping_state_idx(period + 1 + 1, exp + 1, edu + 1, 3, type_)
	emaxs(3) = periods_emax(period + 1 + 1, future_idx + 1)

    END FUNCTION
!***************************************************************************************************
!***************************************************************************************************

END MODULE
