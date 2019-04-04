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

    REAL(our_dble), PARAMETER :: one_hundred_dble = 100.00_our_dble
    REAL(our_dble), PARAMETER :: MISSING_FLOAT = -99.0_our_dble
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
        INTEGER(our_int) :: exp_

        INTEGER(our_int) :: type
        INTEGER(our_int) :: edu

        INTEGER(our_int) :: is_mandatory

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
        covars_wages(3) = covariates%exp_
        covars_wages(4) = (covariates%exp_ ** 2) / one_hundred_dble
	covars_wages(5) = covariates%hs_graduate
        covars_wages(6) = covariates%co_graduate
        covars_wages(7) = covariates%period - one_dble
        covars_wages(8) = covariates%is_minor

        covars_wages(9:) = (/ covariates%any_exp, covariates%work_lagged/)
        wages = EXP(DOT_PRODUCT(covars_wages, coeffs_work(:10)))

	wages = wages * EXP(type_shifts(covariates%type + 1,1))
        

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
    FUNCTION construct_covariates(exp_, edu, choice_lagged, type_, period) RESULT(covariates)

        !/* dummy arguments    */

        TYPE(COVARIATES_DICT) :: covariates

        INTEGER(our_int), INTENT(IN) :: choice_lagged
        INTEGER(our_int), INTENT(IN) :: period
        INTEGER(our_int), INTENT(IN) :: type_
        INTEGER(our_int), INTENT(IN) :: exp_
        
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


        cond = ((exp_ .GT. 0) .AND. choice_lagged .NE. one_int)
        covariates%not_exp_lagged = TRANSFER(cond, our_int)
	covariates%work_lagged = TRANSFER(choice_lagged .EQ. one_int, our_int)
        covariates%edu_lagged = TRANSFER(choice_lagged .EQ. two_int, our_int)
        covariates%not_any_exp = TRANSFER(exp_ .EQ. 0, our_int)
        covariates%any_exp = TRANSFER(exp_ .GT. 0, our_int)
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
        covariates%exp_ = exp_
	covariates%type = type_
        covariates%edu = edu

    END FUNCTION
    !***********************************************************************************************
    !***********************************************************************************************
END MODULE
