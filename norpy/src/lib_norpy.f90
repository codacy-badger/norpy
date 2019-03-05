!******************************************************************************
!******************************************************************************
MODULE lib_norpy


    !/*	setup	                */

    IMPLICIT NONE

!******************************************************************************
!******************************************************************************

    INTEGER, PARAMETER          :: our_int      = selected_int_kind(9)
    INTEGER, PARAMETER :: our_dble = selected_real_kind(15, 307)


    INTEGER(our_int), PARAMETER :: zero_int     = 0_our_int
    INTEGER(our_int), PARAMETER :: one_int      = 1_our_int
    INTEGER(our_int), PARAMETER :: two_int      = 2_our_int
    INTEGER(our_int), PARAMETER :: three_int    = 3_our_int

    INTEGER(our_int), PARAMETER :: MISSING_INT = -99_our_int

    REAL(our_dble), PARAMETER :: MISSING_FLOAT = -99.0_our_dble

    REAL(our_dble), PARAMETER   :: zero_dble    = 0.00_our_dble
    REAL(our_dble), PARAMETER   :: half_dble    = 0.50_our_dble, one_hundred_dble = 100.00_our_dble
    REAL(our_dble), PARAMETER   :: one_dble     = 1.00_our_dble
    REAL(our_dble), PARAMETER   :: two_dble     = 2.00_our_dble
REAL(our_dble), PARAMETER :: three_dble = 3.00_our_dble

    TYPE COVARIATES_DICT

        INTEGER(our_int)                :: is_return_not_high_school
        INTEGER(our_int)                :: is_return_high_school
        INTEGER(our_int)                :: not_exp_a_lagged
        INTEGER(our_int)                :: not_exp_b_lagged
        INTEGER(our_int)                :: is_young_adult
        INTEGER(our_int)                :: choice_lagged
        INTEGER(our_int)                :: work_a_lagged
        INTEGER(our_int)                :: work_b_lagged
        INTEGER(our_int)                :: not_any_exp_a
        INTEGER(our_int)                :: not_any_exp_b
        INTEGER(our_int)                :: hs_graduate
        INTEGER(our_int)                :: co_graduate
        INTEGER(our_int)                :: edu_lagged
        INTEGER(our_int)                :: any_exp_a
        INTEGER(our_int)                :: any_exp_b
        INTEGER(our_int)                :: is_minor
        INTEGER(our_int)                :: is_adult
        INTEGER(our_int)                :: period
        INTEGER(our_int)                :: exp_a
        INTEGER(our_int)                :: exp_b
        INTEGER(our_int)                :: type
        INTEGER(our_int)                :: edu

        INTEGER(our_int)                :: is_mandatory

    END TYPE


        INTERFACE to_boolean

        MODULE PROCEDURE float_to_boolean, integer_to_boolean

    END INTERFACE

    CONTAINS
!******************************************************************************
!******************************************************************************
FUNCTION calculate_wages_systematic(covariates, coeffs_a, coeffs_b, type_shifts) RESULT(wages)

    !/* external objects        */

    REAL(our_dble)                      :: wages(2)

    TYPE(COVARIATES_DICT), INTENT(IN)   :: covariates

    REAL(our_dble), INTENT(IN)          :: coeffs_a(15), coeffs_b(15), type_shifts(:, :)

    !/* internal objects        */
    INTEGER(our_int)                    :: i

    REAL(our_dble)                      :: covars_wages(12)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    covars_wages(1) = one_dble
    covars_wages(2) = covariates%edu
    covars_wages(3) = covariates%exp_a
    covars_wages(4) = (covariates%exp_a ** 2) / one_hundred_dble
    covars_wages(5) = covariates%exp_b
    covars_wages(6) = (covariates%exp_b ** 2) / one_hundred_dble
    covars_wages(7) = covariates%hs_graduate
    covars_wages(8) = covariates%co_graduate
    covars_wages(9) = covariates%period - one_dble
    covars_wages(10) = covariates%is_minor

    ! Calculate systematic part of reward in OCCUPAION A and OCCUPATION B
    covars_wages(11:) = (/ covariates%any_exp_a, covariates%work_a_lagged/)
    wages(1) = EXP(DOT_PRODUCT(covars_wages, coeffs_a(:12)))

    ! Calculate systematic part of reward in Occupation B
    covars_wages(11:) = (/ covariates%any_exp_b, covariates%work_b_lagged/)
    wages(2) = EXP(DOT_PRODUCT(covars_wages, coeffs_b(:12)))

    DO i = 1, 2
        wages(i) = wages(i) * EXP(type_shifts(covariates%type + 1, i))
    END DO

END FUNCTION

!******************************************************************************
!******************************************************************************
FUNCTION calculate_rewards_common(covariates, coeffs_common) RESULT(rewards_common)

    !/* external objects        */

    REAL(our_dble)                      :: rewards_common

    TYPE(COVARIATES_DICT), INTENT(IN)   :: covariates
    REAL(our_dble), INTENT(IN)   :: coeffs_common(2)

    !/* internal objects        */

    REAL(our_dble)                      :: covars_common(2)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    covars_common = (/ covariates%hs_graduate, covariates%co_graduate /)
    rewards_common = DOT_PRODUCT(coeffs_common, covars_common)

END FUNCTION

!******************************************************************************
!******************************************************************************
FUNCTION calculate_rewards_general(covariates, coeffs_a_general, coeffs_b_general) RESULT(rewards_general)

    !/* external objects        */

    REAL(our_dble)                      :: rewards_general(2)

    REAL(our_dble)  , INTENT(IN)   :: coeffs_a_general(3)
    REAL(our_dble)  , INTENT(IN)   :: coeffs_b_general(3)

    TYPE(COVARIATES_DICT), INTENT(IN)   :: covariates

    !/* internal objects        */

    REAL(our_dble)                      :: covars_general(3)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    covars_general = (/ one_int, covariates%not_exp_a_lagged, covariates%not_any_exp_a /)
    rewards_general(1) = DOT_PRODUCT(covars_general, coeffs_a_general)

    covars_general = (/ one_int, covariates%not_exp_b_lagged, covariates%not_any_exp_b /)
    rewards_general(2) = DOT_PRODUCT(covars_general, coeffs_b_general)

END FUNCTION
        !******************************************************************************
!******************************************************************************
FUNCTION integer_to_boolean(input) RESULT(output)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)                :: input

    LOGICAL(our_dble)                           :: output

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (input .EQ. one_int) THEN
        output = .TRUE.
    ELSEIF (input .EQ. zero_int) THEN
        output = .FALSE.
    ELSE
        STOP 'Misspecified request'
    END IF

END FUNCTION

        !******************************************************************************
!******************************************************************************
FUNCTION float_to_boolean(input) RESULT(output)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)                  :: input

    LOGICAL(our_dble)                           :: output

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (input .EQ. one_dble) THEN
        output = .TRUE.
    ELSEIF (input .EQ. zero_dble) THEN
        output = .FALSE.
    ELSE
        STOP 'Misspecified request'
    END IF

END FUNCTION

        SUBROUTINE test()

            PRINT *, 'hello'

        END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION construct_covariates(exp_a, exp_b, edu, choice_lagged, type_, period) RESULT(covariates)

    !/* external objects        */

    TYPE(COVARIATES_DICT)           :: covariates

    INTEGER(our_int), INTENT(IN)    :: choice_lagged
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: type_
    INTEGER(our_int), INTENT(IN)    :: exp_a
    INTEGER(our_int), INTENT(IN)    :: exp_b
    INTEGER(our_int), INTENT(IN)    :: edu

    !/* internal objects        */

    INTEGER(our_int)                :: hs_graduate
    INTEGER(our_int)                :: edu_lagged

    LOGICAL                         :: cond

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    edu_lagged = TRANSFER(choice_lagged .EQ. three_int, our_int)

    ! These are covariates that are supposed to capture the entry costs.
    cond = ((exp_a .GT. 0) .AND. choice_lagged .NE. one_int)
    covariates%not_exp_a_lagged = TRANSFER(cond, our_int)

    cond = ((exp_b .GT. 0) .AND. choice_lagged .NE. two_int)
    covariates%not_exp_b_lagged = TRANSFER(cond, our_int)

    covariates%work_a_lagged = TRANSFER(choice_lagged .EQ. one_int, our_int)
    covariates%work_b_lagged = TRANSFER(choice_lagged .EQ. two_int, our_int)
    covariates%edu_lagged = TRANSFER(choice_lagged .EQ. three_int, our_int)
    covariates%not_any_exp_a = TRANSFER(exp_a .EQ. 0, our_int)
    covariates%not_any_exp_b = TRANSFER(exp_b .EQ. 0, our_int)
    covariates%any_exp_a = TRANSFER(exp_a .GT. 0, our_int)
    covariates%any_exp_b = TRANSFER(exp_b .GT. 0, our_int)

    covariates%is_minor = TRANSFER(period .LT. 3, our_int)
    covariates%is_young_adult = TRANSFER(((period .GE. 3) .AND. (period .LT. 6)), our_int)
    covariates%is_adult = TRANSFER(period .GE. 6, our_int)

    covariates%is_mandatory = TRANSFER(edu .LT. 9, our_int)
    covariates%co_graduate = TRANSFER(edu .GE. 15, our_int)
    covariates%hs_graduate = TRANSFER(edu .GE. 12, our_int)

    hs_graduate = covariates%hs_graduate

    covariates%is_return_not_high_school = TRANSFER((.NOT. to_boolean(edu_lagged)) .AND. (.NOT. to_boolean(hs_graduate)), our_int)
    covariates%is_return_high_school = TRANSFER((.NOT. to_boolean(edu_lagged)) .AND. to_boolean(hs_graduate), our_int)

    covariates%choice_lagged = choice_lagged
    covariates%period = period
    covariates%exp_a = exp_a
    covariates%exp_b = exp_b
    covariates%type = type_
    covariates%edu = edu

END FUNCTION
!******************************************************************************
END MODULE