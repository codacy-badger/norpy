!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_create_state_space(states_all, states_number_period, mapping_state_idx, &
        max_states_period, num_periods, num_types, edu_spec_start, edu_spec_max, min_idx_int,test_indication_optional)

    USE lib_norpy

    !/* dummy arguments        */

    INTEGER, INTENT(OUT) :: mapping_state_idx(num_periods, num_periods, min_idx_int, 4, num_types)
    INTEGER, INTENT(OUT) :: states_all(num_periods, 500000, 4)
    INTEGER, INTENT(OUT) :: states_number_period(num_periods)
    INTEGER, INTENT(OUT) :: max_states_period


    INTEGER, INTENT(IN) :: edu_spec_start(:)
    INTEGER, INTENT(IN) :: edu_spec_max
    INTEGER, INTENT(IN) :: min_idx_int
    INTEGER, INTENT(IN) :: num_periods
    INTEGER, INTENT(IN) :: num_types
    LOGICAL, INTENT(IN),OPTIONAL :: test_indication_optional

    !/* local variables        */
    LOGICAL :: test_indication
    INTEGER :: choice_lagged
    INTEGER :: num_edu_start
    INTEGER :: edu_start
    INTEGER :: edu_add
    INTEGER :: period
    INTEGER :: type_
    INTEGER :: exp_
    INTEGER :: k
    INTEGER :: j

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------
    
    !Initialize test indication variable     
    IF (PRESENT(test_indication_optional) .EQV. .TRUE.) THEN
    	test_indication = test_indication_optional
    ELSE 
        test_indication = .FALSE.
    END IF
		

    ! Construct auxiliary objects
    num_edu_start = SIZE(edu_spec_start)

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

                edu_start = edu_spec_start(j)

                ! Loop over all admissible work experiences for Occupation A
                DO exp_ = 0, num_periods

                    

                        ! Loop over all admissible additional education levels
                        DO edu_add = 0, num_periods


                            IF (edu_add + exp_  .GT. period - 1) CYCLE

                            ! Agent cannot attain more additional education than (EDU_MAX - EDU_START).
                            IF (edu_add .GT. (edu_spec_max - edu_start)) CYCLE

                            DO choice_lagged = 1,3

                                IF (period .GT. one_int) THEN

                                    ! (0, 1) Whenever an agent has only worked in Occupation A, then choice_lagged cannot be anything other than one.
                                    IF ((choice_lagged .NE. one_int) .AND. (exp_ .EQ. period-1)) CYCLE
                                    

                                   

                                    ! (0, 3) Whenever an agent has only acquired additional education, then choice_lagged cannot be  anything other than two.
                                    IF ((choice_lagged .NE. two_int) .AND. (edu_add .EQ. period - 1)) CYCLE

                                    ! (0, 4) Whenever an agent has not acquired any additional education and we are not in the first period, then lagged education cannot take a value of three.
                                    IF ((choice_lagged .EQ. two_int) .AND. (edu_add .EQ. zero_int)) CYCLE
                                    !Whenever an agent has only worked or has only stayed in school he cannot have home as lagged choice
                                    IF ((choice_lagged .EQ. three_int) .AND. (edu_add + exp_  .EQ. period - 1)) CYCLE

                                END IF

                                ! (1, 1) In the first period individual either were in school the previous period as well or at home. The cannot have any work experience.
                                IF (period .EQ. one_int) THEN

                                    IF (choice_lagged .EQ. one_int)  CYCLE

                                END IF
                                ! (2, 1) An individual that has never worked in Occupation A cannot have a that lagged activity.
                                IF ((choice_lagged .EQ. one_int) .AND. (exp_ .EQ. zero_int)) CYCLE

                                ! (3, 1) An individual that has never worked in Occupation B cannot have a that lagged activity.
                                     
                                

				
                                IF (test_indication .EQV. .FALSE.) THEN

                                	IF (mapping_state_idx(period, exp_ + 1 ,edu_start + edu_add + 1 , choice_lagged, type_)&
					&.NE. MISSING_INT) CYCLE
                                END IF

                                ! ! Collect mapping of state space to array index.
                                mapping_state_idx(period , exp_ + 1 , edu_start + edu_add + 1 , choice_lagged, type_ ) = k

                                ! Collect all possible realizations of state space
                                states_all(period, k + 1 , :) = (/ exp_, edu_start + edu_add, choice_lagged, type_ /)

                                ! Update count
                                k = k + 1

                            END DO

                        END DO

                    END DO

                END DO

            END DO

        

        ! Record maximum number of state space realizations by time period
        states_number_period(period) = k

    END DO

    ! Auxiliary object
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

    INTEGER :: choice_lagged

    INTEGER :: covars_home(3)
    INTEGER :: covars_edu(7)
    INTEGER :: period
    INTEGER :: type_
    INTEGER :: exp_

    INTEGER :: edu
    INTEGER :: k
    INTEGER :: i

    DOUBLE PRECISION :: rewards_general
    DOUBLE PRECISION :: rewards_common
    DOUBLE PRECISION :: rewards(3)
    DOUBLE PRECISION :: wages

    TYPE(COVARIATES_DICT) :: covariates
    
    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------

    periods_rewards_systematic = MISSING_FLOAT
		

    
  
    
    ! Calculate systematic instantaneous rewards
    DO period = num_periods, 1, -1
	
        DO k=1, (states_number_period(period))
	    	

            ! Distribute state space
            exp_ = states_all(period, k, 1)
            edu = states_all(period, k, 2)
            choice_lagged = states_all(period, k, 3)
            type_ = states_all(period, k, 4)
            

            ! Construct auxiliary information
            covariates = construct_covariates(exp_, edu, choice_lagged, type_, period)

            ! Calculate common and general rewards component.
            rewards_general = calculate_rewards_general(covariates, coeffs_work(11:13))
            rewards_common = calculate_rewards_common(covariates, coeffs_common)

            ! Calculate the systematic part of OCCUPATION A and OCCUPATION B rewards. these are defined in a general sense, where not only wages matter.
            ! Only occupation a now will give a REAL instead of an ARRAY
            wages = calculate_wages_systematic(covariates, coeffs_work, type_spec_shifts)

            
            rewards(1) = wages + rewards_general
	    	
            

            ! Calculate systematic part of schooling utility
            covars_edu(1) = one_int
            covars_edu(2) = covariates%hs_graduate
            covars_edu(3) = covariates%co_graduate
            covars_edu(4) = covariates%is_return_not_high_school
            covars_edu(5) = covariates%is_return_high_school
            covars_edu(6) = covariates%period - one_int
            covars_edu(7) = covariates%is_mandatory

            rewards(2) = DOT_PRODUCT(covars_edu, coeffs_edu)

            ! Calculate systematic part of HOME
            covars_home(1) = one_int
            
            covars_home(2) = covariates%is_young_adult
            covars_home(3) = covariates%is_adult
            
            rewards(3) = DOT_PRODUCT(covars_home, coeffs_home)

            ! Now we add the type-specific deviation.
            DO i = 2, 3
                rewards(i) = rewards(i) + type_spec_shifts(type_ , i - 1)
                !WRITE(*,*) type_spec_shifts(type_,1)
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
