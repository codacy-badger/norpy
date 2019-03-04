SUBROUTINE f2py_create_state_space(states_all, states_number_period, mapping_state_idx, max_states_period, num_periods, num_types, edu_spec_start, edu_spec_max, min_idx_int)

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx_int, 4, num_types)
    INTEGER, INTENT(OUT)            :: states_all(num_periods, 500000, 5)
    INTEGER, INTENT(OUT)            :: states_number_period(num_periods)
    INTEGER, INTENT(OUT)            :: max_states_period

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_types
    INTEGER, INTENT(IN)             :: edu_spec_start(:)
    INTEGER, INTENT(IN)             :: min_idx_int
    INTEGER, INTENT(IN)             :: edu_spec_max

    INTEGER                    :: states_all_tmp(num_periods, 1000000, 5)
    INTEGER                    :: choice_lagged
    INTEGER                     :: num_edu_start
    INTEGER                    :: edu_start
    INTEGER                 :: edu_add
    INTEGER                     :: period
    INTEGER                   :: type_
    INTEGER                    :: exp_a
    INTEGER                     :: exp_b
    INTEGER                     :: k
    INTEGER                   :: j

        states_all = MISSING_INT

        ! Construct derived types
        !edu_spec%start = edu_start
        !edu_spec%max = edu_max
        ! Auxiliary variables
        num_edu_start = SIZE(edu_spec_start)

        ! Allocate containers that contain information about the model structure
        !ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4    , num_types))
        !ALLOCATE(states_number_period(num_periods))

        ! Initialize output
        states_number_period = MISSING_INT
        mapping_state_idx    = MISSING_INT
        states_all_tmp       = MISSING_INT

        ! ! Construct state space by periods
        DO period = 0, (num_periods - 1)

            ! Count admissible realizations of state space by period
            k = 0

            ! Loop over all types.
            DO type_ = 0, num_types - 1

                ! Loop over all initial level of schooling
                DO j = 1, num_edu_start

                    edu_start = edu_spec_start(j)

                    ! Loop over all admissible work experiences for Occupation A
                    DO exp_a = 0, num_periods
                        ! Loop over all admissible work experience for Occupation B
                        DO exp_b = 0, num_periods

                            ! Loop over all admissible additional education levels
                            DO edu_add = 0, num_periods
                                 ! Note that the total number of activities does not have is less or equal to the total possible number of activities as the rest is implicitly filled with leisure.
                                 IF (edu_add + exp_a + exp_b .GT. period) CYCLE

                                 ! Agent cannot attain more additional education than (EDU_MAX - EDU_START).
                                 IF (edu_add .GT. (edu_spec_max - edu_start)) CYCLE

                                ! Loop over all admissible values for the lagged activity: (0) Home, (1) Education, (2) Occupation A, and (3) Occupation B.
                                DO choice_lagged = 1, 4

                                    IF (period .GT. zero_int) THEN

                                        ! (0, 1) Whenever an agent has only worked in Occupation A, then choice_lagged cannot be anything other than one.
                                        IF ((choice_lagged .NE. one_int) .AND. (exp_a .EQ. period)) CYCLE

                                        ! (0, 2) Whenever an agent has only worked in Occupation B, then choice_lagged cannot be anything other than two.
                                        IF ((choice_lagged .NE. two_int) .AND. (exp_b .EQ. period)) CYCLE

                                        ! (0, 3) Whenever an agent has only acquired additional education, then choice_lagged cannot be  anything other than three.
                                        IF ((choice_lagged .NE. three_int) .AND. (edu_add .EQ. period)) CYCLE

                                        ! (0, 4) Whenever an agent has not acquired any additional education and we are not in the first period, then lagged education cannot take a value of three.
                                        IF ((choice_lagged .EQ. three_int) .AND. (edu_add .EQ. zero_int)) CYCLE

                                    END IF

                                    ! (1, 1) In the first period individual either were in school the previous period as well or at home. The cannot have any work experience.
                                    IF (period .EQ. zero_int) THEN

                                        IF ((choice_lagged .EQ. one_int) .OR. (choice_lagged .EQ. two_int)) CYCLE

                                    END IF
                                    ! (2, 1) An individual that has never worked in Occupation A cannot have a that lagged activity.
                                    IF ((choice_lagged .EQ. one_int) .AND. (exp_a .EQ. zero_int)) CYCLE

                                    ! (3, 1) An individual that has never worked in Occupation B cannot have a that lagged activity.
                                    IF ((choice_lagged .EQ. two_int) .AND. (exp_b .EQ. zero_int)) CYCLE

                                    ! ! If we have multiple initial conditions it might well be the case that we have a duplicate state, i.e. the same state is possible with other initial condition that period.
                                    IF (mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu_start + edu_add + 1 , choice_lagged, type_ + 1) .NE. MISSING_INT) CYCLE

                                    ! ! Collect mapping of state space to array index.
                                    mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu_start + edu_add + 1 , choice_lagged, type_ + 1) = k

                                    ! Collect all possible realizations of state space
                                    states_all_tmp(period + 1, k + 1, :) = (/ exp_a, exp_b, edu_start + edu_add, choice_lagged, type_ /)

                                    ! Update count
                                    k = k + 1

                                 END DO

                          END DO

                         END DO

                     END DO

                 END DO

            END DO

            ! Record maximum number of state space realizations by time period
            states_number_period(period + 1) = k

        END DO

        ! Auxiliary object
        max_states_period = MAXVAL(states_number_period)

        ! Initialize a host of containers, whose dimensions are not clear.
        !ALLOCATE(states_all(num_periods, max_states_period, 5))
        ! states_all = states_all_tmp(:, :max_states_period, :)

        !states_all_int(:, :max_states_period, :) = states_all
        !states_number_period_int = states_number_period

        ! Updated global variables
        !mapping_state_idx_int = mapping_state_idx
        !max_states_period_int = max_states_period

END SUBROUTINE


SUBROUTINE f2py_calculate_immediate_rewards()

PRINT *, "calc comingup"

END SUBROUTINE
