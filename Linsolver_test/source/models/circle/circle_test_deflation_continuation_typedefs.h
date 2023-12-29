#ifndef __CIRCLE_TEST_DEFLATION_CONTINUATION_TYPEDEFS_H__
#define __CIRCLE_TEST_DEFLATION_CONTINUATION_TYPEDEFS_H__
#define Blocks_x_ 64


    typedef SCALAR_TYPE real;


    typedef utils::log_std log_t;
    typedef gpu_vector_operations<real> vec_ops_real;
    
    typedef numerical_algos::lin_solvers::default_monitor<
        vec_ops_real,log_t> monitor_t;
    
    typedef nonlinear_operators::circle<
        vec_ops_real, 
        Blocks_x_> circle_t;

    typedef nonlinear_operators::linear_operator_circle<
        vec_ops_real, circle_t> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_circle<
        vec_ops_real, circle_t, lin_op_t> prec_t;

    typedef numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        lin_op_t,
        prec_t,
        vec_ops_real,
        monitor_t,
        log_t,
        numerical_algos::lin_solvers::cgs> sherman_morrison_linear_system_solve_t;


    typedef deflation::newton_method_extended::convergence_strategy<
        vec_ops_real, 
        circle_t, 
        log_t> convergence_newton_def_t;

    typedef deflation::solution_storage<vec_ops_real> sol_storage_def_t;

    typedef deflation::system_operator_deflation<
        vec_ops_real, 
        circle_t,
        lin_op_t,
        sherman_morrison_linear_system_solve_t,
        sol_storage_def_t> system_operator_def_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        vec_ops_real, 
        circle_t,
        system_operator_def_t, 
        convergence_newton_def_t, 
        real /* point solution class here instead of real!*/ 
        > newton_def_t;
    
    typedef nonlinear_operators::newton_method::convergence_strategy<
        vec_ops_real, 
        circle_t, 
        log_t> convergence_newton_t;
    
    typedef nonlinear_operators::system_operator<
        vec_ops_real, 
        circle_t,
        lin_op_t,
        sherman_morrison_linear_system_solve_t
        > system_operator_t;
        
    typedef numerical_algos::newton_method::newton_solver<
        vec_ops_real, 
        circle_t,
        system_operator_t, 
        convergence_newton_t
        > newton_t;

    typedef deflation::deflation_operator<
        vec_ops_real,
        newton_def_t,
        circle_t,
        sol_storage_def_t,
        log_t>deflation_operator_t;

    typedef continuation::system_operator_continuation<
        vec_ops_real, 
        circle_t,
        lin_op_t,
        sherman_morrison_linear_system_solve_t,
        log_t
        > system_operator_cont_t;

    typedef continuation::newton_method_extended::convergence_strategy<
        vec_ops_real, 
        circle_t, 
        log_t> convergence_newton_cont_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        vec_ops_real, 
        circle_t,
        system_operator_cont_t, 
        convergence_newton_cont_t, 
        real /* point solution class here instead of real!*/ 
        > newton_cont_t;

    typedef continuation::predictor_adaptive<
        vec_ops_real,
        log_t
        > predictor_cont_t;

    typedef continuation::advance_solution<
        vec_ops_real,
        log_t,
        newton_cont_t,
        circle_t,
        system_operator_cont_t,
        predictor_cont_t
        >advance_step_cont_t;

    typedef continuation::initial_tangent<
        vec_ops_real,
        circle_t, 
        lin_op_t,
        sherman_morrison_linear_system_solve_t
        > tangent_0_cont_t;

    typedef typename vec_ops_real::vector_type real_vec; 


#endif    