#ifndef __OVERSCREEN_BREAKDOWN_TEST_NEWTON_DEFLATION_H__
#define __OVERSCREEN_BREAKDOWN_TEST_NEWTON_DEFLATION_H__


    using T =  SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<T>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<T, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using ob_prob_t = nonlinear_operators::overscreening_breakdown<vec_ops_t, mat_ops_t>;
    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;
    using params_t = params_s<T>;
    using timer_type = scfd::utils::system_timer_event;    
    

    using lin_op_t = nonlinear_operators::linear_operator_overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_t>;
    using lin_op_shifted_t = nonlinear_operators::linear_operator_overscreening_breakdown_shifted<vec_ops_t, mat_ops_t, ob_prob_t>;
    using prec_t = nonlinear_operators::preconditioner_overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_t, lin_op_t>;
    using prec_shifted_t = nonlinear_operators::preconditioner_overscreening_breakdown_shifted<vec_ops_t, mat_ops_t, ob_prob_t, lin_op_shifted_t>;

    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t, log_t>;
    using lin_solver_t = numerical_algos::lin_solvers::exact_wrapper<lin_op_t, prec_t, vec_ops_t, monitor_t, log_t>;
    using lin_solver_shifted_t = numerical_algos::lin_solvers::exact_wrapper<lin_op_shifted_t, prec_shifted_t, vec_ops_t, monitor_t, log_t>;    

    using sherman_morrison_linear_system_solve_t = 
    numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        lin_op_t,
        prec_t,
        vec_ops_t,
        monitor_t,
        log_t,
        numerical_algos::lin_solvers::exact_wrapper>;

    using convergence_newton_def_t = deflation::newton_method_extended::convergence_strategy<
        vec_ops_t, 
        ob_prob_t, 
        log_t> ;

    using sol_storage_def_t = deflation::solution_storage<vec_ops_t>;

    using system_operator_def_t = deflation::system_operator_deflation<
        vec_ops_t, 
        ob_prob_t,
        lin_op_t,
        sherman_morrison_linear_system_solve_t,
        sol_storage_def_t> ;

    using newton_def_t = numerical_algos::newton_method_extended::newton_solver_extended<
        vec_ops_t, 
        ob_prob_t,
        system_operator_def_t, 
        convergence_newton_def_t,
        T /* point solution class here instead of real!*/ 
        > ;
    
    using convergence_newton_t = nonlinear_operators::newton_method::convergence_strategy<
        vec_ops_t, 
        ob_prob_t, 
        log_t> ;
    
    using system_operator_t = nonlinear_operators::system_operator<
        vec_ops_t, 
        ob_prob_t,
        lin_op_t,
        lin_solver_t
        >;
        
    using newton_t = numerical_algos::newton_method::newton_solver<
        vec_ops_t, 
        ob_prob_t,
        system_operator_t, 
        convergence_newton_t
        > ;


    using deflation_operator_t = deflation::deflation_operator<
        vec_ops_t,
        newton_def_t,
        ob_prob_t,
        sol_storage_def_t,
        log_t
        >;



#endif    
