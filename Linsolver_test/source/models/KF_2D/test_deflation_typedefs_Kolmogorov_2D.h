#ifndef __TEST_DEFLATION_TYPEDEFS_H__
#define __TEST_DEFLATION_TYPEDEFS_H__
#define Blocks_x_ 32
#define Blocks_y_ 16


    typedef SCALAR_TYPE real;
    typedef utils::log_std log_t;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_t;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex_t;
    typedef gpu_vector_operations<real> gpu_vector_operations_t;
    typedef cufft_wrap_R2C<real> cufft_type;
    
    typedef numerical_algos::lin_solvers::default_monitor<
        gpu_vector_operations_t,log_t> monitor_t;
    
    typedef nonlinear_operators::Kolmogorov_2D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> KF_2D_t;

    typedef nonlinear_operators::linear_operator_K_2D<
        gpu_vector_operations_t, KF_2D_t> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_K_2D<
        gpu_vector_operations_t, KF_2D_t, lin_op_t> prec_t;

    typedef numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        lin_op_t,
        prec_t,
        gpu_vector_operations_t,
        monitor_t,
        log_t,
        numerical_algos::lin_solvers::bicgstabl> sherman_morrison_linear_system_solve_t;

    typedef deflation::newton_method_extended::convergence_strategy<
        gpu_vector_operations_t, 
        KF_2D_t, 
        log_t> convergence_newton_def_t;

    typedef deflation::solution_storage<gpu_vector_operations_t> sol_storage_def_t;

    typedef deflation::system_operator_deflation<
        gpu_vector_operations_t, 
        KF_2D_t,
        lin_op_t,
        sherman_morrison_linear_system_solve_t,
        sol_storage_def_t> system_operator_def_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        gpu_vector_operations_t, 
        KF_2D_t,
        system_operator_def_t, 
        convergence_newton_def_t, 
        real /* point solution class here instead of real!*/ 
        > newton_def_t;
    
    typedef nonlinear_operators::newton_method::convergence_strategy<
        gpu_vector_operations_t, 
        KF_2D_t, 
        log_t> convergence_newton_t;
    
    typedef nonlinear_operators::system_operator<
        gpu_vector_operations_t, 
        KF_2D_t,
        lin_op_t,
        sherman_morrison_linear_system_solve_t
        > system_operator_t;
        
    typedef numerical_algos::newton_method::newton_solver<
        gpu_vector_operations_t, 
        KF_2D_t,
        system_operator_t, 
        convergence_newton_t
        > newton_t;


    typedef deflation::deflation_operator<
        gpu_vector_operations_t,
        newton_def_t,
        KF_2D_t,
        sol_storage_def_t,
        log_t
        >deflation_operator_t;

    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;


#endif    