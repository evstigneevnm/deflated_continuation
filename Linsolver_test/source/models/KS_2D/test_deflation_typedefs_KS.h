#ifndef __TEST_DEFLATION_TYPEDEFS_H__
#define __TEST_DEFLATION_TYPEDEFS_H__
#define Blocks_x_ 32
#define Blocks_y_ 16


    typedef SCALAR_TYPE real;
    typedef utils::log_std log_t;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_real;
    typedef gpu_vector_operations<complex> vec_ops_complex;
    typedef gpu_vector_operations<real> vec_ops_real_im;
    typedef cufft_wrap_R2C<real> fft_t;
    
    typedef numerical_algos::lin_solvers::default_monitor<
        vec_ops_real_im,log_t> monitor_t;
    
    typedef nonlinear_operators::Kuramoto_Sivashinskiy_2D<
        fft_t, 
        vec_ops_real, 
        vec_ops_complex, 
        vec_ops_real_im,
        Blocks_x_, 
        Blocks_y_> KS_2D;

    typedef nonlinear_operators::linear_operator_KS_2D<
        vec_ops_real_im, KS_2D> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_KS_2D<
        vec_ops_real_im, KS_2D, lin_op_t> prec_t;

    typedef numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        lin_op_t,
        prec_t,
        vec_ops_real_im,
        monitor_t,
        log_t,
        numerical_algos::lin_solvers::bicgstabl> sherman_morrison_linear_system_solve_t;

    typedef deflation::newton_method_extended::convergence_strategy<
        vec_ops_real_im, 
        KS_2D, 
        log_t> convergence_newton_def_t;

    typedef deflation::solution_storage<vec_ops_real_im> sol_storage_def_t;

    typedef deflation::system_operator_deflation<
        vec_ops_real_im, 
        KS_2D,
        lin_op_t,
        sherman_morrison_linear_system_solve_t,
        sol_storage_def_t> system_operator_def_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        vec_ops_real_im, 
        KS_2D,
        system_operator_def_t, 
        convergence_newton_def_t, 
        real /* point solution class here instead of real!*/ 
        > newton_def_t;
    
    typedef nonlinear_operators::newton_method::convergence_strategy<
        vec_ops_real_im, 
        KS_2D, 
        log_t> convergence_newton_t;
    
    typedef nonlinear_operators::system_operator<
        vec_ops_real_im, 
        KS_2D,
        lin_op_t,
        sherman_morrison_linear_system_solve_t
        > system_operator_t;
        
    typedef numerical_algos::newton_method::newton_solver<
        vec_ops_real_im, 
        KS_2D,
        system_operator_t, 
        convergence_newton_t
        > newton_t;


    typedef deflation::deflation_operator<
        vec_ops_real_im,
        newton_def_t,
        KS_2D,
        sol_storage_def_t,
        log_t
        >deflation_operator_t;

    typedef typename vec_ops_real::vector_type real_vec; 
    typedef typename vec_ops_complex::vector_type complex_vec;
    typedef typename vec_ops_real_im::vector_type real_im_vec;


#endif    