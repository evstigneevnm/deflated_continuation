#ifndef __TEST_DEFLATION_TYPEDEFS_H__
#define __TEST_DEFLATION_TYPEDEFS_H__
#define Blocks_x_ 64
#define Blocks_y_ 16

    typedef SCALAR_TYPE real;
    typedef utils::log_std log_t;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_real;
    typedef gpu_vector_operations<complex> vec_ops_complex;
    typedef gpu_vector_operations<real> vec_ops_real_im;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef Kuramoto_Sivashinskiy_2D<cufft_type, 
            vec_ops_real, 
            vec_ops_complex, 
            vec_ops_real_im,
            Blocks_x_, 
            Blocks_y_> KS_2D;
    
    typedef numerical_algos::newton_method_extended::convergence_strategy<
            vec_ops_real_im, 
            KS_2D, 
            log_t> convergence_newton_def;

    
    typedef numerical_algos::newton_method_extended::newton_solver_extended<
            vec_ops_real_im, 
            system_operator_def, 
            convergence_newton_def, 
            real /* point class here instead of real!*/ 
            > newton_def;
    
    typedef typename gpu_vector_operations_real::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex::vector_type complex_vec;
    typedef typename gpu_vector_operations_real_reduced::vector_type real_im_vec;


#endif    