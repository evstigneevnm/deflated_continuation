#include <cmath>
#include <iostream>
#include <cstdio>
#include <string>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/cufft_wrap.h>


//vector dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>
//vector dependant ends
//problem dependant
#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
//problem dependant ends

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <main/deflation_continuation.hpp>


#ifndef Blocks_x_
    #define Blocks_x_ 32
#endif
#ifndef Blocks_y_
    #define Blocks_y_ 16
#endif

int main(int argc, char const *argv[])
{
    
    if(argc!=7)
    {
        printf("Usage: %s path_to_project dS S alpha R N, where:\n",argv[0]);
        printf("    path_to_project is the relative path to the storage of bifurcation diagram data;\n");
        printf("    dS - continuation step;\n");
        printf("    S - number of continuation steps;\n");
        printf("    0<alpha<=1;\n");
        printf("    R is the Reynolds number;\n"); 
        printf("    N = 2^n- discretization in one direction.");
        return 0;
    }
    typedef SCALAR_TYPE real;
    
    std::string path_to_prject_(argv[1]);
    real dS = atof(argv[2]);
    unsigned int S = atoi(argv[3]);

    real alpha = std::atof(argv[4]);
    real Rey = std::atof(argv[5]);
    size_t N = std::atoi(argv[6]);
    int one_over_alpha = int(1/alpha);

    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Running bifurcation diagram construction.\nUsing alpha = " << alpha << ", Reynolds = " << Rey << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    typedef utils::log_std log_t;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_real_t;
    typedef gpu_vector_operations<complex> vec_ops_complex_t;
    typedef gpu_vector_operations<real> vec_ops_t;
    typedef cufft_wrap_R2C<real> cufft_type;
    
    typedef gpu_file_operations<vec_ops_t> files_t;

    typedef numerical_algos::lin_solvers::default_monitor<
        vec_ops_t,log_t> monitor_t;
    
    typedef nonlinear_operators::Kolmogorov_3D<cufft_type, 
            vec_ops_real_t, 
            vec_ops_complex_t, 
            vec_ops_t,
            Blocks_x_, Blocks_y_> KF_3D_t;

    typedef nonlinear_operators::linear_operator_K_3D<
        vec_ops_t, KF_3D_t> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_K_3D<
        vec_ops_t, KF_3D_t, lin_op_t> prec_t;

    typedef container::knots<real> knots_t;


    init_cuda(-1); // )(PCI) where PCI is the GPU PCI ID

    //linsolver control
    unsigned int lin_solver_max_it = 2000;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 3;
    real lin_solver_tol = 4.0e-3; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
    bool is_small_alpha = false;

    //newton control
    unsigned int newton_def_max_it = 2000;
    unsigned int newton_def_cont_it = 2000;
    real newton_def_tol = 1.0e-9;
    real newton_cont_tol = 1.0e-9;
    real newton_interp_tol = 1.0e-9;
    //skipping files:
    unsigned int skip_files_ = 100;


    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real_t vec_ops_R(Nx*Ny*Nz, CUBLAS);
    vec_ops_complex_t vec_ops_C(Nx*Ny*Mz, CUBLAS);
    vec_ops_t vec_ops(3*(Nx*Ny*Mz-1), CUBLAS);

    files_t file_ops_im( (vec_ops_t*) &vec_ops);    
    //CUDA GRIDS

    KF_3D_t KF3D(alpha, Nx, Ny, Nz, (vec_ops_real_t*) &vec_ops_R, (vec_ops_complex_t*) &vec_ops_C, (vec_ops_t*) &vec_ops, CUFFT_C2R);

    log_t log;
       

    typedef main_classes::deflation_continuation<
        vec_ops_t, files_t, log_t, monitor_t, KF_3D_t, 
        lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, 
        nonlinear_operators::system_operator> deflation_continuation_t;

    deflation_continuation_t DC( (vec_ops_t*) &vec_ops, (files_t*) &file_ops_im, (log_t*) &log, (KF_3D_t*) &KF3D, path_to_prject_, skip_files_ );


    DC.set_linsolver(lin_solver_tol, lin_solver_max_it, use_precond_resid, resid_recalc_freq, basis_sz);
    DC.set_extended_linsolver(lin_solver_tol, lin_solver_max_it, is_small_alpha, use_precond_resid, resid_recalc_freq, basis_sz);
    DC.set_newton(newton_cont_tol, newton_def_cont_it, real(0.5), true);
    DC.set_steps(S, dS);
    DC.set_deflation_knots({3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0});
    
    DC.execute("bifurcation_diagram.dat");

    return 0;
}
