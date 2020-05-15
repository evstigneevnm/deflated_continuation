#include <cmath>
#include <iostream>
#include <cstdio>
#include <string>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/cufft_wrap.h>


//problem dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>
//problem dependant ends
//problem dependant
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>
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
    
    if(argc!=5)
    {
        printf("Usage: %s path_to_project dS S N, where:\n",argv[0]);
        printf("    path_to_project is the relative path to the storage of bifurcation diagram data;\n");  
        printf("    dS - continuation step;\n");
        printf("    S - number of continuation steps;\n");
        printf("    N - discretization size in one direction.\n");
        return 0;
    }
    typedef SCALAR_TYPE real;

// problem parameters
    size_t N = atoi(argv[4]);
    size_t Nx = N, Ny = N;
    real a_val = real(2.0);
    real b_val = real(4.0);    
//problem parameters ends
    
    std::string path_to_prject_(argv[1]);
    real dS = atof(argv[2]);
    unsigned int S = atoi(argv[3]);

    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_real;
    typedef gpu_vector_operations<complex> vec_ops_complex;
    typedef gpu_vector_operations<real> vec_ops_real_im;
    typedef cufft_wrap_R2C<real> fft_t;
    typedef typename vec_ops_real::vector_type real_vec; 
    typedef typename vec_ops_complex::vector_type complex_vec;
    typedef typename vec_ops_real_im::vector_type real_im_vec;

    typedef gpu_file_operations<vec_ops_real> files_real_t;
    typedef gpu_file_operations<vec_ops_real_im> files_real_im_t;

    typedef utils::log_std log_t;
    typedef numerical_algos::lin_solvers::default_monitor<
        vec_ops_real, log_t> monitor_t;
    typedef nonlinear_operators::Kuramoto_Sivashinskiy_2D<
        fft_t, 
        vec_ops_real, 
        vec_ops_complex, 
        vec_ops_real_im,
        Blocks_x_, 
        Blocks_y_> KS_2D_t;

    typedef nonlinear_operators::linear_operator_KS_2D<
        vec_ops_real_im, KS_2D_t> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_KS_2D<
        vec_ops_real_im, KS_2D_t, lin_op_t> prec_t;

    typedef container::knots<real> knots_t;


    init_cuda(1); // )(PCI) where PCI is the GPU PCI ID

    //linsolver control
    unsigned int lin_solver_max_it = 1500;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 4;
    real lin_solver_tol = 5.0e-3; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
    bool is_small_alpha = false;

    //newton control
    unsigned int newton_max_it = 300;
    unsigned int newton_def_max_it = 400;
    unsigned int newton_cont_max_it = 100;
    real newton_tol = 1.0e-9;
    real newton_def_tol = 1.0e-9;
    real newton_cont_tol = 1.0e-8;
    //skipping files:
    unsigned int skip_files_ = 100;


    fft_t *CUFFT_C2R = new fft_t(Nx, Ny);
    size_t My=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real vec_ops_R(Nx*Ny, CUBLAS);
    vec_ops_complex vec_ops_C(Nx*My, CUBLAS);
    vec_ops_real_im vec_ops_R_im(Nx*My-1, CUBLAS);

    files_real_t file_ops( (vec_ops_real*) &vec_ops_R);
    files_real_im_t file_ops_im( (vec_ops_real_im*) &vec_ops_R_im);    
    //CUDA GRIDS
    dim3 Blocks; dim3 Grids; dim3 Grids_F;
    KS_2D_t KS2D(a_val, b_val, Nx, Ny, (vec_ops_real*) &vec_ops_R, (vec_ops_complex*) &vec_ops_C, (vec_ops_real_im*) &vec_ops_R_im, CUFFT_C2R);
    KS2D.get_cuda_grid(Grids, Grids_F, Blocks);
    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);


    log_t log;
    log_t log_linsolver;
    log_linsolver.set_verbosity(1);       

    typedef main_classes::deflation_continuation<vec_ops_real_im, files_real_im_t, log_t, monitor_t, KS_2D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator> deflation_continuation_t;

    deflation_continuation_t DC( (vec_ops_real_im*) &vec_ops_R_im, (files_real_im_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KS_2D_t*) &KS2D, path_to_prject_, skip_files_ );


    DC.set_linsolver(lin_solver_tol, lin_solver_max_it, use_precond_resid, resid_recalc_freq, basis_sz);
    DC.set_extended_linsolver(lin_solver_tol, lin_solver_max_it, is_small_alpha, use_precond_resid, resid_recalc_freq, basis_sz);
    DC.set_newton(newton_tol, newton_max_it, real(1.0), true);
    DC.set_newton_continuation(newton_cont_tol, newton_cont_max_it, real(1.0), true);
    DC.set_newton_deflation(newton_def_tol, newton_def_max_it, real(1.0), true);
    
    DC.set_steps(S, dS);
    DC.set_deflation_knots({3.0, 4.5, 5.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5, 30.0});
    
    DC.execute("bifurcation_diagram.dat");

    return 0;
}
