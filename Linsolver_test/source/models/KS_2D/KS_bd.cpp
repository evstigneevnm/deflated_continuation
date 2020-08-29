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
#include <common/gpu_matrix_vector_operations.h>
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
#include <main/stability_continuation.hpp>
#include <main/plot_diagram_to_pos.hpp>

#ifndef Blocks_x_
    #define Blocks_x_ 32
#endif
#ifndef Blocks_y_
    #define Blocks_y_ 16
#endif

int main(int argc, char const *argv[])
{
    
    if((argc!=5)&&(argc!=3&&(argc!=4)))
    {
        printf("For continuation:\n");
        printf("Usage: %s path_to_project N dS S, where:\n",argv[0]);
        printf("    path_to_project is the relative path to the storage of bifurcation diagram data;\n");  
        printf("    N - discretization size in one direction.\n");
        printf("    dS - continuation step;\n");
        printf("    S - number of continuation steps;\n");
        printf("For editing:\n");
        printf("Usage: %s path_to_project N, where:\n",argv[0]);
        printf("    path_to_project is the relative path to the storage of bifurcation diagram data;\n");  
        printf("    N - discretization size in one direction.\n");        
        printf("For stability analysis:\n");
        printf("Usage: %s path_to_project N m, where:\n",argv[0]);
        printf("    path_to_project is the relative path to the storage of bifurcation diagram data;\n");  
        printf("    N - discretization size in one direction.\n");
        printf("    m - size of the Krylov subspace in Arnoldi process.\n");    
        printf("For plotting:\n");
        printf("Usage: %s path_to_project N p, where:\n",argv[0]);
        printf("    path_to_project is the relative path to the storage of bifurcation diagram data;\n");  
        printf("    N - discretization size in one direction.\n");
        printf("    p - just the char 'p' for plotting.\n");             
        return 0;
    }
    typedef SCALAR_TYPE real;
    
// problem parameters
    char what_to_execute = 'E';
    std::string path_to_prject_(argv[1]);
    size_t N = atoi(argv[2]);

    real dS = 0.1;
    unsigned int S = 100;
    unsigned int m_Krylov = 1;
    if(argc==5)
    {
        dS = atof(argv[3]);
        S = atoi(argv[4]);
        what_to_execute = 'D';
    }
    if(argc == 4)
    {
        if(argv[3][0] == 'p')
        {
            what_to_execute = 'P';
        }
        else
        {
            m_Krylov = atoi(argv[3]);
            what_to_execute = 'S';
        }
    }
    size_t Nx = N, Ny = N;
    real a_val = real(2.0);
    real b_val = real(4.0);    
//problem parameters ends

    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_real;

    typedef gpu_vector_operations<complex> vec_ops_complex;
    typedef gpu_vector_operations<real> vec_ops_real_im;
    typedef cufft_wrap_R2C<real> fft_t;
    typedef typename vec_ops_real::vector_type real_vec; 
    typedef typename vec_ops_complex::vector_type complex_vec;
    typedef typename vec_ops_real_im::vector_type real_im_vec;

    typedef gpu_matrix_vector_operations<real, real_im_vec> mat_vec_ops_t;
    typedef typename mat_vec_ops_t::matrix_type mat;


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


    init_cuda(-1); // )(PCI) where PCI is the GPU PCI ID

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

    mat_vec_ops_t mat_vec_ops(Nx*My-1, m_Krylov, CUBLAS);


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

    if( (what_to_execute=='D')||(what_to_execute == 'E') )
    {

        typedef main_classes::deflation_continuation<vec_ops_real_im, files_real_im_t, log_t, monitor_t, KS_2D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator> deflation_continuation_t;

        deflation_continuation_t DC( (vec_ops_real_im*) &vec_ops_R_im, (files_real_im_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KS_2D_t*) &KS2D, path_to_prject_, skip_files_ );


        DC.set_linsolver(lin_solver_tol, lin_solver_max_it, use_precond_resid, resid_recalc_freq, basis_sz);
        DC.set_extended_linsolver(lin_solver_tol, lin_solver_max_it, is_small_alpha, use_precond_resid, resid_recalc_freq, basis_sz);
        DC.set_newton(newton_tol, newton_max_it, real(1.0), true);
        DC.set_newton_continuation(newton_cont_tol, newton_cont_max_it, real(1.0), true);
        DC.set_newton_deflation(newton_def_tol, newton_def_max_it, real(1.0), true);
        DC.set_steps(S, dS);
        //DC.set_deflation_knots({3.0, 4.5, 5.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5, 30.0});
        DC.set_deflation_knots({3.0, 8.05, 8.1, 8.15, 8.2, 8.5, 8.676, 8.912, 9.0, 9.1, 9.12, 9.15, 9.35, 9.45, 9.56, 9.7, 9.89, 10.1, 10.3, 10.6, 10.8, 10.9, 11.4, 30.0});
        
        if(what_to_execute == 'D')
        {
            DC.use_analytical_solution(false);
            DC.execute("bifurcation_diagram.dat");
        }
        else if(what_to_execute == 'E')
        {
            DC.edit("bifurcation_diagram.dat");
        }
    }
    else if(what_to_execute == 'S')
    {
        typedef main_classes::stability_continuation<vec_ops_real_im, mat_vec_ops_t, files_real_im_t, log_t, monitor_t, KS_2D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator> stability_t;

        vec_ops_real_im vec_ops_small(m_Krylov, CUBLAS);
        mat_vec_ops_t mat_ops_small(m_Krylov, m_Krylov, CUBLAS);

        stability_t ST( (vec_ops_real_im*) &vec_ops_R_im, (mat_vec_ops_t*) &mat_vec_ops, (vec_ops_real_im*) &vec_ops_small, (mat_vec_ops_t*) &mat_ops_small, (files_real_im_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KS_2D_t*) &KS2D, path_to_prject_, skip_files_ );

        ST.set_linsolver(lin_solver_tol, lin_solver_max_it, use_precond_resid, resid_recalc_freq, basis_sz);
        ST.set_newton(newton_tol, newton_max_it, real(1.0), true);

        ST.set_liner_operator_stable_eigenvalues_halfplane(real(1.0)); // +1 for RHP, -1 for LHP (default) 
        ST.edit("stability_diagram.dat");
        ST.execute("bifurcation_diagram.dat", "stability_diagram.dat");
    }
    else if(what_to_execute == 'P')
    {
        typedef main_classes::plot_diagram_to_pos<vec_ops_real_im, mat_vec_ops_t, files_real_im_t, log_t, monitor_t, KS_2D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator> plot_diagram_t;       

        plot_diagram_t PD( (vec_ops_real_im*) &vec_ops_R_im, (files_real_im_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KS_2D_t*) &KS2D, path_to_prject_);

        PD.set_linsolver(lin_solver_tol, lin_solver_max_it, use_precond_resid, resid_recalc_freq, basis_sz);
        PD.set_newton(newton_tol, newton_max_it, real(1.0), true);

        PD.set_plot_pos_sols(3);
        PD.execute("bifurcation_diagram.dat", "stability_diagram.dat");
    }
    else
    {
        std::cout << "No correct usage scheme was selected." << std::endl;
    }


    return 0;
}
