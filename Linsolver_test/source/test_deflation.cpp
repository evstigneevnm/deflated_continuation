#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include "macros.h"

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D.h>
#include <nonlinear_operators/linear_operator_KS_2D.h>
#include <nonlinear_operators/preconditioner_KS_2D.h>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>

#include <deflation/system_operator_deflation.h>
#include <deflation/solution_storage.h>

#include <numerical_algos/newton_solvers/convergence_strategy.h>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>


#include "gpu_file_operations.h"
#include "gpu_vector_operations.h"
#include "test_deflation_typedefs.h"

int main(int argc, char const *argv[])
{
    
    //linsolver control
    unsigned int lin_solver_max_it = 200;
    real lin_solver_tol = 1.0e-9;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 3;
    //newton deflation control
    unsigned int newton_def_max_it = 1000;
    real newton_def_tol = 1.0e-9;


    init_cuda(-1);
    size_t Nx=64;
    size_t Ny=64;

    fft_t *CUFFT_C2R = new fft_t(Nx, Ny);
    size_t My=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);

    vec_ops_real *vec_ops_R = new vec_ops_real(Nx*Ny, CUBLAS);
    vec_ops_complex *vec_ops_C = new vec_ops_complex(Nx*My, CUBLAS);
    vec_ops_real_im *vec_ops_R_im = new vec_ops_real_im(Nx*My-1, CUBLAS);
    //CUDA GRIDS
    dim3 Blocks; dim3 Grids; dim3 Grids_F;
    KS_2D *KS2D = new KS_2D(2.0, 4.0, Nx, Ny, vec_ops_R, vec_ops_C, vec_ops_R_im, CUFFT_C2R);
    KS2D->get_cuda_grid(Grids, Grids_F, Blocks);
    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);
    real lambda_0 = 7.1;
    log_t *log = new log_t();
    lin_op_t Ax(KS2D);
    prec_t *prec = new prec_t(KS2D);
    monitor_t *mon;
    
    sherman_morrison_linear_system_solve_t *SM = new sherman_morrison_linear_system_solve_t(prec, vec_ops_R_im, log);
    mon = &SM->get_linsolver_handle()->monitor();
    mon->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle()->set_basis_size(basis_sz);


    convergence_newton_def_t *conv_newton_def = new convergence_newton_def_t(vec_ops_R_im, KS2D, log, newton_def_tol, newton_def_max_it, real(1) );

    sol_storage_def_t *sol_storage_def = new sol_storage_def_t(vec_ops_R_im, 20, real(15.0));

    system_operator_def_t *system_operator_def = new system_operator_def_t(vec_ops_R_im, KS2D, SM, sol_storage_def);
    newton_def_t *newton_def = new newton_def_t(vec_ops_R_im, system_operator_def, conv_newton_def);

    real_im_vec u_in, u_out;
    vec_ops_R_im->init_vector(u_in); vec_ops_R_im->start_use_vector(u_in);
    vec_ops_R_im->init_vector(u_out); vec_ops_R_im->start_use_vector(u_out);
    real_vec u_out_ph;
    vec_ops_R->init_vector(u_out_ph); vec_ops_R->start_use_vector(u_out_ph);
    vec_ops_R->assign_random(u_out_ph);

    KS2D->fourier_solution(u_out_ph, u_in);

    real lambda;
    bool found_solution = false;
    unsigned int solution_number = 0;
    while(found_solution)
    {
        found_solution = newton_def->solve(Ax, u_in, lambda_0, u_out, lambda);
        if(found_solution)
        {
            sol_storage_def->push(u_out);
            KS2D->physical_solution(u_out, u_out_ph);
            std::ostringstream stringStream;
            stringStream << "u_out_" << solution_number << ".dat";
            gpu_file_operations::write_matrix<real>(stringStream.str(), Nx, Ny, u_out_ph);
            solution_number++;
            
            vec_ops_R->assign_random(u_out_ph);
            
            KS2D->fourier_solution(u_out_ph, u_in);

            printf("\n================= found %i solutions =================\n", solution_number);
        }
    }
    printf("\n lambda = %lf\n", (double)lambda);
    printf("\n================= found %i solutions =================\n", solution_number);



    //gpu_file_operations::write_vector<real>("u_im_out.dat", Nx*My-1, u_out, 3);
    //gpu_file_operations::write_vector<real>("u_out_vec.dat", Nx*Ny, u_out_ph, 3);

    vec_ops_R_im->stop_use_vector(u_in); vec_ops_R_im->free_vector(u_in);
    vec_ops_R_im->stop_use_vector(u_out); vec_ops_R_im->free_vector(u_out);
    vec_ops_R->stop_use_vector(u_out_ph); vec_ops_R->free_vector(u_out_ph);

    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops_R_im;
    delete KS2D;
    delete prec;
    delete SM;
    delete log;
    delete conv_newton_def;
    delete sol_storage_def;
    delete system_operator_def;
    delete newton_def;
    return 0;
}