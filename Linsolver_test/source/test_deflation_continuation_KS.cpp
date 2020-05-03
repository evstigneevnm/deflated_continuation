#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <common/macros.h>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>

#include <deflation/system_operator_deflation.h>
#include <deflation/solution_storage.h>
#include <deflation/convergence_strategy.h>
#include <deflation/deflation_operator.h>

#include <numerical_algos/newton_solvers/newton_solver.h>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>

#include <continuation/predictor_adaptive.h>
#include <continuation/system_operator_continuation.h>
#include <continuation/advance_solution.h>
#include <continuation/initial_tangent.h>
#include <continuation/convergence_strategy.h>

#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include "test_deflation_continuation_typedefs.h"

int main(int argc, char const *argv[])
{
    
    if(argc!=6)
    {
        printf("Usage: %s Nx Ny lambda_0 dS S\n   lambda_0 - starting parameter\n   dS - continuation step\n   S - number of continuation steps\n",argv[0]);
        return 0;
    }
    size_t Nx = atoi(argv[1]);
    size_t Ny = atoi(argv[2]);
    real lambda0 = atof(argv[3]);
    real dS = atof(argv[4]);
    unsigned int S = atoi(argv[5]);





    init_cuda(1);
    real norm_wight = std::sqrt(real(Nx*Ny));

    //linsolver control
    unsigned int lin_solver_max_it = 1500;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 4;
    real lin_solver_tol = 5.0e-2;
    
    //newton control
    unsigned int newton_def_max_it = 350;
    unsigned int newton_def_cont_it = 100;
    real newton_def_tol = 1.0e-9;
    real newton_cont_tol = 1.0e-9;

    real a_val = real(2);
    real b_val = real(4);

    fft_t *CUFFT_C2R = new fft_t(Nx, Ny);
    size_t My=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);

    vec_ops_real *vec_ops_R = new vec_ops_real(Nx*Ny, CUBLAS);
    vec_ops_complex *vec_ops_C = new vec_ops_complex(Nx*My, CUBLAS);
    vec_ops_real_im *vec_ops_R_im = new vec_ops_real_im(Nx*My-1, CUBLAS);
    //CUDA GRIDS
    dim3 Blocks; dim3 Grids; dim3 Grids_F;
    KS_2D *KS2D = new KS_2D(a_val, b_val, Nx, Ny, vec_ops_R, vec_ops_C, vec_ops_R_im, CUFFT_C2R);
    KS2D->get_cuda_grid(Grids, Grids_F, Blocks);
    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);
    log_t *log = new log_t();
    lin_op_t *Ax = new lin_op_t(KS2D);
    prec_t *prec = new prec_t(KS2D);
    monitor_t *mon;
    monitor_t *mon_orig; 
    //setup deflation system
    sherman_morrison_linear_system_solve_t *SM = new sherman_morrison_linear_system_solve_t(prec, vec_ops_R_im, log);
    mon = &SM->get_linsolver_handle()->monitor();
    mon->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    mon->out_min_resid_norm();
    SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle()->set_basis_size(basis_sz);
    SM->is_small_alpha(false);

    convergence_newton_def_t *conv_newton_def = new convergence_newton_def_t(vec_ops_R_im, log, newton_def_tol, newton_def_max_it, real(1), true );
    sol_storage_def_t *sol_storage_def = new sol_storage_def_t(vec_ops_R_im, 50, norm_wight);
    system_operator_def_t *system_operator_def = new system_operator_def_t(vec_ops_R_im, Ax, SM, sol_storage_def);
    newton_def_t *newton_def = new newton_def_t(vec_ops_R_im, system_operator_def, conv_newton_def);

    //setup linear system:
    mon_orig = &SM->get_linsolver_handle_original()->monitor();
    mon_orig->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon_orig->set_save_convergence_history(true);
    mon_orig->set_divide_out_norms_by_rel_base(true);
    mon_orig->out_min_resid_norm();
    SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle_original()->set_basis_size(basis_sz);    
    convergence_newton_t *conv_newton = new convergence_newton_t(vec_ops_R_im, log, newton_cont_tol, newton_def_max_it, real(1) );
    system_operator_t *system_operator = new system_operator_t(vec_ops_R_im, Ax, SM);
    newton_t *newton = new newton_t(vec_ops_R_im, system_operator, conv_newton);

    //setup continuation system:
    predictor_cont_t* predict = new predictor_cont_t(vec_ops_R_im, log, dS, 0.0765, 0.1, 30);
    system_operator_cont_t* system_operator_cont = new system_operator_cont_t(vec_ops_R_im, Ax, SM);
    convergence_newton_cont_t *conv_newton_cont = new convergence_newton_cont_t(vec_ops_R_im, log, newton_cont_tol, newton_def_cont_it, real(1), true);
    newton_cont_t* newton_cont = new newton_cont_t(vec_ops_R_im, system_operator_cont, conv_newton_cont);
    advance_step_cont_t* continuation_step = new advance_step_cont_t(vec_ops_R_im, log, system_operator_cont, newton_cont, predict, 'S');
    tangent_0_cont_t* init_tangent = new tangent_0_cont_t(vec_ops_R_im, Ax, SM);


    real_vec u_out_ph;
    vec_ops_R->init_vector(u_out_ph); vec_ops_R->start_use_vector(u_out_ph);
    

    deflation_operator_t *deflation_op = new deflation_operator_t(vec_ops_R_im, log, newton_def, 5);

    
    //deflation_op->execute_all(lambda0, KS2D, sol_storage_def);
    deflation_op->find_add_solution(lambda0, KS2D, sol_storage_def);
    

    unsigned int p=0;
    for(auto &x: *sol_storage_def)
    {        
        KS2D->physical_solution((real_im_vec&)x, u_out_ph);
        std::ostringstream stringStream;
        stringStream << "u_out_" << (p++) << ".dat";
        //gpu_file_operations::write_matrix<real>(stringStream.str(), Nx, Ny, u_out_ph);
    }

    real_im_vec x0, x0_s, x1, x1_s, x1_p, d_x, f, xxx, xxx1;
    real_im_vec x0p, x_1_g;
    vec_ops_R_im->init_vector(x0); vec_ops_R_im->start_use_vector(x0);
    vec_ops_R_im->init_vector(x1_p); vec_ops_R_im->start_use_vector(x1_p);
    vec_ops_R_im->init_vector(x0_s); vec_ops_R_im->start_use_vector(x0_s);
    vec_ops_R_im->init_vector(x1); vec_ops_R_im->start_use_vector(x1);
    vec_ops_R_im->init_vector(x1_s); vec_ops_R_im->start_use_vector(x1_s);
    vec_ops_R_im->init_vector(d_x); vec_ops_R_im->start_use_vector(d_x);
    vec_ops_R_im->init_vector(f); vec_ops_R_im->start_use_vector(f);
    vec_ops_R_im->init_vector(xxx); vec_ops_R_im->start_use_vector(xxx);
    vec_ops_R_im->init_vector(xxx1); vec_ops_R_im->start_use_vector(xxx1);
    vec_ops_R_im->init_vector(x0p); vec_ops_R_im->start_use_vector(x0p);
    vec_ops_R_im->init_vector(x_1_g); vec_ops_R_im->start_use_vector(x_1_g);


    x0 = (*sol_storage_def)[0].get_ref();
    printf("solutions in container = %i\n",sol_storage_def->get_size());
    (*sol_storage_def)[0].copy(x0);
    KS2D->physical_solution((real_im_vec&)x0, u_out_ph);
    //gpu_file_operations::write_matrix<real>("u_out_C0.dat", Nx, Ny, u_out_ph);

    real lambda0_s, lambda1_s, lambda1_p;
    real lambda1;
    real d_lambda;
    real lambda_0p, lambda_1_g;

    init_tangent->execute(KS2D, -1, x0, lambda0, x0_s, lambda0_s);
    real norm = 1;

/*



    predict->reset_tangent_space(x0, lambda0, x0_s, lambda0_s);
    predict->apply(x1_p, lambda1_p, x1, lambda1);

    printf("\nlambda0=== %le ===", lambda0);
    printf("\nlambda0_s=== %le ===", lambda0_s);
    printf("\nlambda1_p=== %le ===", lambda1_p);
    printf("\nlambda1=== %le ===", lambda1);
    printf("\n||x_ph=== %le ===", vec_ops_R->norm(u_out_ph));
    printf("\n||x0||=== %le ===", vec_ops_R_im->norm(x0));


    printf("\n");

//*
    while(norm>newton_def_tol)
    {        
        KS2D->set_linearization_point(x1, lambda1);
        KS2D->jacobian_alpha(x1, lambda1, xxx);
        KS2D->F(x1, lambda1, f);
        vec_ops_R_im->add_mul_scalar(real(0), real(-1), f);
        vec_ops_R_im->assign_mul(real(1), x1, real(-1), x1_p, d_x);
        
        real x_proj = vec_ops_R_im->scalar_prod(d_x, x0_s);
        real lambda_proj = (lambda1-lambda1_p)*lambda0_s;
        real proj = -(x_proj + lambda_proj);

        printf("proj = %le\n", proj);
        printf("lambda0_s = %le\n", lambda0_s);
        printf("||f||=%le\n", vec_ops_R_im->norm(f));
        printf("||xxx||=%le\n", vec_ops_R_im->norm(xxx));
        printf("||x0_s||=%le\n", vec_ops_R_im->norm(x0_s));

        SM->solve((*Ax), x0_s, xxx, lambda0_s, f, proj, d_x, d_lambda);
    //insert residual estimation
        Ax->apply(d_x,xxx1);
    vec_ops_R_im->add_mul(d_lambda, xxx, xxx1);
    vec_ops_R_im->add_mul(real(-1), f, xxx1);
    norm = vec_ops_R_im->norm(xxx1);
    real norm1 = vec_ops_R_im->scalar_prod(x0_s,d_x)+lambda0_s*d_lambda-proj;
    printf("===lin_system_norm=%le,%le====\n",(double)norm, (double)norm1);


    //ends
        lambda1+=d_lambda;
        vec_ops_R_im->add_mul(real(1), d_x, x1);

        KS2D->F(x1, lambda1, f);
        norm = vec_ops_R_im->norm(f);
        printf("===(%le, %le)===\n\n\n", lambda1, norm);

    }
//*/
    std::ofstream file_diag("diagram.dat", std::ofstream::out);
    
    file_diag << std::setprecision(10) << lambda0 << " " << vec_ops_R_im->norm_l2(x0) << " " << lambda0 << " " << vec_ops_R_im->norm_l2(x0) << std::endl;
    for(unsigned int s=0;s<S;s++)
    {
        continuation_step->solve(KS2D, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
        predict->apply(x0p, lambda_0p, x_1_g, lambda_1_g);
        vec_ops_R_im->assign(x1,x0);
        vec_ops_R_im->assign(x1_s,x0_s);
        lambda0 = lambda1;
        lambda0_s = lambda1_s;
        
        file_diag << lambda1 << " " << vec_ops_R_im->norm_l2(x0p) << " " << lambda1 << " " << vec_ops_R_im->norm_l2(x1) << std::endl;
        std::flush(file_diag);
    }
    file_diag.close();

    KS2D->F(x1, lambda1, f);
    norm = vec_ops_R_im->norm_l2(f);
    printf("\n===(%le, %le)===", lambda1, norm);

    KS2D->physical_solution((real_im_vec&)x1, u_out_ph);
    //gpu_file_operations::write_matrix<real>("u_out_1.dat", Nx, Ny, u_out_ph);


    vec_ops_R_im->stop_use_vector(x0); vec_ops_R_im->free_vector(x0);
    vec_ops_R_im->stop_use_vector(x0_s); vec_ops_R_im->free_vector(x0_s);
    vec_ops_R_im->stop_use_vector(x1_p); vec_ops_R_im->free_vector(x1_p);
    vec_ops_R_im->stop_use_vector(x1); vec_ops_R_im->free_vector(x1);
    vec_ops_R_im->stop_use_vector(x1_s); vec_ops_R_im->free_vector(x1_s);
    vec_ops_R_im->stop_use_vector(d_x); vec_ops_R_im->free_vector(d_x);
    vec_ops_R_im->stop_use_vector(f); vec_ops_R_im->free_vector(f);
    vec_ops_R_im->stop_use_vector(xxx); vec_ops_R_im->free_vector(xxx);
    vec_ops_R_im->stop_use_vector(xxx1); vec_ops_R_im->free_vector(xxx1);
    vec_ops_R_im->stop_use_vector(x0p); vec_ops_R_im->free_vector(x0p);
    vec_ops_R_im->stop_use_vector(x_1_g); vec_ops_R_im->free_vector(x_1_g);
    
    
    vec_ops_R->stop_use_vector(u_out_ph); vec_ops_R->free_vector(u_out_ph);

    delete init_tangent;
    delete continuation_step;
    delete system_operator_cont;
    delete predict;
    delete deflation_op;
    delete KS2D;
    delete prec;
    delete SM;
    delete log;
    delete conv_newton_def;
    delete sol_storage_def;
    delete system_operator_def;
    delete newton_def;

    delete conv_newton_cont;
    delete conv_newton;
    delete system_operator;
    delete newton;

    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops_R_im;

    return 0;
}
