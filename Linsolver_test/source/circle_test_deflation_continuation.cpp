#include <cmath>
#include <iostream>
#include <cstdio>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>

//problem dependant
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/circle/preconditioner_circle.h>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <nonlinear_operators/circle/system_operator.h>
//problem dependant ends

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/cgs.h>
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

//problem dependant
#include <gpu_file_operations.h>
#include <gpu_vector_operations.h>
#include "circle_test_deflation_continuation_typedefs.h"
//problem dependant ends

int main(int argc, char const *argv[])
{
    
    if(argc!=4)
    {
        printf("Usage: %s lambda_0 dS S\n   lambda_0 - starting parameter\n   dS - continuation step\n   S - number of continuation steps\n",argv[0]);
        return 0;
    }
    size_t Nx = 1; //size of the vector variable. 1 in this case
    real lambda0 = atof(argv[1]);
    real dS = atof(argv[2]);
    unsigned int S = atoi(argv[3]);





    init_cuda(6); // )(PCI) where PCI is the GPU PCI ID
    real norm_wight = std::sqrt(real(Nx));
    real Rad = 1.0;



    //linsolver control
    unsigned int lin_solver_max_it = 1500;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 1;
    real lin_solver_tol = 5.0e-3; //relative tolerance wrt rhs vector. For Krylov-Newton method can be set low
    
    //newton control
    unsigned int newton_def_max_it = 350;
    unsigned int newton_def_cont_it = 100;
    real newton_def_tol = 1.0e-9;
    real newton_cont_tol = 1.0e-9;


    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real *vec_ops_R = new vec_ops_real(Nx, CUBLAS);

   //CUDA GRIDS
    dim3 Blocks; dim3 Grids;
    circle_t *CIRCLE = new circle_t(Rad, Nx, vec_ops_R);
    CIRCLE->get_cuda_grid(Grids, Blocks);
    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    log_t *log = new log_t();
    lin_op_t *Ax = new lin_op_t(CIRCLE);
    prec_t *prec = new prec_t(CIRCLE);
    monitor_t *mon;
    monitor_t *mon_orig; 
    //setup deflation system
    sherman_morrison_linear_system_solve_t *SM = new sherman_morrison_linear_system_solve_t(prec, vec_ops_R, log);
    mon = &SM->get_linsolver_handle()->monitor();
    mon->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    mon->out_min_resid_norm();
    //SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
    //SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
    //SM->get_linsolver_handle()->set_basis_size(basis_sz);
    SM->is_small_alpha(false);

    convergence_newton_def_t *conv_newton_def = new convergence_newton_def_t(vec_ops_R, log, newton_def_tol, newton_def_max_it, real(1), true );
    sol_storage_def_t *sol_storage_def = new sol_storage_def_t(vec_ops_R, 50, norm_wight);
    system_operator_def_t *system_operator_def = new system_operator_def_t(vec_ops_R, Ax, SM, sol_storage_def);
    newton_def_t *newton_def = new newton_def_t(vec_ops_R, system_operator_def, conv_newton_def);

    //setup linear system:
    mon_orig = &SM->get_linsolver_handle_original()->monitor();
    mon_orig->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon_orig->set_save_convergence_history(true);
    mon_orig->set_divide_out_norms_by_rel_base(true);
    mon_orig->out_min_resid_norm();
    //SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
    //SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
    //SM->get_linsolver_handle_original()->set_basis_size(basis_sz);    
    convergence_newton_t *conv_newton = new convergence_newton_t(vec_ops_R, log, newton_cont_tol, newton_def_max_it, real(1) );
    system_operator_t *system_operator = new system_operator_t(vec_ops_R, Ax, SM);
    newton_t *newton = new newton_t(vec_ops_R, system_operator, conv_newton);

    //setup continuation system:
    predictor_cont_t* predict = new predictor_cont_t(vec_ops_R, log, dS, 0.25, 20);
    system_operator_cont_t* system_operator_cont = new system_operator_cont_t(vec_ops_R, Ax, SM);
    convergence_newton_cont_t *conv_newton_cont = new convergence_newton_cont_t(vec_ops_R, log, newton_cont_tol, newton_def_cont_it, real(1), true);
    newton_cont_t* newton_cont = new newton_cont_t(vec_ops_R, system_operator_cont, conv_newton_cont);
    advance_step_cont_t* continuation_step = new advance_step_cont_t(vec_ops_R, log, system_operator_cont, newton_cont, predict);
    tangent_0_cont_t* init_tangent = new tangent_0_cont_t(vec_ops_R, Ax, SM);


    

    deflation_operator_t *deflation_op = new deflation_operator_t(vec_ops_R, log, newton_def, 5);

    
    //deflation_op->execute_all(lambda0, CIRCLE, sol_storage_def);
    //deflation_op->find_solution(lambda0, CIRCLE, sol_storage_def);
    

    // unsigned int p=0;
    // for(auto &x: *sol_storage_def)
    // {        
    //     KS2D->physical_solution((real_im_vec&)x, u_out_ph);
    //     std::ostringstream stringStream;
    //     stringStream << "u_out_" << (p++) << ".dat";
    //     gpu_file_operations::write_matrix<real>(stringStream.str(), Nx, Ny, u_out_ph);
    // }

    real_vec x0, x0_s, x1, x1_s, x1_p, d_x, f, xxx, xxx1;
    real_vec x0p, x_1_g;

    vec_ops_R->init_vector(x0); vec_ops_R->start_use_vector(x0);
    vec_ops_R->init_vector(x1_p); vec_ops_R->start_use_vector(x1_p);
    vec_ops_R->init_vector(x0_s); vec_ops_R->start_use_vector(x0_s);
    vec_ops_R->init_vector(x1); vec_ops_R->start_use_vector(x1);
    vec_ops_R->init_vector(x1_s); vec_ops_R->start_use_vector(x1_s);
    vec_ops_R->init_vector(d_x); vec_ops_R->start_use_vector(d_x);
    vec_ops_R->init_vector(f); vec_ops_R->start_use_vector(f);
    vec_ops_R->init_vector(xxx); vec_ops_R->start_use_vector(xxx);
    vec_ops_R->init_vector(xxx1); vec_ops_R->start_use_vector(xxx1);
    vec_ops_R->init_vector(x0p); vec_ops_R->start_use_vector(x0p);
    vec_ops_R->init_vector(x_1_g); vec_ops_R->start_use_vector(x_1_g);    

    // x0 = (*sol_storage_def)[0].get_ref();
    // printf("solutions in container = %i\n",sol_storage_def->get_size());
    // (*sol_storage_def)[0].copy(x0);

    real lambda0_s, lambda1_s, lambda1_p;
    real lambda1;
    real d_lambda;
    real lambda_0p, lambda_1_g;

    
    lambda0 = real(0);//std::sqrt(real(2))*0.5;
    vec_ops_R->assign_scalar(real(1),x0);

    init_tangent->execute(CIRCLE, -1, x0, lambda0, x0_s, lambda0_s);
    real norm = 1;

    real* x_host = (real*)malloc(Nx*sizeof(real));
    real* xp_host = (real*)malloc(Nx*sizeof(real));

    std::ofstream file_diag("diagram.dat", std::ofstream::out);
    
    file_diag << lambda0 << " " << vec_ops_R->norm_l2(x0)<< " " << lambda0 << " " << vec_ops_R->norm_l2(x0) << std::endl;

    for(unsigned int s=0;s<S;s++)
    {
        continuation_step->solve(CIRCLE, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
        //to check predicted values:
        predict->apply(x0p, lambda_0p, x_1_g, lambda_1_g);
        device_2_host_cpy(xp_host, x0p, Nx);

        vec_ops_R->assign(x1, x0);
        vec_ops_R->assign(x1_s, x0_s);
        lambda0 = lambda1;
        lambda0_s = lambda1_s;

        device_2_host_cpy(x_host, x1, Nx);


        file_diag << lambda_0p << " " << xp_host[0] << " " << lambda1 << " " << x_host[0] << std::endl; //vec_ops_R->norm_l2(x1)
        std::flush(file_diag);
    }
    file_diag.close();

    free(x_host);
    free(xp_host);

    CIRCLE->F(x1, lambda1, f);
    norm = vec_ops_R->norm_l2(f);
    printf("\n===(%le, %le)===\n", lambda1, norm);


    vec_ops_R->stop_use_vector(x0); vec_ops_R->free_vector(x0);
    vec_ops_R->stop_use_vector(x0_s); vec_ops_R->free_vector(x0_s);
    vec_ops_R->stop_use_vector(x1_p); vec_ops_R->free_vector(x1_p);
    vec_ops_R->stop_use_vector(x1); vec_ops_R->free_vector(x1);
    vec_ops_R->stop_use_vector(x1_s); vec_ops_R->free_vector(x1_s);
    vec_ops_R->stop_use_vector(d_x); vec_ops_R->free_vector(d_x);
    vec_ops_R->stop_use_vector(f); vec_ops_R->free_vector(f);
    vec_ops_R->stop_use_vector(xxx); vec_ops_R->free_vector(xxx);
    vec_ops_R->stop_use_vector(xxx1); vec_ops_R->free_vector(xxx1);
    vec_ops_R->stop_use_vector(x_1_g); vec_ops_R->free_vector(x_1_g);
    vec_ops_R->stop_use_vector(x0p); vec_ops_R->free_vector(x0p);

    delete init_tangent;
    delete continuation_step;
    delete system_operator_cont;
    delete predict;
    delete deflation_op;
    delete CIRCLE;
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

    return 0;
}
