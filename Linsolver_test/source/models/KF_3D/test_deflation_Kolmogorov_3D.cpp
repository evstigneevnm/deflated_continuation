#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <common/macros.h>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>

#include <deflation/system_operator_deflation.h>
#include <deflation/solution_storage.h>
#include <deflation/convergence_strategy.h>
#include <deflation/deflation_operator.h>

#include <numerical_algos/newton_solvers/newton_solver.h>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>

#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>

#include <models/KF_3D/test_deflation_typedefs_Kolmogorov_3D.h>

int main(int argc, char const *argv[])
{
    using vec_file_ops_t = gpu_file_operations<gpu_vector_operations_t>;


    if(argc != 4)
    {
        std::cout << argv[0] << " alpha R N:\n 0<alpha<=1, R is the Reynolds number, N = 2^n- discretization in one direction\n";
        return(0);       
    }

    real alpha = std::atof(argv[1]);
    real Rey = std::atof(argv[2]);
    size_t N = std::atoi(argv[3]);
    int one_over_alpha = int(1/alpha);

    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Testing deflation.\nUsing alpha = " << alpha << ", Reynolds = " << Rey << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    
    init_cuda(-1);

    //linsolver control
    unsigned int lin_solver_max_it = 2000;
    real lin_solver_tol = 1.0e-1;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 3;
    //newton deflation control
    unsigned int newton_def_max_it = 2000;
    real newton_def_tol = 1.0e-9;
    real Power = 2.0;
    real update_weight = 0.5;


    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    size_t Nv = real(3*(Nx*Ny*Mz-1));
    real norm_wight = std::sqrt(Nv);


    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);

    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(Nv, CUBLAS);
    vec_file_ops_t file_ops(vec_ops);

    KF_3D_t *KF_3D = new KF_3D_t(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);


    log_t *log = new log_t();
    log->set_verbosity(1);
    log_t *log3 = new log_t();
    log3->set_verbosity(0);

    lin_op_t *Ax = new lin_op_t(KF_3D);
    prec_t *prec = new prec_t(KF_3D);
    monitor_t *mon;
    
    //setup deflation system
    sherman_morrison_linear_system_solve_t *SM = new sherman_morrison_linear_system_solve_t(prec, vec_ops, log3);
    mon = &SM->get_linsolver_handle()->monitor();
    mon->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle()->set_basis_size(basis_sz);

    convergence_newton_def_t *conv_newton_def = new convergence_newton_def_t(vec_ops, log, newton_def_tol, newton_def_max_it, update_weight, true );

    sol_storage_def_t *sol_storage_def = new sol_storage_def_t(vec_ops, 50, norm_wight, Power);
    system_operator_def_t *system_operator_def = new system_operator_def_t(vec_ops, Ax, SM, sol_storage_def);
    newton_def_t *newton_def = new newton_def_t(vec_ops, system_operator_def, conv_newton_def);

    //setup linear system:
    mon = &SM->get_linsolver_handle_original()->monitor();
    mon->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle_original()->set_basis_size(basis_sz);    
    convergence_newton_t *conv_newton = new convergence_newton_t(vec_ops, log, newton_def_tol, newton_def_max_it, real(0.5) );
    system_operator_t *system_operator = new system_operator_t(vec_ops, Ax, SM);
    newton_t *newton = new newton_t(vec_ops, system_operator, conv_newton);

  
    deflation_operator_t *deflation_op = new deflation_operator_t(vec_ops, log, newton_def, 5);

    
    deflation_op->execute_all(Rey, KF_3D, sol_storage_def);
    //deflation_op->find_add_solution(Rey, KF_3D, sol_storage_def);
    

    unsigned int p=0;
    for(auto &x: *sol_storage_def)
    {        
        std::string f_name_save("res_" + std::to_string(p) + ".dat");
        std::string f_name_vec("vec_" + std::to_string(p) + ".pos");
        std::string f_name_abs("abs_" + std::to_string(p) + ".pos");      
        file_ops.write_vector(f_name_save, (vec&)x);
        KF_3D->write_solution_vec(f_name_vec, (vec&)x);
        KF_3D->write_solution_abs(f_name_abs, (vec&)x);
        p++;
    }
   


    
    delete deflation_op;
    delete KF_3D;
    delete prec;
    delete SM;
    delete log;
    delete log3;
    delete conv_newton_def;
    delete sol_storage_def;
    delete system_operator_def;
    delete newton_def;

    delete conv_newton;
    delete system_operator;
    delete newton;

    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;
    delete CUBLAS;
    delete CUFFT_C2R;
    
    return 0;
}