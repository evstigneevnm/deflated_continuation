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
#include <numerical_algos/lin_solvers/gmres.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>

#include <deflation/system_operator_deflation.h>
#include <deflation/system_operator_deflation_with_translation.h>
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
    
    if((argc != 7)&&(argc != 9))
    {
        std::cout << argv[0] << " alpha R N high_prec homotopy nz [folder \"file_name_regex\"]:\n 0<alpha<=1, R is the Reynolds number, N = 2^n- discretization in one direction\n high_prec=(0/1) use(1) or not(0) high precision reduciton methods\n nz = 0,1,2,... is the forcing term in Z direction\n  optional: folder is the path to the folder, where previously found solutions are\n  optional: \"file_name_regex\" IN QUOTES is the regular expression for the solution files  \n";
        return(0);       
    }

    real alpha = std::stof(argv[1]);
    real Rey = std::stof(argv[2]);
    size_t N = std::stoi(argv[3]);
    int high_prec = std::stoi(argv[4]);
    real homotopy = std::stof(argv[5]);
    int nz = std::stoi(argv[6]);
    int one_over_alpha = int(1/alpha);
    std::string folder_saved_solutions;
    std::string regex_saved_solutions;

    if(argc == 9)
    {
        folder_saved_solutions = std::string(argv[7]);
        regex_saved_solutions = std::string(argv[8]);
    }

    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Testing deflation.\nUsing " << "nz = " << nz << "alpha = " << alpha << ", high precision = " << high_prec << ", homotopy = " << homotopy << ", Reynolds = " << Rey << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;
    if( !folder_saved_solutions.empty() )
    {
        std::cout << "folder_saved_solutions = " << folder_saved_solutions << ", regex_saved_solutions = " << regex_saved_solutions << std::endl;
    }
    
    init_cuda(-1);

    //linsolver control
    unsigned int lin_solver_max_it = 300;
    real lin_solver_tol = 6.0e-1;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 800;
    //newton deflation control
    unsigned int newton_def_max_it = 1000;
    real newton_def_tol = 5.0e-9;
    real Power = 1.0;
    real newton_update = 0.5;


    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    size_t Nv = 3*(Nx*Ny*Mz-1);
    real norm_wight = std::sqrt(Nv);


    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);

    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(Nv, CUBLAS);
    vec_file_ops_t file_ops(vec_ops);

    if(high_prec == 1)
    {
        vec_ops_R -> use_high_precision();
        vec_ops_C -> use_high_precision();
        vec_ops -> use_high_precision();
    }

    KF_3D_t *KF_3D = new KF_3D_t(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R, true, nz);

    KF_3D->set_homotopy_value(homotopy);
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

    convergence_newton_def_t *conv_newton_def = new convergence_newton_def_t(vec_ops, log, newton_def_tol, newton_def_max_it, newton_update, true );

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
    convergence_newton_t *conv_newton = new convergence_newton_t(vec_ops, log, newton_def_tol, newton_def_max_it, real(1.0) );
    system_operator_t *system_operator = new system_operator_t(vec_ops, Ax, SM);
    
    sol_storage_def->set_ignore_zero();
    // vec x1;
    // vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
    // KF_3D->exact_solution(Rey, x1);
    // sol_storage_def->set_known_solution(x1);
    // vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);

    unsigned int p=0;
    if( !folder_saved_solutions.empty() )
    {
        auto solution_files = file_operations::match_file_names(folder_saved_solutions, regex_saved_solutions);
        vec x1;
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        for(auto &v: solution_files)
        {   

            file_ops.read_vector(v, (vec&)x1);
            sol_storage_def->push_back(x1);
            std::cout << "added data from " << v << " to deflation storage as solution number "  << p << std::endl;
            p++;
        }
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    }


    newton_t *newton = new newton_t(vec_ops, system_operator, conv_newton);

  
    deflation_operator_t *deflation_op = new deflation_operator_t(vec_ops, log, newton_def, 5);

    deflation_op->save_norms("deflation_norms_NS.dat");
    deflation_op->execute_all(Rey, KF_3D, sol_storage_def);
    //deflation_op->find_add_solution(Rey, KF_3D, sol_storage_def);
    

    unsigned int counter = 0; //to output only new data
    for(auto &x: *sol_storage_def)
    {        
        counter++;
        if( counter >= p)
        {
            std::string f_name_vec("vec_" + std::to_string(p) + ".pos");
            std::string f_name_abs("abs_" + std::to_string(p) + ".pos");  
            std::string f_name_save("res_" + std::to_string(p) + ".dat");
            file_ops.write_vector(f_name_save, (vec&)x);  
            KF_3D->write_solution_vec(f_name_vec, (vec&)x);
            KF_3D->write_solution_abs(f_name_abs, (vec&)x);

            p++;
        }
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