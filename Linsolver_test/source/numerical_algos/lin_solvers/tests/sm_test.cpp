#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
//#include <numerical_algos/lin_solvers/bicgstab.h>
#include <common/file_operations.h>
#include <common/cpu_vector_operations.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#include "test_matrix_liner_operator.h"

using namespace numerical_algos::lin_solvers;
using namespace numerical_algos::sherman_morrison_linear_system;

using real = SCALAR_TYPE;
using vec_ops_t = cpu_vector_operations<real>;
using vector_t = typename vec_ops_t::vector_type;
using mat_ops_t = cpu_matrix_vector_operations_var_prec<vec_ops_t>;
using matrix_t = typename mat_ops_t::matrix_type;
using system_operator_t = system_operator<mat_ops_t>;
using prec_operator_t = prec_operator<vec_ops_t, mat_ops_t>;


typedef utils::log_std log_t;
typedef default_monitor<vec_ops_t,log_t> monitor_t;
//typedef bicgstabl<system_operator,prec_operator,vec_ops_t,monitor_t,log_t> lin_solver_bicgstabl_t;
// Sherman Morrison class
typedef sherman_morrison_linear_system_solve<system_operator_t,prec_operator_t,vec_ops_t,monitor_t,log_t,bicgstabl> sherman_morrison_linear_system_solve_t;

int main(int argc, char **args)
{
    if (argc != 7) {
        std::cout << "USAGE: " << std::string(args[0]) << " <maximum iterations> <relative tolerance> <use preconditioned residual> <residual recalculation frequency> <basis size> <use_small_alpha_optimization>"  << std::endl;
        return 0;
    }

    int max_iters = std::stoi(args[1]);
    real rel_tol = std::stof(args[2]);
    bool use_precond_resid = std::atoi(args[3]);
    int resid_recalc_freq = std::atoi(args[4]);
    int basis_sz = std::atoi(args[5]);
    bool small_alpha = static_cast<bool>(std::stoi(args[6]));

    int sz = file_operations::read_matrix_size_square("./dat_files/A.dat");
    matrix_t A, iP;
    vector_t x, x0, b, c, d;
    vec_ops_t vec_ops(sz);
    mat_ops_t mat_ops(sz, sz, &vec_ops);
    mat_ops.init_matrices(A, iP); mat_ops.start_use_matrices(A, iP);
    vec_ops.init_vectors(x0, x, c, d, b); vec_ops.start_use_vectors(x0, x, c, d, b);

    real alpha=1.0e-9;
    real beta=1.9;
    real v=0.0;

    file_operations::read_matrix<real>("./dat_files/A.dat",  sz, sz, A);
    file_operations::read_matrix<real>("./dat_files/iP.dat",  sz, sz, iP);
    file_operations::read_vector<real>("./dat_files/x0.dat",  sz, x0);
    file_operations::read_vector<real>("./dat_files/b.dat",  sz, b);
    file_operations::read_vector<real>("./dat_files/c.dat",  sz, c);
    file_operations::read_vector<real>("./dat_files/d.dat",  sz, d);

    std::cout << "matrix size = " << sz << std::endl;

    std::cout << sz << std::endl;
    log_t log;
    system_operator_t Ax(sz, &mat_ops, A);
    prec_operator_t prec(sz, &vec_ops, &mat_ops, iP);
    
    monitor_t *mon;
    monitor_t *mon_original;
    
    //lin_solver_bicgstabl_t lin_solver_bicgstabl(&vec_ops, &log);
    //lin_solver_bicgstabl.set_preconditioner(&prec);
    //mon = &lin_solver_bicgstabl.monitor();

    sherman_morrison_linear_system_solve_t *SM = new sherman_morrison_linear_system_solve_t(&prec, &vec_ops, &log);   
    mon = &SM->get_linsolver_handle()->monitor();
    mon_original = &SM->get_linsolver_handle_original()->monitor();

    mon->init(rel_tol, real(0), max_iters);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    mon_original->init(rel_tol, real(0), max_iters);
    mon_original->set_save_convergence_history(true);
    mon_original->set_divide_out_norms_by_rel_base(true);

    bool res_flag;
    int iters_performed;

    // lin_solver_bicgstabl.set_use_precond_resid(use_precond_resid);
    // lin_solver_bicgstabl.set_resid_recalc_freq(resid_recalc_freq);
    // lin_solver_bicgstabl.set_basis_size(basis_sz);
    // bool res_flag = lin_solver_bicgstabl.solve(Ax, b, x);

    SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle()->set_basis_size(basis_sz);
    SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle_original()->set_basis_size(basis_sz);

    SM->is_small_alpha(small_alpha);

    std::cout << "\n ========= \ntesting: rank1_update(A)u = b; v=f(u,b) \n";
    vec_ops.assign_scalar(1.0, x);
    res_flag = SM->solve(Ax, c, d, alpha, b, beta, x, v);
    iters_performed = mon->iters_performed();
    log.info_f("linsolver total iterations = %i", iters_performed);
    if (res_flag) 
        log.info("lin_solver returned success result");
    else
        log.error("lin_solver returned fail result");

    // for(auto &x: mon->convergence_history() )
    // {
    //     std::cout << x.first << " " << x.second << std::endl;
    // }

    file_operations::write_vector<real>("./dat_files/x_rank1.dat", sz, x);
    std::cout << " v = " << v << std::endl;
//  add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
    vec_ops.add_mul(1.0, x0, -1.0, x);
    std::cout << "||x-x0|| = " << vec_ops.norm(x);

    //test (beta A - 1/alpha d c^T) u = b;
    std::cout << "\n ========= \ntesting: (beta A - 1/alpha d c^T) u = b \n";
    vec_ops.assign_scalar(1.0, x);
    res_flag = SM->solve(beta, Ax, alpha, c, d, b, x);
    
    iters_performed = mon->iters_performed();
    log.info_f("linsolver total iterations = %i", iters_performed);
    
    if (res_flag)
        log.info("lin_solver returned success result");
    else
        log.error("lin_solver returned fail result");    
    
    // for(auto &xxx: mon->convergence_history() )
    // {
    //     std::cout << xxx.first << " " << xxx.second << std::endl;
    // } 

    vec_ops.add_mul(1.0, x0, -1.0, x);
    std::cout << "||x-x0|| = " << vec_ops.norm(x);

    file_operations::write_vector<real>("./dat_files/x_sm.dat", sz, x);

    std::cout << "\n ========= \ntesting: A u = b \n";
    res_flag = SM->solve(Ax, b, x);
    iters_performed = mon->iters_performed();
    log.info("linsolver total iterations = %i", iters_performed);
    if (res_flag)
        log.info("lin_solver returned success result");
    else
        log.error("lin_solver returned fail result");    
    
    // for(auto &x: mon_original->convergence_history() )
    // {
    //     std::cout << x.first << " " << x.second << std::endl;
    // }  

    file_operations::write_vector<real>("./dat_files/x_orig.dat", sz, x);
           

    mat_ops.stop_use_matrices(A, iP); mat_ops.free_matrices(A, iP);
    vec_ops.stop_use_vectors(x0, x, c, d, b); vec_ops.free_vectors(x0, x, c, d, b);

    return 0;
}