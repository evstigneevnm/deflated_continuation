#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/gmres.h>
#include <common/file_operations.h>
#include <common/cpu_vector_operations.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>
#include "test_matrix_liner_operator.h"

using namespace numerical_algos::lin_solvers;

using real = SCALAR_TYPE;
using vec_ops_t = cpu_vector_operations<real>;
using vector_t = typename vec_ops_t::vector_type;
using mat_ops_t = cpu_matrix_vector_operations_var_prec<vec_ops_t>;
using matrix_t = typename mat_ops_t::matrix_type;
using system_operator_t = system_operator<mat_ops_t>;
using prec_operator_t = prec_operator<vec_ops_t, mat_ops_t>;


typedef utils::log_std log_t;
typedef default_monitor<vec_ops_t,log_t> monitor_t;
typedef bicgstabl<system_operator_t,prec_operator_t,vec_ops_t,monitor_t,log_t> bicgstabl_t;
typedef bicgstab<system_operator_t,prec_operator_t,vec_ops_t,monitor_t,log_t> bicgstab_t;
using gmres_t = gmres<system_operator_t,prec_operator_t,vec_ops_t,monitor_t,log_t>;


int main(int argc, char **args)
{
    if (argc != 6)
    {
        std::cout << "USAGE: " << std::string(args[0]) << " <maximum iterations> <relative tolerance> <use preconditioned residual> <residual recalculation frequency> <basis size>"  << std::endl;
        if(argc != 1)
            return 0;
        else
            std::cout << "using default values." << std::endl;
    }

    int sz = file_operations::read_matrix_size_square("./dat_files/A.dat");
    int max_iters = sz/2;
    real rel_tol = 1.0e-6;
    bool use_precond_resid = 1;
    int resid_recalc_freq = 3;
    int basis_sz = 8;
    if(argc != 1)
    {
        max_iters = std::stoi(args[1]);
        rel_tol = std::stof(args[2]);
        use_precond_resid = std::atoi(args[3]);
        resid_recalc_freq = std::atoi(args[4]);
        basis_sz = std::atoi(args[5]);
    }


    matrix_t A, iP;
    vector_t x0, b, c, d;
    vec_ops_t vec_ops(sz);
    mat_ops_t mat_ops(sz, sz, &vec_ops);
    mat_ops.init_matrices(A, iP); mat_ops.start_use_matrices(A, iP);
    vec_ops.init_vectors(x0, c, d, b); vec_ops.start_use_vectors(x0, c, d, b);


    file_operations::read_matrix<real>("./dat_files/A.dat",  sz, sz, A);
    file_operations::read_matrix<real>("./dat_files/iP.dat",  sz, sz, iP);
    file_operations::read_vector<real>("./dat_files/x0.dat",  sz, x0);
    file_operations::read_vector<real>("./dat_files/b.dat",  sz, b);

    std::cout << "matrix size = " << sz << std::endl;

    std::cout << sz << std::endl;
    log_t log;
    system_operator_t Ax(sz, &mat_ops, A);
    prec_operator_t prec(sz, &vec_ops, &mat_ops, iP);    
    monitor_t *mon;
    bool res_flag = true;
    {
        vector_t x;
        vec_ops.init_vectors(x); vec_ops.start_use_vectors(x);
        gmres_t::params params;
        params.basis_size = 80;
        params.preconditioner_side = 'R';
        params.reorthogonalization = true;

        gmres_t gmres(&vec_ops, &log, params);
        gmres.set_preconditioner(&prec);
        mon = &gmres.monitor();

        mon->init(rel_tol, real(0), max_iters);
        mon->set_save_convergence_history(true);
        mon->set_divide_out_norms_by_rel_base(true);

        // gmres.set_use_precond_resid(use_precond_resid);
        // gmres.set_resid_recalc_freq(resid_recalc_freq);
        // gmres.set_restarts(basis_sz);

        std::cout << "\n ========= gmres: A u = b \n";
        bool res_flag_l = gmres.solve(Ax, b, x);

        vec_ops.add_mul(1.0, x0, -1.0, x);
        auto res_norm = vec_ops.norm(x)/vec_ops.norm(b);
        std::cout << "||x-x0||/||b|| = " << res_norm << std::endl;
        int iters_performed = mon->iters_performed();
        log.info("gmres total iterations = %i", iters_performed);

        if (res_flag_l)
            log.info("gmres returned success result");
        else
            log.error("gmres returned fail result");   

        if(!res_flag_l)
        {
            res_flag = false;
        }

        vec_ops.stop_use_vectors(x); vec_ops.free_vectors(x);

    }
    {
        vector_t x;
        vec_ops.init_vectors(x); vec_ops.start_use_vectors(x);
        bicgstabl_t bicgstabl(&vec_ops, &log);
        bicgstabl.set_preconditioner(&prec);
        mon = &bicgstabl.monitor();

        mon->init(rel_tol, real(0), max_iters);
        mon->set_save_convergence_history(true);
        mon->set_divide_out_norms_by_rel_base(true);

        bicgstabl.set_use_precond_resid(use_precond_resid);
        bicgstabl.set_resid_recalc_freq(resid_recalc_freq);
        bicgstabl.set_basis_size(basis_sz);

        std::cout << "\n ========= bicgstabl: A u = b \n";
        bool res_flag_l = bicgstabl.solve(Ax, b, x);

        vec_ops.add_mul(1.0, x0, -1.0, x);
        auto res_norm = vec_ops.norm(x)/vec_ops.norm(b);
        std::cout << "||x-x0||/||b|| = " << res_norm << std::endl;
        int iters_performed = mon->iters_performed();
        log.info("bicgstabl total iterations = %i", iters_performed);

        if (res_flag_l)
            log.info("bicgstabl returned success result");
        else
            log.error("bicgstabl returned fail result");   

        if(!res_flag_l)
        {
            res_flag = false;
        }

        vec_ops.stop_use_vectors(x); vec_ops.free_vectors(x);

    }
    {
        vector_t x;
        vec_ops.init_vectors(x); vec_ops.start_use_vectors(x);
        bicgstab_t bicgstab(&vec_ops, &log);
        bicgstab.set_preconditioner(&prec);
        mon = &bicgstab.monitor();

        mon->init(rel_tol, real(0), max_iters);
        mon->set_save_convergence_history(true);
        mon->set_divide_out_norms_by_rel_base(true);

        bicgstab.set_use_precond_resid(use_precond_resid);
        bicgstab.set_resid_recalc_freq(resid_recalc_freq);

        std::cout << "\n ========= bicgstab: A u = b \n";
        bool res_flag_l = bicgstab.solve(Ax, b, x);

        vec_ops.add_mul(1.0, x0, -1.0, x);
        auto res_norm = vec_ops.norm(x)/vec_ops.norm(b);
        std::cout << "||x-x0||/||b|| = " << res_norm << std::endl;
        int iters_performed = mon->iters_performed();
        log.info("bicgstabl total iterations = %i", iters_performed);

        if (res_flag_l)
            log.info("bicgstabl returned success result");
        else
            log.error("bicgstabl returned fail result");   

        if(!res_flag_l)
        {
            res_flag = false;
        }

        vec_ops.stop_use_vectors(x); vec_ops.free_vectors(x);        

    }

    mat_ops.stop_use_matrices(A, iP); mat_ops.free_matrices(A, iP);
    vec_ops.stop_use_vectors(x0, c, d, b); vec_ops.free_vectors(x0, c, d, b);

    if(res_flag)
    {
        std::cout << "PASSED" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
}