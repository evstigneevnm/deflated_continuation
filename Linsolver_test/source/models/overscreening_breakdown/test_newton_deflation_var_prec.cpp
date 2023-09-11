#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>

#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_params.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/linear_operator_overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/preconditioner_overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/linear_operator_overscreening_breakdown_shifted.h>
#include <nonlinear_operators/overscreening_breakdown/preconditioner_overscreening_breakdown_shifted.h>
#include <nonlinear_operators/overscreening_breakdown/system_operator.h>
#include <nonlinear_operators/overscreening_breakdown/convergence_strategy.h>

#include <deflation/system_operator_deflation.h>
#include <deflation/solution_storage.h>
#include <deflation/convergence_strategy.h>
#include <deflation/deflation_operator.h>

#include <numerical_algos/newton_solvers/newton_solver.h>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>

#include <common/macros.h>
#include <common/cpu_file_operations.h>
#include <common/cpu_vector_operations_var_prec.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>
#include <contrib/scfd/include/scfd/utils/system_timer_event.h>

#include <models/overscreening_breakdown/test_newton_deflation_var_prec.h>


int main(int argc, char const *argv[])
{

    if(argc != 8)
    {
        std::cout << argv[0] << " DOF sigma L gamma delta mu u0" << std::endl;
        std::cout << " DOF - deg of freedom, sigma>=0 is the parameter value" << std::endl;
        std::cout << " L>0 - mapping value, gamma>=0 - regularization value of the first part of the rhs" << std::endl;
        std::cout << " delta>=0 - 4-th derivative value, mu>=0 - rhs second part multiplayer, u0>0 - initial condition value" << std::endl;
        return(0);       
    }
    

    size_t N = std::stoi(argv[1]);
    T sigma = std::stof(argv[2]);
    T L = std::stof(argv[3]);
    T gamma = std::stof(argv[4]);
    T delta = std::stof(argv[5]);
    T mu = std::stof(argv[6]);
    T u0 = std::stof(argv[7]);

    params_t params(N, 0, {sigma, L, gamma, delta, mu, u0} );

    T lin_solver_tol = 1.0e-10;
    unsigned int newton_def_max_it = 50;
    unsigned int lin_solver_max_it = 5;
    T newton_def_tol = 1.0e-10;
    T Power = 3.0;
    T newton_wight = 1.0;
    T norm_wight = sqrt(N);

    vec_ops_t vec_ops(N);
    mat_ops_t mat_ops(vec_ops.get_vector_size(), vec_ops.get_vector_size(), &vec_ops );
    vec_file_ops_t vec_file_ops(&vec_ops);
    
    ob_prob_t ob_prob(&vec_ops, &mat_ops, params );

    monitor_t *mon;

    log_t log;
    log.set_verbosity(1);
    log_t log3;
    log3.set_verbosity(0);

    lin_op_t Ax(&ob_prob);
    prec_t prec(&ob_prob);
    
    //setup deflation system
    sherman_morrison_linear_system_solve_t SM(&prec, &vec_ops, &log3);
    mon = &SM.get_linsolver_handle()->monitor();
    mon->init(lin_solver_tol, T(0), lin_solver_max_it);

    convergence_newton_def_t conv_newton_def(&vec_ops, &log, newton_def_tol, newton_def_max_it, newton_wight, true );

    sol_storage_def_t sol_storage_def(&vec_ops, 50, norm_wight, Power);
    sol_storage_def.set_ignore_zero();

    system_operator_def_t system_operator_def(&vec_ops, &Ax, &SM, &sol_storage_def);
    newton_def_t newton_def(&vec_ops, &system_operator_def, &conv_newton_def);

    //setup linear system:
    mon = &SM.get_linsolver_handle_original()->monitor();
    mon->init(lin_solver_tol, T(0), lin_solver_max_it);
    // mon->set_save_convergence_history(true);
    // mon->set_divide_out_norms_by_rel_base(true);

    // convergence_newton_t conv_newton(&vec_ops, &log, newton_def_tol, newton_def_max_it, T(0.5) );
    // system_operator_t system_operator(&vec_ops, &Ax, &SM);
    // newton_t newton(&vec_ops, &system_operator, &conv_newton);

    deflation_operator_t deflation_op(&vec_ops, &log, &newton_def, 5);

    deflation_op.execute_all(sigma, &ob_prob, &sol_storage_def);
    //deflation_op->find_add_solution(Rey, &ob_prob, sol_storage_def);
    
//*


    unsigned int p=0;
    for(auto &x: sol_storage_def)
    {        
        std::stringstream f_name;
        f_name << "solution_" << p << "_for_" << params.N << "_" << params.L << params.L << "_mu" << mu << "_var_prec.dat";
        ob_prob.write_solution_basis(f_name.str(), (T_vec&)x);
        log.info_f("solution %i, norm = %le\n", p, static_cast<double>(vec_ops.norm( (T_vec&) x)) );
        p++;
    }
    

//*/


    return 0;
}