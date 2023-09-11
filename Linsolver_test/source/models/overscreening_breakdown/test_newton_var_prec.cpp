#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

// #include <external_libraries/lapack_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>

#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_params.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown.h>

#include <nonlinear_operators/overscreening_breakdown/linear_operator_overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/preconditioner_overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/linear_operator_overscreening_breakdown_shifted.h>
#include <nonlinear_operators/overscreening_breakdown/preconditioner_overscreening_breakdown_shifted.h>

#include <nonlinear_operators/overscreening_breakdown/system_operator.h>
#include <nonlinear_operators/overscreening_breakdown/convergence_strategy.h>

#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/macros.h>
#include <common/cpu_file_operations.h>
#include <common/cpu_vector_operations_var_prec.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>
#include <contrib/scfd/include/scfd/utils/system_timer_event.h>


int main(int argc, char const *argv[])
{
    
    using vec_ops_t = cpu_vector_operations_var_prec;
    using T = typename vec_ops_t::scalar_type;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = cpu_matrix_vector_operations_var_prec<vec_ops_t>;
    using T_mat = typename mat_ops_t::matrix_type;
    using ob_prob_ker_t = nonlinear_operators::overscreening_breakdown_ker<vec_ops_t, mat_ops_t>;
    using vec_file_ops_t = cpu_file_operations<vec_ops_t>;
    using ob_prob_t = nonlinear_operators::overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_ker_t, vec_file_ops_t>;
    using params_t = params_s<T>;
    using timer_t = scfd::utils::system_timer_event;


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

    //linsolver control
    T lin_solver_tol = 1.0e-10;
    //newton deflation control
    unsigned int newton_def_max_it = 500;
    T newton_def_tol = 1.0e-10;

    vec_ops_t vec_ops(N);
    mat_ops_t mat_ops(vec_ops.get_vector_size(), vec_ops.get_vector_size(), &vec_ops );
    vec_file_ops_t vec_file_ops(&vec_ops);
    
    ob_prob_t ob_prob(&vec_ops, &mat_ops, params );

    // linear solver config
    using lin_op_t = nonlinear_operators::linear_operator_overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_t>;
    using lin_op_shifted_t = nonlinear_operators::linear_operator_overscreening_breakdown_shifted<vec_ops_t, mat_ops_t, ob_prob_t>;
    using prec_t = nonlinear_operators::preconditioner_overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_t, lin_op_t>;
    using prec_shifted_t = nonlinear_operators::preconditioner_overscreening_breakdown_shifted<vec_ops_t, mat_ops_t, ob_prob_t, lin_op_shifted_t>;

    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t, log_t>;
    using lin_solver_t = numerical_algos::lin_solvers::exact_wrapper<lin_op_t, prec_t, vec_ops_t, monitor_t, log_t>;
    using lin_solver_shifted_t = numerical_algos::lin_solvers::exact_wrapper<lin_op_shifted_t, prec_shifted_t, vec_ops_t, monitor_t, log_t>;    

    monitor_t *mon, *mon_shifted;

    log_t log;
    log_t log3;
    log3.set_verbosity(1);
    lin_op_t lin_op(&ob_prob);
    prec_t prec(&ob_prob);

//  for stability analysis
    lin_op_shifted_t lin_op_shifted(&vec_ops, &ob_prob); 
    prec_shifted_t prec_shifted(&vec_ops, &ob_prob);


    lin_solver_t lin_solver(&vec_ops, &log3);
    lin_solver.set_preconditioner(&prec);

    mon = &lin_solver.monitor();
    mon->init(lin_solver_tol, 0.0, 10);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);   

    lin_solver_shifted_t lin_solver_shifted(&vec_ops, &log3);
    lin_solver_shifted.set_preconditioner(&prec_shifted);

    mon_shifted = &lin_solver.monitor();
    mon_shifted->init(lin_solver_tol, 0.0, 10);
    mon_shifted->set_save_convergence_history(true);
    mon_shifted->set_divide_out_norms_by_rel_base(true);     




    T_vec x0, x1, dx, x_back, b;

    vec_ops.init_vectors(b, x0, x1, dx, x_back); vec_ops.start_use_vectors(b, x0, x1, dx, x_back);

    

    ob_prob.randomize_vector(x0);
    vec_ops.assign(x0, x_back);
    ob_prob.physical_solution(x0, x1);
    vec_file_ops.write_vector("test_newton_solution_initial_var_prec.dat", x1);

    std::cout << "initial solution norm = " << vec_ops.norm(x0) << std::endl;

    T solution_norm = 1;
    unsigned int iter = 0;
    // T mu_min = 0.005;
    // T mu_0 = 1.0;
    // T mu = mu_0;
    bool ok_flag = true;
    std::vector<T> newton_norm;

    ob_prob.F(x0, sigma, b);
    solution_norm = vec_ops.norm(b);
    newton_norm.push_back(solution_norm);

    
    using convergence_newton_t = nonlinear_operators::newton_method::convergence_strategy<
        vec_ops_t, 
        ob_prob_t, 
        log_t>;

    using system_operator_t = nonlinear_operators::system_operator<
        vec_ops_t,
        ob_prob_t,
        lin_op_t,
        lin_solver_t>;

    using newton_t = numerical_algos::newton_method::newton_solver<
        vec_ops_t,
        ob_prob_t,
        system_operator_t,
        convergence_newton_t>;

    log_t *log_p = &log;
    convergence_newton_t conv_newton(&vec_ops, &log);
    
    system_operator_t system_operator(&vec_ops, &lin_op, &lin_solver);
    vec_ops_t *vec_ops_p = &vec_ops;
    system_operator_t *system_operator_p = &system_operator;
    convergence_newton_t *conv_newton_p = &conv_newton;
    newton_t newton(vec_ops_p, system_operator_p, conv_newton_p);

    conv_newton.set_convergence_constants(newton_def_tol, newton_def_max_it);

    vec_ops.assign(x_back, x0);

    ob_prob_t *ob_prob_p = &ob_prob;

    bool converged = newton.solve(ob_prob_p, x0, sigma);
    if(!converged)
    {
        printf("Newton 2 failed to converge!\n");
    }
    std::cout << "Newton 2 solution norm = " << vec_ops.norm(x0) << std::endl;
    std::stringstream f_name, f_name_domain;
    f_name << "test_newton_solution_" << params.N << "_" << params.L << "_var_prec.dat";
    f_name_domain << "test_newton_solution_domain_" << params.N << "_" << params.L << "_var_prec.dat";
    // ob_prob.physical_solution(x0, x1);
    // vec_file_ops.write_vector(f_name.str(), x1);

    ob_prob.write_solution_basis(f_name.str(), x0);
    ob_prob.write_solution_domain(f_name_domain.str(), x0); //add another method with the domain range

    //testing stability
//  eigensolver types
/*    using lapack_wrap_t = lapack_wrap<T>;
    using s_inv_system_op_t = stability::system_operator_shift_inverse<vec_ops_t, ob_prob_t, lin_op_shifted_t, lin_solver_shifted_t, log_t>;
    using arnoldi_t = numerical_algos::eigen_solvers::arnoldi_process<vec_ops_t, mat_ops_t, s_inv_system_op_t, log_t>;
    using iram_t = stability::IRAM::iram_process<vec_ops_t, mat_ops_t, lapack_wrap_t, arnoldi_t, s_inv_system_op_t, lin_op_t, log_t>;
    using stability_analysis_t = stability::stability_analysis<vec_ops_t, ob_prob_t, log_t, newton_t, iram_t>;

//  eigensolver config
    s_inv_system_op_t s_inv_system_op(&vec_ops, &ob_prob, &lin_op_shifted, &lin_solver_shifted, log_p);
    s_inv_system_op.set_tolerance(1.0e-7);
    
    T sigma_s = 10.0;
    s_inv_system_op.set_sigma(sigma_s);
    size_t m = N-1;
    vec_ops_t vec_ops_small(m, &cublas);
    mat_ops_t mat_ops_small(m, m, &cublas);
    mat_ops_t mat_ops_Krlob(N, m, &cublas);
    lapack_wrap_t lapack(m);

    arnoldi_t arnoldi(&vec_ops, &vec_ops_small, &mat_ops_Krlob, &mat_ops_small, &s_inv_system_op, log_p);
    iram_t iram(&vec_ops, &mat_ops_Krlob, &vec_ops_small, &mat_ops_small, &lapack, &arnoldi, &s_inv_system_op, &lin_op, log_p);
    iram.set_verbocity(true);
    iram.set_target_eigs("LR");
    iram.set_number_of_desired_eigenvalues(m-1);
    iram.set_tolerance(1.0e-6);
    iram.set_max_iterations(100);
    iram.set_verbocity(true);

    stability_analysis_t stabl(&vec_ops, log_p, &ob_prob, &newton, &iram);    
    stabl.set_linear_operator_stable_eigenvalues_halfplane(1);
    stabl.execute(x0, sigma);

    */
    vec_ops.stop_use_vectors(b, x0, x1, dx, x_back); vec_ops.free_vectors(b, x0, x1, dx, x_back);

    return 0;
}