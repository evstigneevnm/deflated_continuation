#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <array>
#include <cmath>
#include <memory>
// #include <utils/init_cuda.h>

// #include <scfd/utils/log_std.h>
#include <scfd/utils/log.h>
// #include <numerical_algos/lin_solvers/default_monitor.h>
// #include <numerical_algos/lin_solvers/bicgstabl.h>

// #include <common/macros.h>
// #include <common/gpu_file_operations.h>
// #include <common/gpu_vector_operations.h>
#include <common/file_operations.h>
#include <common/scfd_serial_cpu_vector_operations.h>
#include <common/cpu_file_operations.h>
#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <periodic_orbit/periodic_orbit_nonlinear_operator.h>
#include <periodic_orbit/system_operator_single_section.h>
#include <periodic_orbit/convergence_strategy_single_section.h>
#include <numerical_algos/newton_solvers/newton_solver.h>
#include "rossler_operator.h"
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
//continuation
// #include <containers/curve_container.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>

#include <continuation/predictor_adaptive.h>
#include <continuation/system_operator_continuation.h>
#include <continuation/advance_solution.h>
#include <continuation/initial_tangent.h>
#include <continuation/convergence_strategy.h>

#include <deflation/solution_storage.h>
#include <containers/knots.hpp>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>
#include <continuation/continuation.hpp>



int main(int argc, char const *argv[])
{
    using real = SCALAR_TYPE;
    using log_t = scfd::utils::log_std;

    using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
    using vec_file_ops_t = cpu_file_operations<vec_ops_t>;
    using vec_t = typename vec_ops_t::vector_type;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t,log_t>;

    using nlin_op_t = nonlinear_operators::rossler<vec_ops_t>;
    
    // using periodic_orbit_nonlinear_operator_t = periodic_orbit::periodic_orbit_nonlinear_operator<vec_ops_t, nlin_op_t, log_t, time_steppers::time_step_adaptation_error_control, time_steppers::explicit_time_step>;

    using periodic_orbit_nonlinear_operator_t = periodic_orbit::periodic_orbit_nonlinear_operator<vec_ops_t, nlin_op_t, log_t, time_steppers::time_step_adaptation_constant, time_steppers::explicit_time_step>;


    using periodic_orbit_linear_operator_t = typename periodic_orbit_nonlinear_operator_t::linear_operator_type;
    using periodic_orbit_preconditioner_t = typename periodic_orbit_nonlinear_operator_t::preconditioner_type;

    using lin_solve_t = numerical_algos::lin_solvers::bicgstabl<periodic_orbit_linear_operator_t, periodic_orbit_preconditioner_t, vec_ops_t, monitor_t, log_t>;

    using system_operator_single_section_t = nonlinear_operators::newton_method::system_operator_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, periodic_orbit_linear_operator_t, lin_solve_t>;    
    using convergence_strategy_single_section_t = nonlinear_operators::newton_method::convergence_strategy_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, log_t>;
    using newton_solver_t = numerical_algos::newton_method::newton_solver<vec_ops_t, periodic_orbit_nonlinear_operator_t, system_operator_single_section_t, convergence_strategy_single_section_t>;    

//   for continuation
    using sherman_morrison_linear_system_solve_t = numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        periodic_orbit_linear_operator_t,
        periodic_orbit_preconditioner_t,
        vec_ops_t,
        monitor_t,
        log_t,
        numerical_algos::lin_solvers::bicgstabl>;

    using knots_t = container::knots<real>;

    using container_helper_t = container::curve_helper_container<vec_ops_t>;

    using sol_storage_def_t = deflation::solution_storage<vec_ops_t>;

    using bif_diag_curve_t = container::bifurcation_diagram_curve<
        vec_ops_t,
        vec_file_ops_t, 
        log_t,
        periodic_orbit_nonlinear_operator_t,
        newton_solver_t, 
        sol_storage_def_t,
        container_helper_t
        >;
    using bif_diag_t = container::bifurcation_diagram<
        vec_ops_t,
        vec_file_ops_t, 
        log_t,
        periodic_orbit_nonlinear_operator_t,
        newton_solver_t, 
        sol_storage_def_t,
        bif_diag_curve_t,
        container_helper_t
        > ;

    using continuate_t = continuation::continuation<
        vec_ops_t, 
        vec_file_ops_t, 
        log_t, 
        periodic_orbit_nonlinear_operator_t, 
        periodic_orbit_linear_operator_t,  
        knots_t,
        sherman_morrison_linear_system_solve_t,  
        newton_solver_t,
        bif_diag_curve_t
        >;





	std::string scheme_name("RKDP45");
    real a_param = 0.1, b_param = 0.1, c_param = 4.0, max_time_simlation = 500.0, max_param_val = 0.4; 
    int which_param = 2;

    std::cout << "Usage: " << argv[0] << " scheme_name" << std::endl;
    std::cout << "scheme_name: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
    std::cout << "or" << std::endl;
    std::cout << "Usage: " << argv[0] << " scheme_name a_initial_val b_initial_val c_initial_val max_time_pre_simlation which_parameter_to_continuate[0,1,2] param_final_val" << std::endl;

    if(argc == 8)
    {
        scheme_name = std::string(argv[1]);
        a_param = std::stof(argv[2]);
        b_param = std::stof(argv[3]);
        c_param = std::stof(argv[4]);
        max_time_simlation = std::stof(argv[5]);
        which_param = std::stoi(argv[6]);
        max_param_val = std::stof(argv[7]);
    }
    
    log_t log_newton;
    log_t log_time;
    log_t log_linsolver;
    log_time.set_verbosity(0);
    log_linsolver.set_verbosity(0);

    log_newton.info("test periodic orbit continuation for rossler operator.");    
    vec_ops_t vec_ops(3);
    vec_file_ops_t file_ops(&vec_ops);

    vec_t x0, x1, b, x;
    vec_ops.init_vectors(x0, x1, b, x); vec_ops.start_use_vectors(x0, x1, b, x);
    nlin_op_t rossler(&vec_ops, which_param, a_param, b_param, c_param); 
    rossler.set_period_point(x0);    
    // rossler.set_initial(x0);
    auto param = rossler.get_selected_parameter_value();
    periodic_orbit_nonlinear_operator_t periodic_orbit_nonlin_op(&vec_ops, &rossler, &log_time, max_time_simlation, param, scheme_name);        

    auto periodic_orbit_lin_op = periodic_orbit_nonlin_op.linear_operator;
    auto periodic_orbit_precond = periodic_orbit_nonlin_op.preconditioner;
    //linear solver setup
    real rel_tol = 1.0e-8;
    size_t max_iters = 100;
    //newton setup
    real newton_tolerance = 1.0e-10;
    unsigned int newton_max_it = 100;
    real newton_relax_tolerance_factor_ = 100;
    unsigned int newton_relax_tolerance_steps = 50;    

    // lin_solve_t solver(&vec_ops, &log, 0);    
    // solver.set_basis_size(1);
    // auto& mon = solver.monitor();
    // mon.init(rel_tol, real(0), max_iters);
    // mon.set_save_convergence_history(true);
    // mon.set_divide_out_norms_by_rel_base(true);    


    //extended linear solver setup
    sherman_morrison_linear_system_solve_t sm_solver(periodic_orbit_precond, &vec_ops, &log_linsolver);
    auto solver_orig = sm_solver.get_linsolver_handle_original();
    auto mon_orig = &sm_solver.get_linsolver_handle_original()->monitor();
    mon_orig->init(rel_tol, real(0), max_iters);
    mon_orig->set_save_convergence_history(true);
    mon_orig->set_divide_out_norms_by_rel_base(true);
    mon_orig->out_min_resid_norm();    
    auto mon_sm = &sm_solver.get_linsolver_handle()->monitor();
    mon_sm->init(rel_tol, real(0), max_iters);
    mon_sm->set_save_convergence_history(true);
    mon_sm->set_divide_out_norms_by_rel_base(true);
    mon_sm->out_min_resid_norm();


    system_operator_single_section_t sys_op(&vec_ops, periodic_orbit_lin_op, solver_orig);
    convergence_strategy_single_section_t convergence(&vec_ops, &log_newton);
    convergence.set_convergence_constants(newton_tolerance, newton_max_it);
    newton_solver_t newton(&vec_ops, &sys_op, &convergence);

    periodic_orbit_nonlin_op.set_hyperplane_from_initial_guesses(x0, param);
    periodic_orbit_nonlin_op.time_stepper(x0, param, {0, max_time_simlation});
    periodic_orbit_nonlin_op.save_norms("rossler_initial.dat");

    newton.solve(&periodic_orbit_nonlin_op, x0, param);
    std::stringstream ss_periodic_estimate;
    ss_periodic_estimate << "rossler_period_" << scheme_name << ".dat";
    periodic_orbit_nonlin_op.save_period_estmate_norms(ss_periodic_estimate.str() );
    real T = periodic_orbit_nonlin_op.get_period_estmate_time();
    log_newton.info_f("solution point: [%le,%le,%le], stimated period: %le", x0(0), x0(1), x0(2), T);


    // continuate
    knots_t knots;
    // knots.add_element({0.1*param, param, 4*param});
    knots.add_element({0.1*param, param, 4*param});
    unsigned int max_S = 100;
    real ds_0 = 0.005;
    real ds_max = 0.05;
    real sign = -1;
    real newton_wight = 1.1;
    bool store_norms_history = true;
    
    vec_t z, x_0s, x_1s, f, f1,  f_lambda, d_x, x1_p;
    vec_ops.init_vectors(z, x_0s, x_1s,f_lambda, f, d_x, x1_p, f1); vec_ops.start_use_vectors(z, x_0s, x_1s,f_lambda, f, d_x, x1_p, f1);

    // //manual continuation
    // //remove these comments from *start* to *end* to perform test manual continuation of periodic orbits
    // //*start*
    // real d_lambda = 0.0;
    // real lambda_0 = param, lambda_1 = param, lambda_0s = 0, lambda_1s = 0, lambda_1_p = 0;
    // //initial tangent
    // periodic_orbit_nonlin_op.set_linearization_point(x0, lambda_0);
    // periodic_orbit_nonlin_op.F_and_jacobian_alpha(x0, lambda_0, f, f_lambda);
    // periodic_orbit_nonlin_op.F(x0, lambda_0, f1);
    // vec_ops.add_mul(-1.0, f, f1);
    // log_newton.info_f("x0 = (%le,%le,%le), lambda_0 = %le, f = (%le,%le,%le), f_lambda = (%le,%le,%le), ||df|| = %le", x0[0], x0[1], x0[2], lambda_0, f[0], f[1], f[2], f_lambda[0], f_lambda[1], f_lambda[2], vec_ops.norm(f1) );    
    // vec_ops.add_mul_scalar(0.0, -1.0, f_lambda);
    // solver_orig->solve(*periodic_orbit_lin_op, f_lambda, x_0s);
    // lambda_0s = sign/std::sqrt( vec_ops.scalar_prod(x_0s,x_0s)+1.0 );
    // vec_ops.add_mul_scalar(0.0, lambda_0s, x_0s);

    // auto orth_proj = [&vec_ops, &b](const vec_t& x0, const real lambda_0, const vec_t& x_0s, const real lambda_0s, const vec_t& x1, const real lambda_1, const real ds)
    // {
    //     vec_ops.assign_mul(1.0, x1, -1.0, x0, b);
    //     auto d_x_x_s = vec_ops.scalar_prod(b, x_0s);
    //     auto d_lambda_lambda_s = (lambda_1 - lambda_0)*lambda_0s;
    //     return d_x_x_s + d_lambda_lambda_s - ds;
    // };

    // std::vector< std::tuple<real, real, real> > bif_data;
    // bif_data.emplace_back(lambda_0, vec_ops.norm(x0), T);

    // for(int s = 0; s<150; s++)
    // {        
    //     log_newton.info_f("lambda_0s = %le, x_0s = (%le,%le,%le)", lambda_0s, x_0s[0], x_0s[1], x_0s[2] );
    //     vec_ops.assign_mul(1.0, x0, ds_0, x_0s, x1);
    //     lambda_1 = lambda_0 + ds_0*lambda_0s;
    //     periodic_orbit_nonlin_op.F(x1, lambda_1, b);
    //     log_newton.info_f("lambda_0 = %le, x0 = (%le,%le,%le), lambda_1 = %le, x1 = (%le,%le,%le), diff = (%le, %le, %le), ||f|| = %le", lambda_0, x0[0], x0[1], x0[2], lambda_1, x1[0], x1[1], x1[2], x0[0] - x1[0], x0[1] - x1[1], x0[2] - x1[2], vec_ops.norm(b) );

    //     periodic_orbit_nonlin_op.set_linearization_point(x1, lambda_1);
    //     periodic_orbit_nonlin_op.F_and_jacobian_alpha(x1, lambda_1, f, f_lambda);
    //     periodic_orbit_nonlin_op.F(x1, lambda_1, f1);

    //     real newton_norm = vec_ops.norm(f);
    //     real newton_norm_prev = newton_norm;
    //     std::size_t iters = 0;
    //     vec_ops.assign(x1, x1_p);
    //     while(newton_norm > newton_tolerance)
    //     {
    //         iters++;
    //         periodic_orbit_nonlin_op.set_linearization_point(x1, lambda_1);
    //         periodic_orbit_nonlin_op.F_and_jacobian_alpha(x1, lambda_1, f, f_lambda);
    //         periodic_orbit_nonlin_op.F(x1, lambda_1, f1);
    //         vec_ops.add_mul(-1.0, f, f1);
    //         real beta = -orth_proj(x0, lambda_0, x_0s, lambda_0s, x1, lambda_1, ds_0);
    //         sm_solver.solve(*periodic_orbit_lin_op, f_lambda, x_0s, lambda_0s, f, beta, d_x, d_lambda);
    //         real w = 1.0;
    //         real lambda_1_priv = 0, newton_norm_up = 1.0;
    //         // do
    //         // {
    //         //     vec_ops.add_mul(w, d_x, 1.0, x1, 0.0, x1_p);
    //         //     lambda_1_priv = lambda_1;
    //         //     lambda_1_p = lambda_1 + w*d_lambda;
    //         //     periodic_orbit_nonlin_op.F(x1_p, lambda_1_p, f);
    //         //     beta = -orth_proj(x0, lambda_0, x_0s, lambda_0s, x1_p, lambda_1_p, ds_0);
    //         //     newton_norm_up = vec_ops.norm_rank1(f, beta);
    //         //     log_newton.info_f("norm: %le, w: %le", newton_norm_up, w);
    //         //     w *= 0.5;
    //         // } while (newton_norm_up > newton_norm);
    //         vec_ops.add_mul(w, d_x, 1.0, x1, 0.0, x1_p);
    //         lambda_1_priv = lambda_1;
    //         lambda_1_p = lambda_1 + w*d_lambda;            
    //         periodic_orbit_nonlin_op.F(x1_p, lambda_1_p, f);
    //         beta = -orth_proj(x0, lambda_0, x_0s, lambda_0s, x1_p, lambda_1_p, ds_0);
    //         vec_ops.assign(x1_p, x1);
    //         lambda_1 = lambda_1_p;


    //         // vec_ops.add_mul_scalar(0.0, -1.0, f); //???
    //         newton_norm_prev  = newton_norm;
    //         newton_norm = vec_ops.norm_rank1(f, beta);
    //         log_newton.info_f("%lu newton: %le -> %le, %le -> %le, ||df|| = %le", iters, newton_norm_prev, newton_norm, lambda_1_priv, lambda_1, vec_ops.norm(f1));
            
    //     }
    //     periodic_orbit_nonlin_op.F(x1, lambda_1, b);
    //     log_newton.info_f("lambda_0 = %le, x0 = (%le,%le,%le), lambda_1 = %le, x1 = (%le,%le,%le), diff = (%le, %le, %le), ||f|| = %le", lambda_0, x0[0], x0[1], x0[2], lambda_1, x1[0], x1[1], x1[2], x0[0] - x1[0], x0[1] - x1[1], x0[2] - x1[2], vec_ops.norm(b) );
    //     periodic_orbit_nonlin_op.save_period_estmate_norms("test_rossler_periodic_T_1.dat");

    //     T = periodic_orbit_nonlin_op.get_period_estmate_time();
    //     log_newton.info_f("estimated period: %le", T);
    //     bif_data.emplace_back(lambda_1, vec_ops.norm(x1), T);
        
    //     periodic_orbit_nonlin_op.set_linearization_point(x1, lambda_1);
    //     periodic_orbit_nonlin_op.F_and_jacobian_alpha(x1, lambda_1, f, f_lambda);
    //     periodic_orbit_nonlin_op.F(x1, lambda_1, f1);

    //     vec_ops.assign_scalar(0, f);
    //     sm_solver.solve(*periodic_orbit_lin_op, f_lambda, x_0s, lambda_0s, f, 1.0, x_1s, lambda_1s);
        
    //     log_newton.info_f("tangent update: x_0s: (%le, %le, %le), lambda_0s: %le -> x_1s: (%le, %le, %le), lambda_1s: %le", x_0s[0], x_0s[1], x_0s[2], lambda_0s, x_1s[0], x_1s[1], x_1s[2], lambda_1s); 

    //     vec_ops.assign(x_0s, x_1s);
    //     lambda_0s = lambda_1s;
    //     vec_ops.assign(x1, x0);
    //     lambda_0 = lambda_1;
    // }

    // for(auto& b: bif_data)
    // {
    //     real l, n, T;
    //     std::tie(l, n, T) = b;
    //     std::cout << l << ", " << n << ", " << T << std::endl;
    // }
    // //*end*    


    continuate_t continuate(&vec_ops, &file_ops, &log_newton, &periodic_orbit_nonlin_op, periodic_orbit_lin_op, &knots, &sm_solver, &newton);
    continuate.set_steps(max_S, ds_0, ds_max, sign);
    continuate.set_newton( newton_tolerance, newton_max_it, newton_relax_tolerance_factor_, newton_relax_tolerance_steps, newton_wight, store_norms_history);
    
    std::string project_dir = "./test_rossler_continuate_cpu/";
    unsigned int skip_files = 1;
    bif_diag_t bif_diag(&vec_ops, &file_ops, &log_newton, &periodic_orbit_nonlin_op, &newton, project_dir, skip_files);
    // bif_diag_curve_t* bdf;
    log_newton.info("init_new_curve()");
    bif_diag.init_new_curve();
    log_newton.info("get_current_ref( bdf ):");
    // bif_diag.get_current_ref( bdf );
    auto bdf = bif_diag.get_current_ref();
    log_newton.info("continuate_curve( bdf , x0, param )");
    continuate.continuate_curve( bdf , x0, param );
    log_newton.info("close_curve()");
    bif_diag.close_curve();

    vec_ops.stop_use_vectors(z, x_0s, x_1s, f_lambda, d_x, x1_p, f, f1); vec_ops.free_vectors(z, x_0s, x_1s, f_lambda, d_x, x1_p, f, f1);
    vec_ops.stop_use_vectors(x0, x1, b, x); vec_ops.free_vectors(x0, x1, b, x);
    
    

	return 0;
}
