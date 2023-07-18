#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <array>
#include <cmath>
#include <utils/init_cuda.h>
#include <external_libraries/cublas_wrap.h>
#include <utils/log.h>
// #include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <common/file_operations.h>
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



int main(int argc, char const *argv[])
{

    using real = SCALAR_TYPE;
    using log_t = utils::log_std;

    
    using vec_ops_t = gpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t,log_t>;

    using nlin_op_t = nonlinear_operators::rossler<vec_ops_t>;
    
    using periodic_orbit_nonlinear_operator_t = periodic_orbit::periodic_orbit_nonlinear_operator<vec_ops_t, nlin_op_t, log_t, time_steppers::time_step_adaptation_error_control, time_steppers::explicit_time_step>;

    using periodic_orbit_linear_operator_t = typename periodic_orbit_nonlinear_operator_t::linear_operator_type;
    using periodic_orbit_preconditioner_t = typename periodic_orbit_nonlinear_operator_t::preconditioner_type;

    using lin_solve_t = numerical_algos::lin_solvers::bicgstabl<periodic_orbit_linear_operator_t, periodic_orbit_preconditioner_t, vec_ops_t, monitor_t, log_t>;

    using system_operator_single_section_t = nonlinear_operators::newton_method::system_operator_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, periodic_orbit_linear_operator_t, lin_solve_t>;    
    using convergence_strategy_single_section_t = nonlinear_operators::newton_method::convergence_strategy_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, log_t>;
    using newton_solver_t = numerical_algos::newton_method::newton_solver<vec_ops_t, periodic_orbit_nonlinear_operator_t, system_operator_single_section_t, convergence_strategy_single_section_t>;

    std::string scheme_name("RKDP45");
    real a_param = 0.1, b_param = 0.1, c_param = 6.1, max_time_simlation = 250.0;   
    if((argc >1 )&&(argc != 6)&&(argc != 2))
    {
        std::cout << "Usage: " << argv[0] << " scheme_name" << std::endl;
        std::cout << "scheme_name: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        std::cout << "or" << std::endl;
        std::cout << "Usage: " << argv[0] << " scheme_name a b c max_time_simlation" << std::endl;
        return 1;        
    }
    else if(argc == 2)
    {
        scheme_name = std::string(argv[1]);
        if((scheme_name == "-h")||(scheme_name == "--help"))
        {
            std::cout << "Usage: " << argv[0] << " scheme_name" << std::endl;
            std::cout << "scheme_name: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
            std::cout << "or" << std::endl;
            std::cout << "Usage: " << argv[0] << " scheme_name a b c max_time_simlation" << std::endl;
            return 1;
        }
    }
    else if(argc == 6)
    {
        scheme_name = std::string(argv[1]);
        a_param = std::stof(argv[2]);
        b_param = std::stof(argv[3]);
        c_param = std::stof(argv[4]);
        max_time_simlation = std::stof(argv[5]);
    }
    auto method = time_steppers::detail::methods::EXPLICIT_EULER;
    if(scheme_name == "EE")
    {
        method = time_steppers::detail::methods::EXPLICIT_EULER;
    }
    else if(scheme_name == "RKDP45")
    {
        method = time_steppers::detail::methods::RKDP45;
    }
    else if(scheme_name == "RK33SSP")
    {
        method = time_steppers::detail::methods::RK33SSP;
    }    
    else if(scheme_name == "RK43SSP")
    {
        method = time_steppers::detail::methods::RK43SSP;
    } 
    else if(scheme_name == "RK64SSP")
    {
        method = time_steppers::detail::methods::RK64SSP;
    }     
    else if(scheme_name == "HE")
    {
        method = time_steppers::detail::methods::HEUN_EULER;
    }  
    else
    {
        throw std::logic_error("incorrect method string type provided.");
    }
    real ref_error = std::numeric_limits<real>::epsilon();


    if(!utils::init_cuda(-1))
    {
        return 2;
    }

    cublas_wrap cublas;


    log_t log;
    log.info("test periodic orbit stabilization for rossler operator.");

    vec_ops_t vec_ops(3, &cublas);
    
    vec_t x0, x1, b, x;

    vec_ops.init_vectors(x0, x1, b, x); vec_ops.start_use_vectors(x0, x1, b, x);

    size_t parameter_select = 2;
    nlin_op_t rossler(&vec_ops, parameter_select, a_param, b_param, c_param); //use second parameter as a bifurcation parameter.
    rossler.set_period_point(x0);

    auto mu = rossler.get_selected_parameter_value();

    periodic_orbit_nonlinear_operator_t periodic_orbit_nonlin_op(&vec_ops, &rossler, &log, 100.0, mu, method);

    auto periodic_orbit_lin_op = periodic_orbit_nonlin_op.linear_operator;


    lin_solve_t solver(&vec_ops, &log, 0);
    solver.set_basis_size(1);
    real rel_tol = 1.0e-2;
    size_t max_iters = 100;
    auto& mon = solver.monitor();
    mon.init(rel_tol, real(0), max_iters);
    mon.set_save_convergence_history(true);
    mon.set_divide_out_norms_by_rel_base(true);

    system_operator_single_section_t sys_op(&vec_ops, periodic_orbit_lin_op, &solver);
    convergence_strategy_single_section_t convergence(&vec_ops, &log);
    newton_solver_t newton(&vec_ops, &sys_op, &convergence);

    periodic_orbit_nonlin_op.set_hyperplane_from_initial_guesses(x0, mu);
    periodic_orbit_nonlin_op.time_stepper(x0, mu, {0, max_time_simlation});
    periodic_orbit_nonlin_op.save_norms("rossler_initial.dat");

    newton.solve(&periodic_orbit_nonlin_op, x0, mu);

    for(int j = 0; j<3; j++)
    {
        std::cout << x0[j] << std::endl;
    }

    std::stringstream ss_periodic_estimate;
    ss_periodic_estimate << "rossler_period_" << scheme_name << ".dat";
    periodic_orbit_nonlin_op.save_period_estmate_norms(ss_periodic_estimate.str() );


    vec_ops.stop_use_vectors(x0, x1, b, x); vec_ops.free_vectors(x0, x1, b, x);
    return 0;
}

