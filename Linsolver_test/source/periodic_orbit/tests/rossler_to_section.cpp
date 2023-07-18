#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <array>
#include <cmath>
// #include <utils/init_cuda.h>

#include <utils/log.h>
// #include <numerical_algos/lin_solvers/default_monitor.h>
// #include <numerical_algos/lin_solvers/bicgstabl.h>

// #include <common/macros.h>
// #include <common/gpu_file_operations.h>
// #include <common/gpu_vector_operations.h>
#include <common/file_operations.h>
#include <common/cpu_vector_operations.h>

#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <periodic_orbit/time_stepper_to_section.h>
#include <periodic_orbit/hyperplane.h>
#include "rossler_operator.h"


int main(int argc, char const *argv[])
{

    using real = SCALAR_TYPE;
    using log_t = utils::log_std;

    using vec_ops_t = cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;

    using nlin_op_t = nonlinear_operators::rossler<vec_ops_t>;
    using time_step_err_ctrl_t = time_steppers::time_step_adaptation_error_control<vec_ops_t, log_t>;
    using single_step_err_ctrl_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_err_ctrl_t>;
    using hyperplane_t = periodic_orbit::hyperplane<vec_ops_t,nlin_op_t>;    
    using time_stepper_err_ctrl_t = periodic_orbit::time_steppers::time_stepper_to_section<vec_ops_t, nlin_op_t, single_step_err_ctrl_t, hyperplane_t, log_t>;


    if(argc != 8)
    {
        std::cout << argv[0] << " a b c select time time_before_section_check name\n  a,b,c - parameters,\n select - main parameter number (0, 1 or 2), \n time - total simulation time,\n";
        std::cout << "   time_before_section_check(<time) - simulation time before starting hyperplane intersection check,\n";
        std::cout << "   name - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return(0);       
    }    
    real a = std::stof(argv[1]);
    real b = std::stof(argv[2]);
    real c = std::stof(argv[3]);
    unsigned int select = std::stoi(argv[4]);
    real simulation_time = std::stof(argv[5]);
    real time_b4_section_check = std::stof(argv[6]);
    std::string scheme_name(argv[7]);

    log_t log;

    vec_ops_t vec_ops(3);
    
    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);


    nlin_op_t rossler(&vec_ops, select, a, b, c); //use second parameter as a bifurcation parameter.
    rossler.set_initial(x0);

    time_step_err_ctrl_t time_step_err_ctrl(&vec_ops, &log);

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
    auto mu = rossler.get_selected_parameter_value();

    single_step_err_ctrl_t explicit_step_err_control(&vec_ops, &time_step_err_ctrl, &log,  &rossler, mu, method);
    
    hyperplane_t hyperplane(&vec_ops, &rossler, x0, rossler.get_selected_parameter_value() );

    time_stepper_err_ctrl_t time_stepper_err_ctrl(&vec_ops, &rossler, &explicit_step_err_control, &log);

    log.info_f("executing time stepper with time = %.2le", simulation_time);
    // time_stepper_err_ctrl.set_parameter(mu);
    // time_stepper_err_ctrl.set_initial_conditions(x0, 0.0);
    // time_stepper_err_ctrl.execute();

    time_stepper_err_ctrl.execute(&hyperplane, x0, mu, {0, simulation_time}, time_b4_section_check );
    log.info_f("simulation time = %le, estimated periodic time = %le", time_stepper_err_ctrl.get_simulated_time(), time_stepper_err_ctrl.get_period_estmate_time());
    std::stringstream ss, ss_periodic_estimate;
    ss << "rossler_result_" << scheme_name << ".dat";
    time_stepper_err_ctrl.save_norms( ss.str() );

    ss_periodic_estimate << "rossler_period_estimate_result_" << scheme_name << ".dat";
    time_stepper_err_ctrl.save_period_estmate_norms(ss_periodic_estimate.str() );
    // std::stringstream ss;
    // ss << "x_" << simulation_time << "_sim.pos";


    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

