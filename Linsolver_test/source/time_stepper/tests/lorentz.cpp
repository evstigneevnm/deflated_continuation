#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <array>
#include <cmath>

#include <scfd/utils/log_std.h>

#include <common/file_operations.h>
#include <common/scfd_serial_cpu_vector_operations.h>

#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/time_stepper.h>
#include "external_lorentz_stop.h"

#include "../../periodic_orbit/tests/lorentz_operator.h"


int main( int argc, char const *argv[] )
{

    using real  = SCALAR_TYPE;
    using log_t = scfd::utils::log_std;

    using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
    using vec_t     = typename vec_ops_t::vector_type;

    using nlin_op_t = nonlinear_operators::lorentz<vec_ops_t>;

    using time_step_err_ctrl_t   = time_steppers::time_step_adaptation_error_control<vec_ops_t, log_t>;
    using single_step_err_ctrl_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_err_ctrl_t>;
    using external_manager_h     = time_steppers::detail::external_lorentz_stop<vec_ops_t, single_step_err_ctrl_t>;
    using time_stepper_err_ctrl_t =
        time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_err_ctrl_t, log_t, external_manager_h>;

    real         sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0, epsilon = 0.0055, delta = 0;
    std::string  scheme_name( "RKDP45" );
    unsigned int select          = 3;
    real         simulation_time = 100;
    if ( ( argc != 9 ) && ( argc != 3 ) )
    {
        std::cout << argv[0]
                  << " sigma rho beta epsilon delta select time name\n  sigma rho beta epsilon delta - parameters,\n "
                     "select - main parameter "
                     "time - simulation time,\n";
        std::cout << "OR" << std::endl;
        std::cout << argv[0] << " epsilon time" << std::endl;
        std::cout << "   name - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return ( 0 );
    }
    else if ( argc == 3 )
    {
        epsilon         = std::stof( argv[1] );
        simulation_time = std::stof( argv[2] );
    }
    else if ( argc == 9 )
    {
        sigma           = std::stof( argv[1] );
        rho             = std::stof( argv[2] );
        beta            = std::stof( argv[3] );
        epsilon         = std::stof( argv[4] );
        delta           = std::stof( argv[5] );
        select          = std::stoi( argv[6] );
        simulation_time = std::stof( argv[7] );
        scheme_name     = argv[8];
    }
    log_t log;

    vec_ops_t vec_ops( 3 );

    vec_t x0;

    vec_ops.init_vector( x0 );
    vec_ops.start_use_vector( x0 );


    nlin_op_t lorentz(
        &vec_ops, select, sigma, rho, beta, epsilon, delta
    ); // use second parameter as a bifurcation parameter.

    lorentz.set_initial( x0 );

    time_step_err_ctrl_t time_step_err_ctrl( &vec_ops, &log );

    auto mu = lorentz.get_selected_parameter_value();

    single_step_err_ctrl_t explicit_step_err_control( &vec_ops, &time_step_err_ctrl, &log, &lorentz, mu, scheme_name );

    external_manager_h      external_manager( &vec_ops );
    time_stepper_err_ctrl_t time_stepper_err_ctrl(
        &vec_ops, &lorentz, &explicit_step_err_control, &log, &external_manager
    );

    log.info_f( "executing time stepper with time = %.2le", simulation_time );
    // time_stepper_err_ctrl.set_parameter(mu);
    // time_stepper_err_ctrl.set_initial_conditions(x0, 0.0);
    // time_stepper_err_ctrl.execute();

    time_stepper_err_ctrl.execute( x0, mu, { 0, simulation_time } );
    std::stringstream ss;
    ss << "lorentz_result_" << scheme_name << ".dat";
    time_stepper_err_ctrl.save_norms( ss.str() );

    // std::stringstream ss;
    // ss << "x_" << simulation_time << "_sim.pos";
    auto ttt = time_stepper_err_ctrl.get_simulated_time();
    std::cout << "simulated_time = " << ttt << std::endl;
    vec_ops.stop_use_vector( x0 );
    vec_ops.free_vector( x0 );
    return 0;
}
