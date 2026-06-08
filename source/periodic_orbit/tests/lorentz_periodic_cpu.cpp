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
#include <periodic_orbit/periodic_orbit_nonlinear_operator.h>
#include <periodic_orbit/system_operator_single_section.h>
#include <periodic_orbit/convergence_strategy_single_section.h>
#include <numerical_algos/newton_solvers/newton_solver.h>
#include "lorentz_operator.h"
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>


int main( int argc, char const *argv[] )
{

    using real  = SCALAR_TYPE;
    using log_t = scfd::utils::log_std;

    using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
    using vec_t     = typename vec_ops_t::vector_type;

    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t, log_t>;

    using nlin_op_t = nonlinear_operators::lorentz<vec_ops_t>;

    using periodic_orbit_nonlinear_operator_t = periodic_orbit::periodic_orbit_nonlinear_operator<
        vec_ops_t,
        nlin_op_t,
        log_t,
        time_steppers::time_step_adaptation_error_control,
        time_steppers::explicit_time_step>;

    using periodic_orbit_linear_operator_t = typename periodic_orbit_nonlinear_operator_t::linear_operator_type;
    using periodic_orbit_preconditioner_t  = typename periodic_orbit_nonlinear_operator_t::preconditioner_type;

    using lin_solve_t = numerical_algos::lin_solvers::
        bicgstabl<periodic_orbit_linear_operator_t, periodic_orbit_preconditioner_t, vec_ops_t, monitor_t, log_t>;

    using system_operator_single_section_t = nonlinear_operators::newton_method::system_operator_single_section<
        vec_ops_t,
        periodic_orbit_nonlinear_operator_t,
        periodic_orbit_linear_operator_t,
        lin_solve_t>;
    using convergence_strategy_single_section_t = nonlinear_operators::newton_method::
        convergence_strategy_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, log_t>;
    using newton_solver_t = numerical_algos::newton_method::newton_solver<
        vec_ops_t,
        periodic_orbit_nonlinear_operator_t,
        system_operator_single_section_t,
        convergence_strategy_single_section_t>;

    std::string scheme_name( "RKDP45" );
    real        sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0, epsilon = 0.0055, delta = 0;
    real        max_time_simulation = 30.0;
    if ( ( argc > 1 ) && ( argc != 8 ) && ( argc != 2 ) && ( argc != 3 ) )
    {
        std::cout << "Usage: " << argv[0] << " scheme_name" << std::endl;
        std::cout << "scheme_name: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        std::cout << "or" << std::endl;
        std::cout << "Usage: " << argv[0] << " scheme_name max_time_simulation" << std::endl;
        std::cout << "or" << std::endl;
        std::cout << "Usage: " << argv[0] << " scheme_name a b c max_time_simulation" << std::endl;
        return 1;
    }
    else if ( argc == 2 )
    {
        scheme_name = std::string( argv[1] );
        if ( ( scheme_name == "-h" ) || ( scheme_name == "--help" ) )
        {
            std::cout << "Usage: " << argv[0] << " scheme_name" << std::endl;
            std::cout << "scheme_name: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
            std::cout << "or" << std::endl;
            std::cout << "Usage: " << argv[0] << " scheme_name a b c max_time_simulation" << std::endl;
            return 1;
        }
    }
    else if ( argc == 3 )
    {
        scheme_name         = std::string( argv[1] );
        max_time_simulation = std::stof( argv[2] );
    }
    else if ( argc == 8 )
    {
        scheme_name         = std::string( argv[1] );
        sigma               = std::stof( argv[2] );
        rho                 = std::stof( argv[3] );
        beta                = std::stof( argv[4] );
        epsilon             = std::stof( argv[5] );
        delta               = std::stof( argv[6] );
        max_time_simulation = std::stof( argv[7] );
    }

    log_t log;
    log.info( "test periodic orbit stabilization for lorentz operator." );

    vec_ops_t vec_ops( 3 );

    vec_t x0, x1, b, x;

    vec_ops.init_vectors( x0, x1, b, x );
    vec_ops.start_use_vectors( x0, x1, b, x );

    size_t    parameter_select = 0;
    nlin_op_t lorentz(
        &vec_ops,
        parameter_select,
        sigma,
        rho,
        beta,
        epsilon,
        delta
    ); // use second parameter as a bifurcation parameter.
    lorentz.set_period_point( x0 );

    sigma = lorentz.get_selected_parameter_value();

    periodic_orbit_nonlinear_operator_t periodic_orbit_nonlin_op( &vec_ops, &lorentz, &log, 100.0, sigma, scheme_name );

    auto periodic_orbit_lin_op = periodic_orbit_nonlin_op.linear_operator;

    lin_solve_t solver( &vec_ops, &log, 0 );
    solver.set_basis_size( 1 );
    real   rel_tol   = 1.0e-2;
    size_t max_iters = 100;
    auto  &mon       = solver.monitor();
    mon.init( rel_tol, real( 0 ), max_iters );
    mon.set_save_convergence_history( true );
    mon.set_divide_out_norms_by_rel_base( true );

    system_operator_single_section_t      sys_op( &vec_ops, periodic_orbit_lin_op, &solver );
    convergence_strategy_single_section_t convergence( &vec_ops, &log );
    newton_solver_t                       newton( &vec_ops, &sys_op, &convergence );

    periodic_orbit_nonlin_op.set_hyperplane_from_initial_guesses( x0, sigma );

    periodic_orbit_nonlin_op.time_stepper( x0, sigma, { 0, max_time_simulation } );
    std::cout << "===== initial done ======" << std::endl;
    // periodic_orbit_nonlin_op.time_stepper( x0, sigma, { 0, max_time_simulation } );
    // std::cout << "===== initial done ======" << std::endl;
    periodic_orbit_nonlin_op.save_norms( "lorentz_initial.dat" );


    /*
      periodic_orbit_nonlin_op.F(x0, mu, b);
      auto b_norm = vec_ops.norm(b);
      log.info_f("=> ||b|| = %le", b_norm);
      size_t n_iter = 0;
      while(b_norm>1.0e-10)
      {

          vec_ops.assign_scalar(0, x);
          sys_op.solve(&periodic_orbit_nonlin_op, x0, mu, x);
          real wight = 1.0;
          real b_norm_new = std::numeric_limits<real>::max();
          vec_ops.assign(x0, x1);
          while(b_norm_new>1.5*b_norm)
          {
              vec_ops.add_mul(wight, x, x1);
              periodic_orbit_nonlin_op.F(x1, mu, b);
              b_norm_new = vec_ops.norm(b);
              wight *= 0.5;
              if(wight < 1.0e-6)
              {
                  log.error_f("failed with wight = %le and ||b|| = %le", wight,
     b_norm_new); break;
              }
          }
          if(wight < 1.0e-6)
          {
              break;
          }
          vec_ops.assign(x1, x0);
          // periodic_orbit_nonlin_op.set_hyperplane_from_initial_guesses(x0,
     mu); b_norm = b_norm_new; log.info_f("=> ||b|| = %le", b_norm);
          std::stringstream fn;
          fn << "lorentz_convergence_" << n_iter++ << ".dat";
          periodic_orbit_nonlin_op.save_period_estmate_norms( fn.str() );
      }
  */
    // x0[0] = 0.777;
    // x0[1] = 3.28;
    // x0[2] = 25.6;
    // x0[0] = 8.32;
    // x0[1] = 8.63;
    // x0[2] = 16.9;

    newton.solve( &periodic_orbit_nonlin_op, x0, sigma );

    for ( int j = 0; j < 3; j++ )
    {
        std::cout << x0(j) << std::endl;
    }

    std::stringstream ss_periodic_estimate;
    ss_periodic_estimate << "lorentz_period_" << scheme_name << ".dat";
    periodic_orbit_nonlin_op.save_period_estmate_norms( ss_periodic_estimate.str() );

    vec_ops.stop_use_vectors( x0, x1, b, x );
    vec_ops.free_vectors( x0, x1, b, x );
    return 0;
}
