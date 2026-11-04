#ifndef PERIODIC_ORBIT_ODE_ORBIT_TEST_H
#define PERIODIC_ORBIT_ODE_ORBIT_TEST_H

#include "ode_orbit_support.h"

namespace ode_orbit_tests
{
template <class Model>
void test_correction( std::size_t count )
{
    fixture<Model> f( count, Model::default_step( "DOP853" ) );
    auto           z = f.block_buffer(), residual = f.block_buffer(), v = f.block_buffer(), image = f.block_buffer(),
         work = f.block_buffer();
    f.perturb( .02, *z );
    f.perturb( .3, *v );
    f.problem->set_linearization_point( *z );
    auto jacobian = f.problem->get_jacobi_operator();
    jacobian->apply( *v, *image );
    const double epsilon = 1e-5;
    f.space->assign_lin_comb( 1, *z, epsilon, *v, *work );
    f.problem->apply( *work, *work );
    f.space->assign_lin_comb( 1, *z, -epsilon, *v, *residual );
    f.problem->apply( *residual, *residual );
    f.space->assign_lin_comb( .5 / epsilon, *work, -.5 / epsilon, *residual, *work );
    f.space->add_lin_comb( -1, *image, *work );
    require( f.space->norm( *work ) < 2e-5, "ODE section derivative finite difference" );
    f.space->assign_lin_comb( 2, *v, *work );
    jacobian->apply( *work, *work );
    f.space->add_lin_comb( -2, *image, *work );
    require( f.space->norm( *work ) < 1e-10, "ODE prepared derivative linearity/trial isolation" );
    auto solver = make_newton( f.space, f.problem );
    require( solver->solve( *z, *work ), "ODE multiple-section Newton correction" );
    f.problem->apply( *work, *residual );
    const auto report = f.problem->report();
    require( f.space->norm( *residual ) < 1e-10, "ODE true matching residual" );
    require( std::abs( report.period - Model::period ) < 1e-6, "ODE reference period" );
    require( f.closure( *work, report.period, f.dt / 2 ) < 2e-6, "ODE independent refined closure" );
    auto repeated = f.block_buffer();
    require( solver->solve( *z, *repeated ), "Repeated ODE orbit correction" );
    f.space->add_lin_comb( -1, *work, *repeated );
    require( f.space->norm( *repeated ) < 1e-9, "Repeated ODE correction changed solution" );
    f.rate->fail_jacobian = true;
    f.space->assign_scalar( 7, *residual );
    const auto before = f.space->norm( *residual );
    require_throws<std::runtime_error>( [&] { jacobian->apply( *v, *residual ); }, "ODE tangent failure ignored" );
    require( f.space->norm( *residual ) == before, "Failed ODE tangent published partial output" );
}

template <class Model>
void test_system()
{
    // A pointwise Jv check guards equations independently of orbit matching.
    auto ops   = std::make_shared<operations>( 3 );
    auto state = buffer( ops ), v = buffer( ops ), plus = buffer( ops ), minus = buffer( ops ), image = buffer( ops ),
         positive_rate = buffer( ops ), negative_rate = buffer( ops );
    fill( *ops, *state, Model::seed[0], Model::seed[1], Model::seed[2], 3 );
    fill( *ops, *v, .2, -.3, .4, 3 );
    tangent_rate<Model> rate;
    rate.apply_jacobian( .37, *state, *v, *image );
    ops->assign_lin_comb( 1, *state, 1e-5, *v, *plus );
    rate.evaluate( .37, *plus, *positive_rate );
    ops->assign_lin_comb( 1, *state, -1e-5, *v, *minus );
    rate.evaluate( .37, *minus, *negative_rate );
    ops->assign_lin_comb( 5e4, *positive_rate, -5e4, *negative_rate, *plus );
    ops->add_lin_comb( -1, *image, *plus );
    require( ops->norm( *plus ) < 1e-7, "ODE analytical RHS Jacobian" );
    for ( const auto count : { 1u, 2u, 4u, 8u } )
        test_correction<Model>( count );
    require( guarded_memory::host_bytes == 0, "ODE orbit solve transferred vector data" );
}
}
#endif
