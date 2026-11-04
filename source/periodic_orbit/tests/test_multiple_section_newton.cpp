#include "cycle_newton_support.h"
using namespace cycle_tests;

// The isolated unit cycle is corrected from perturbed section radii. Runtime
// multiple matching uses unchanged NMFD Newton/GMRES, not a project Newton loop.
void check( std::size_t count )
{
    fixture f( count, .1, "DOP853" );
    auto    rate    = std::make_shared<cycle_rate>();
    auto    problem = std::make_shared<orbit_problem>( f.physical, f.space, f.layout, recorded_segments( f ), rate );
    auto    base = block_buffer( f.space ), v = block_buffer( f.space ), w = block_buffer( f.space ),
         out = block_buffer( f.space ), saved = block_buffer( f.space ), work = block_buffer( f.space );
    f.space->assign_scalar( .02, *base );
    f.space->assign_scalar( .3, *v );
    f.space->assign_scalar( -.2, *w );
    require_throws<std::logic_error>( [&] { problem->get_jacobi_operator(); }, "Unprepared Jacobian exposed" );
    problem->set_linearization_point( *base );
    auto jacobian = problem->get_jacobi_operator();
    jacobian->apply( *v, *saved );
    jacobian->apply( *w, *work );
    f.space->assign_lin_comb( 2, *v, -3, *w, *out );
    jacobian->apply( *out, *out );
    f.space->add_mul( -2, *saved, 3, *work, 1, *out );
    require( f.space->norm( *out ) < 1e-11, "Matching Jacobian is not linear" );
    const double epsilon = 1e-5;
    f.space->assign_lin_comb( 1, *base, epsilon, *v, *out );
    problem->apply( *out, *out );
    f.space->assign_lin_comb( 1, *base, -epsilon, *v, *work );
    problem->apply( *work, *work );
    f.space->assign_lin_comb( .5 / epsilon, *out, -.5 / epsilon, *work, *out );
    f.space->add_lin_comb( -1, *saved, *out );
    require( f.space->norm( *out ) < 2e-6, "Matching Jacobian finite difference" );
    jacobian->apply( *v, *out );
    f.space->add_lin_comb( -1, *saved, *out );
    require( f.space->norm( *out ) < 1e-13, "Trial residual overwrote prepared Jacobian" );
    rate->fail_rate = true;
    require_throws<std::runtime_error>(
        [&] { problem->set_linearization_point( *base ); }, "Failed preparation accepted"
    );
    require( problem->get_jacobi_operator().get() == jacobian.get(), "Failed preparation replaced old Jacobian" );
    rate->fail_rate = false;
    jacobian->apply( *v, *out );
    f.space->add_lin_comb( -1, *saved, *out );
    require( f.space->norm( *out ) < 1e-13, "Failed preparation corrupted old Jacobian" );
    const auto saved_time = jacobian->flight_time_increment( 0 );
    f.space->assign_scalar( 7, *out );
    const auto output_norm = f.space->norm( *out );
    rate->fail_jacobian    = true;
    require_throws<std::runtime_error>( [&] { jacobian->apply( *v, *out ); }, "Matching tangent failure ignored" );
    require(
        f.space->norm( *out ) == output_norm && jacobian->flight_time_increment( 0 ) == saved_time,
        "Failed matching Jv published output or diagnostics"
    );
    rate->fail_jacobian = false;

    auto newton = make_newton( f.space, problem );
    require( newton->solve( *base, *out ), "Multiple-section Newton correction failed" );
    problem->apply( *out, *work );
    require(
        f.space->norm( *work ) < 1e-10 && std::abs( problem->report().period - 2 * pi ) < 1e-7,
        "Corrected cycle residual/period"
    );
    require( f.space->norm( *out ) < 1e-7, "Newton did not recover the unit-cycle section states" );
    require( newton->solve( *base, *out ), "Repeated orbit correction failed" );
    jacobian->apply( *v, *out );
    f.space->add_lin_comb( -1, *saved, *out );
    require( f.space->norm( *out ) < 1e-13, "Rebinding invalidated an extracted Jacobian" );
    std::weak_ptr<orbit_problem> weak = problem;
    newton->set_nonlinear_operator( nullptr );
    problem.reset();
    require( weak.expired(), "Matching operator ownership cycle" );
    jacobian->apply( *v, *out );
    require( f.space->is_valid_number( *out ), "Extracted Jacobian lost dependencies" );
}
void test()
{
    for ( const auto count : { 1u, 2u, 4u, 8u } )
        check( count );
    require( guarded_memory::host_bytes == 0, "Newton correction transferred vector data" );
}
int main()
{
    return run( test );
}
