#include "bdf_flight_test_support.h"
#include "newton_test_support.h"
using namespace bdf_flights;

// The radial oscillator has the isolated r=1 cycle, period 2*pi and transverse
// multiplier exp(-4*pi). Each section flight restarts its own adaptive BDF1-5
// history. Correct displaced section radii without introducing extra unknowns.
void check( std::size_t count, unsigned int cap )
{
    std::cout << "Adaptive BDF matching: M=" << count << ", order cap=" << cap << std::endl;
    cycle_tests::fixture f( count );
    using chart_type   = cycle_tests::chart;
    using problem_type = periodic_orbit::multiple_section_problem<
        operations, operations, adaptive_map, provider_factory, chart_type, factory_type>;
    auto       providers = std::make_shared<provider_factory>( provider_factory{ f.physical, {} } );
    const auto before    = guarded_memory::allocation_calls;
    auto factory = std::make_shared<factory_type>( f.physical, providers, factory_type::params{ cap, 1000000, true } );
    require( guarded_memory::allocation_calls == before, "Adaptive factory allocated eagerly" );
    std::vector<problem_type::segment> segments;
    for ( std::size_t i = 0; i < count; ++i )
        segments.push_back(
            { std::make_shared<adaptive_map>(
                  f.physical, make_adaptive_step( f.physical, {}, cap, 1e-7 ),
                  std::make_shared<physical_rate>( physical_rate{ f.physical, {} } )
              ),
              std::make_shared<chart_type>( f.physical, f.coordinates, f.layout->at( i ).section ) }
        );
    auto problem = std::make_shared<problem_type>( f.physical, f.space, f.layout, std::move( segments ), factory );
    auto initial = cycle_tests::block_buffer( f.space ), result = cycle_tests::block_buffer( f.space );
    auto residual = cycle_tests::block_buffer( f.space ), saved = cycle_tests::block_buffer( f.space );
    f.space->assign_scalar( .02, *initial );
    problem->apply( *initial, *result );
    require( providers->creations == 0, "Adaptive residual constructed sensitivity storage" );
    problem->set_linearization_point( *initial );
    require( providers->creations == 1, "Adaptive matching allocated workspace per flight" );
    auto jac = problem->get_jacobi_operator();
    f.space->assign_scalar( .3, *residual );
    const auto allocated = guarded_memory::allocation_calls;
    jac->apply( *residual, *saved );
    jac->apply( *residual, *result );
    require( guarded_memory::allocation_calls == allocated, "Adaptive matching Jv allocated vectors" );
    f.space->add_lin_comb( -1, *saved, *result );
    require( f.space->norm( *result ) < 1e-12, "Adaptive matching Jv changed between calls" );
    providers->flags->fail_creation = true;
    require_throws<std::runtime_error>(
        [&] { problem->set_linearization_point( *initial ); }, "Failed adaptive preparation published"
    );
    require( problem->get_jacobi_operator() == jac, "Failed adaptive preparation replaced Jacobian" );
    providers->flags->fail_creation = false;
    auto solver                     = section_tests::make_newton( f.space, problem, 1e-9, 12 );
    require( solver->solve( *initial, *result ), "Adaptive BDF orbit correction failed" );
    problem->apply( *result, *residual );
    require( f.space->norm( *residual ) < 1e-9, "Adaptive BDF corrected residual" );
    require( std::abs( problem->report().period - 2 * pi ) < 1e-4, "Adaptive BDF period accuracy" );
    f.space->assign_scalar( .3, *residual );
    jac->apply( *residual, *result );
    f.space->add_lin_comb( -1, *saved, *result );
    require( f.space->norm( *result ) < 1e-12, "Adaptive trials corrupted published history" );
    solver->set_nonlinear_operator( nullptr );
    std::weak_ptr<problem_type> weak = problem;
    problem.reset();
    require( weak.expired(), "Adaptive matching ownership cycle" );
    jac->apply( *residual, *result );
    require( f.space->is_valid_number( *result ), "Adaptive extracted Jacobian lost ownership" );
}
void test()
{
    for ( std::size_t count : { 1u, 2u, 4u, 8u } )
        check( count, 5 );
    check( 2, 2 );
    require( guarded_memory::host_bytes == 0, "Adaptive matching transferred vectors" );
    require( guarded_memory::live_allocations == 0, "Adaptive matching leaked storage" );
}
int main()
{
    return run( test );
}
