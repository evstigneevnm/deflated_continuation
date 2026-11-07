#include "implicit_flight_test_support.h"
using namespace implicit_flights;

// Correct the isolated unit cycle through 1,2,4 or 8 transverse sections.
// The matching/Newton interface is unchanged; only the prepared-flight factory
// selects DIRK. A separate workspace is allocated per published Jacobian.
void check( std::size_t count )
{
    cycle_tests::fixture f( count, .04, "RK45" );
    using chart_type   = cycle_tests::chart;
    using problem_type = periodic_orbit::multiple_section_problem<
        operations, operations, recorded_map, provider_factory, chart_type, factory_type>;
    auto       providers = std::make_shared<provider_factory>( provider_factory{ f.physical, {} } );
    const auto baseline  = guarded_memory::allocation_calls;
    auto       factory   = std::make_shared<factory_type>( f.physical, providers );
    require( guarded_memory::allocation_calls == baseline, "Factory allocated before preparation" );
    std::vector<problem_type::segment> segments;
    for ( std::size_t i = 0; i < count; ++i )
        segments.push_back(
            { make_map( f.physical, {}, "SDIRK2(1)2", .04 ),
              std::make_shared<chart_type>( f.physical, f.coordinates, f.layout->at( i ).section ) }
        );
    require_throws<std::invalid_argument>(
        [&] { problem_type missing( f.physical, f.space, f.layout, segments, nullptr ); },
        "Legacy null-provider constructor no longer rejects invalid configuration"
    );
    auto problem = std::make_shared<problem_type>( f.physical, f.space, f.layout, std::move( segments ), factory );
    auto base = cycle_tests::block_buffer( f.space ), v = cycle_tests::block_buffer( f.space ),
         out = cycle_tests::block_buffer( f.space ), saved = cycle_tests::block_buffer( f.space );
    f.space->assign_scalar( .02, *base );
    f.space->assign_scalar( .3, *v );
    problem->apply( *base, *out );
    require( providers->creations == 0, "Residual allocated implicit sensitivity provider" );
    problem->set_linearization_point( *base );
    require( providers->creations == 1, "Matching allocated a workspace per segment instead of per Jacobian" );
    auto       jac       = problem->get_jacobi_operator();
    const auto allocated = guarded_memory::allocation_calls;
    jac->apply( *v, *saved );
    jac->apply( *v, *out );
    require( guarded_memory::allocation_calls == allocated, "Matching DIRK Jv allocated backend vectors" );
    f.space->add_lin_comb( -1, *saved, *out );
    require( f.space->norm( *out ) < 1e-12, "Repeated DIRK matching derivative changed" );
    providers->flags->fail_creation = true;
    require_throws<std::runtime_error>(
        [&] { problem->set_linearization_point( *base ); }, "Failed DIRK preparation accepted"
    );
    require( problem->get_jacobi_operator().get() == jac.get(), "Failed factory replaced prepared Jacobian" );
    providers->flags->fail_creation = false;
    problem->set_linearization_point( *base );
    jac->apply( *v, *out );
    f.space->add_lin_comb( -1, *saved, *out );
    require( f.space->norm( *out ) < 1e-12, "New preparation invalidated old stage bindings" );
    using matching_linear = nmfd::solvers::gmres<
        problem_type::space_type, nmfd::solvers::monitor_krylov<problem_type::space_type, log_type>, log_type,
        problem_type::jacobi_operator_type>;
    using matching_iteration = nmfd::solvers::newton_iteration<problem_type::space_type, problem_type, matching_linear>;
    using matching_solver =
        nmfd::solvers::nonlinear_solver<problem_type::space_type, log_type, problem_type, matching_iteration>;
    using strategy = nmfd::solvers::default_convergence_strategy<problem_type::space_type, log_type, problem_type>;
    matching_linear::params lp;
    lp.basis_size      = 12;
    lp.batch_size      = 1;
    lp.monitor.abs_tol = 1e-11;
    lp.monitor.rel_tol = 1e-9;
    strategy::params cp;
    cp.abs_tol       = 1e-9;
    cp.rel_tol       = 1e-10;
    cp.max_iters_num = 15;
    matching_solver newton(
        f.space, nullptr,
        std::make_shared<matching_iteration>( f.space, std::make_shared<matching_linear>( f.space, nullptr, lp ) ),
        std::make_shared<strategy>( f.space, nullptr, cp )
    );
    newton.set_nonlinear_operator( problem );
    require( newton.solve( *base, *out ), "DIRK periodic matching Newton failed" );
    problem->apply( *out, *v );
    require(
        f.space->norm( *v ) < 1e-9 && std::abs( problem->report().period - 2 * pi ) < .01,
        "DIRK corrected orbit residual/period"
    );
    require( f.space->norm( *out ) < .01, "DIRK corrected orbit is not near the unit cycle" );
    newton.set_nonlinear_operator( nullptr );
    std::weak_ptr<problem_type> weak = problem;
    problem.reset();
    require( weak.expired(), "Implicit matching ownership cycle" );
    f.space->assign_scalar( .3, *v );
    jac->apply( *v, *out );
    require( f.space->is_valid_number( *out ), "Extracted DIRK Jacobian lost dependencies" );
}
void test()
{
    for ( std::size_t count : { 1u, 2u, 4u, 8u } )
        check( count );
    require( guarded_memory::host_bytes == 0, "DIRK matching transferred vectors" );
    require( guarded_memory::live_allocations == 0, "DIRK matching workspace leaked" );
}
int main()
{
    return run( test );
}
