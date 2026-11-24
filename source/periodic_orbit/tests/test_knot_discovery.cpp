#include "benchmarks/ode_continuation_fixture.h"
#include "benchmarks/branch_archive_trace.h"
#include "two_cycle_problem.h"
#include <periodic_orbit/knot_discovery.h>

namespace
{
using namespace ode_orbit_tests;
using periodic_orbit::seed_outcome;

void check_knot( unsigned m )
{
    using setup = continuation_fixture<two_cycles>;
    using plain_op = nonlinear_operators::fixed_parameter_operator<setup::space, setup::orbit, setup::orbit>;
    using flight = periodic_orbit::section_map<operations, step<two_cycles>, plane, policy, rate_provider<two_cycles>>;
    constexpr double p = 1., h = .06, tol = 2e-10;
    constexpr unsigned budget = 1000;
    const double period = 2*pi;
    setup f( {}, two_cycles::seed, period, m, h, 0, budget );
    setup check( f.physical, f.layout, p, period, h, budget );
    auto u = buffer( f.physical ), scratch = buffer( f.physical );
    auto trial = f.workspace(), solved = f.workspace(), residual = f.workspace();
    flight map( f.physical, make_step<two_cycles>( f.physical, h ), std::make_shared<policy>( f.physical ),
                {}, std::make_shared<rate_provider<two_cycles>>() );
    flight::request_type req;
    req.target = make_plane( f.physical, 0, 0, 0, 1 );
    req.rule.minimum_crossing_speed = .01;
    periodic_orbit::orbit_comparison<operations, flight> cmp( f.physical, map, p, req, { 1e-6, 1e-6, 1e-6, 1e-5 } );
    periodic_orbit::knot_discovery knot( cmp, f.coords, f.op, 4, 1e-6, 1e-6, 1e-7 );
    auto ordinary = make_newton( f.coords, std::make_shared<plain_op>( f.op, f.op, p ), tol, 16 );
    auto deflated = make_newton( f.coords, knot.deflated_operator(), tol, 16, deflation::deflated_residual_norm{} );
    auto correct = [&]( auto &solver, double radius ) {
        for ( unsigned i = 0; i < m; ++i )
        {
            f.physical->assign_mul( radius, f.layout->at(i).section->center(), *scratch );
            f.charts[i]->restrict( *scratch, (*trial).comp(i) );
        }
        return periodic_orbit::try_orbit_solve<double>( [&]{ return solver->solve( *trial, *solved ); } );
    };
    unsigned plain_calls = 0, recharted = 0;
    auto plain = [&]( double r ){ ++plain_calls; return correct( ordinary, r ); };
    unsigned deflated_calls = 0;
    auto search = [&]( double r ){
        if ( ++deflated_calls == 2 )
            require_throws<std::logic_error>( [&]{ knot.deflated_operator()->get_jacobi_operator(); },
                                              "A new root left the old Jacobian prepared" );
        return correct( deflated, r );
    };
    auto verify = [&]( X &anchor ) -> std::optional<double> {
        const orbit_benchmarks::archive_trace_settings settings{ p, p, 0, budget, 1e-6 };
        auto v = orbit_benchmarks::orbit_measurement( check, *solved, p, *residual, anchor, settings );
        if ( !v || !v->valid() ) return {};
        return v->period;
    };
    auto rechart = [&]( const X &state, double T, setup::state &z ) {
        ++recharted;
        f.physical->assign( state, *scratch );
        return periodic_orbit::rechart_orbit( map, *f.layout, f.charts, *scratch, T, z );
    };
    const std::array<double,3> seeds{ 1.04, 1.15, 1. };
    auto first = knot.run( seeds, *u, plain, search, verify, rechart );
    require( first.size() == 3 && knot.catalog().size() == 2 && knot.active_count() == 2, "Incomplete discovery" );
    for ( unsigned i = 0; i < 2; ++i )
    {
        require( first[i].seed_index == i && first[i].active_before == i &&
                 first[i].outcome == seed_outcome::activated && first[i].root_index == i, "Wrong solve/activation order" );
        const auto &root = knot.catalog().at(i);
        require( std::abs( f.physical->norm( root.state() ) - (i+1) ) < 1e-6 &&
                 std::abs( root.period() - period ) < 1e-6, "Wrong verified radius or period" );
    }
    require( first[2].active_before == 2 && recharted == 2 && plain_calls == 1,
             "Discovery reused stale exclusions or recharted old roots" );
    const auto replay = knot.run( seeds, *u, plain, search, verify, rechart );
    require( replay.size() == seeds.size() && knot.catalog().size() == 2 && recharted == 2 && plain_calls == 1,
             "Replay inserted or recharted a root" );
    for ( const auto &a : replay )
        require( a.active_before == 2 && !a.activation, "Replay changed the active set" );

    std::cout << "M=" << m << ": active before each seed:";
    for ( const auto &a : first ) std::cout << ' ' << a.active_before;
    std::cout << "; verified orbits=" << knot.catalog().size() << "; replay attempts=" << replay.size()
              << "; total recharts=" << recharted << '\n';

    // Drive the failure paths with the verified physical anchors from the real
    // solves. No attempt after the second root's failed activation may execute.
    periodic_orbit::knot_discovery failing( cmp, f.coords, f.op, 4, 1e-6, 1e-6, 1e-7 );
    unsigned calls = 0, verifications = 0, activations = 0;
    int current = -1;
    auto scripted = [&]( int seed ){ ++calls; current = seed; return seed != 0; };
    auto inspect = [&]( X &anchor ) -> std::optional<double> {
        ++verifications;
        if ( current == 1 ) return {};
        const auto &root = knot.catalog().at( current < 5 ? 0 : 1 );
        f.physical->assign( root.state(), anchor );
        return current == 4 ? 2*root.period() : root.period();
    };
    auto fail_second = [&]( const X &state, double T, setup::state &z ) {
        return ++activations != 2 && rechart( state, T, z );
    };
    const std::array<int,7> scripts{ 0, 1, 2, 3, 4, 5, 6 };
    const auto failed = failing.run( scripts, *u, scripted, scripted, inspect, fail_second );
    require( failed.size() == 6 && calls == 6 && verifications == 5 && activations == 2 &&
             failing.active_count() == 1 && failing.catalog().size() == 2, "Activation failure did not halt the batch" );
    require( failed[0].outcome == seed_outcome::correction_failed && failed[1].outcome == seed_outcome::verification_failed &&
             failed[2].outcome == seed_outcome::activated && failed[3].outcome == seed_outcome::duplicate &&
             failed[4].outcome == seed_outcome::inconclusive && failed[5].outcome == seed_outcome::activation_failed &&
             failed[5].root_index == 1 && failed[5].activation == periodic_orbit::chart_root_status::rechart_failed,
             "Missing per-seed diagnostics" );
    require_throws<std::logic_error>( [&]{ failing.run( scripts, *u, scripted, scripted, inspect, fail_second ); },
                                      "Restarted a stale exclusion set" );
    require( calls == 6, "Attempted a solve after a failed activation" );

    periodic_orbit::knot_discovery limited( cmp, f.coords, f.op, 1, 1e-6, 1e-6, 1e-7 );
    const auto before = recharted;
    unsigned launches = 0;
    auto launch = [&]( double, std::size_t, const auto & ){ ++launches; return true; };
    const auto incomplete = limited.run_and_continue( std::array<int,3>{ 2, 5, 6 }, *u, scripted, scripted, inspect, rechart, launch );
    const auto &exhausted = incomplete.attempts;
    require( !incomplete.discovery_complete && incomplete.launches.empty() && !launches, "Launched a partial batch" );
    require( exhausted.size() == 2 && exhausted[0].outcome == seed_outcome::activated &&
             exhausted[1].activation == periodic_orbit::chart_root_status::capacity_reached &&
             limited.active_count() == 1 && recharted == before + 1, "Capacity exceeded or old roots recharted" );

    periodic_orbit::knot_discovery prepared( cmp, f.coords, f.op, 2, 1e-6, 1e-6, 1e-7 );
    unsigned preparations = 0;
    auto miss_first_launch = [&]( const X &state, double T, setup::state &z ) {
        return ++preparations != 3 && rechart( state, T, z );
    };
    const auto traced = prepared.run_and_continue( std::array<int,2>{ 2, 5 }, *u, scripted, scripted, inspect,
                                                    miss_first_launch, launch );
    require( traced.discovery_complete && traced.launches.size() == 2 && launches == 1 &&
             traced.launches[0].preparation == periodic_orbit::chart_root_status::rechart_failed &&
             !traced.launches[0].trace_complete && traced.launches[1].trace_complete &&
             prepared.active_count() == 2, "Failed handoff changed catalog or suppressed next launch" );
}
}
int main()
{
    return section_tests::run( [] {
        for ( unsigned m : {1u, 4u} ) check_knot( m );
        require( guarded_memory::host_bytes == 0, "Discovery transferred state vectors" );
    } );
}
