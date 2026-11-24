#include "benchmarks/discovered_branch.h"
#include "benchmarks/ode_continuation_fixture.h"
#include "two_cycle_problem.h"

namespace
{
using namespace ode_orbit_tests;
using setup = continuation_fixture<two_cycles>;
using flight = periodic_orbit::section_map<operations, step<two_cycles>, plane, policy, rate_provider<two_cycles>>;

void check( unsigned m, bool cap_first, const std::filesystem::path &directory )
{
    constexpr double h = .06;
    const orbit_benchmarks::archive_trace_settings controls{ .95, 1.05, .02, 1000, 1e-6 };
    setup f( {}, two_cycles::seed, 2*pi, m, h, 0., controls.budget );
    flight map( f.physical, make_step<two_cycles>( f.physical, h ), std::make_shared<policy>( f.physical ),
                {}, std::make_shared<rate_provider<two_cycles>>() );
    flight::request_type req;
    req.target = make_plane( f.physical, 0, 0, 0, 1 );
    req.rule.minimum_crossing_speed = .01;
    orbit_benchmarks::discovered_branch batch( f, map, req, h, controls, { 1e-6, 1e-6, 1e-6, 1e-5 }, 1e-6 );
    auto scratch = buffer( f.physical );
    auto initialize = [&]( double radius, setup::state &z ) {
        for ( unsigned i = 0; i < m; ++i )
        {
            f.physical->assign_mul( radius, f.layout->at(i).section->center(), *scratch );
            f.charts[i]->restrict( *scratch, z.comp(i) );
        }
    };
    unsigned traces = 0;
    auto trace = [&]( double p, std::size_t index, const setup::state &z ) {
        ++traces;
        const bool capped = cap_first && index == 0;
        const auto data = batch.record( { cap_first ? "capped" : "discovery", "two_cycles", "ode:2", "omega", f.observable_label, "euclidean" },
                                        p, index, z, capped ? 1 : 40 );
        orbit_benchmarks::save_discovery_archive( directory, data );
        const auto &arms = data.at( "segments" );
        require( data.at( "trace_complete" ) == !capped && data.at( "all_points_verified" ) == true &&
                 data.at( "discovered_at" ).at( "parameter" ) == 1. &&
                 data.at( "discovered_at" ).at( "root_index" ) == index && arms.size() == 2, "Invalid branch archive" );
        const auto radius = f.physical->norm( batch.catalog().at( index ).state() );
        for ( unsigned i = 0; i < 2; ++i )
        {
            const auto &arm = arms.at(i), &points = arm.at( "points" );
            require( arm.at( "endpoint" ) == ( capped ? "max_steps" : i ? "boundary_max" : "boundary_min" ),
                     "Wrong branch endpoint" );
            if ( !capped ) require( points.back().at( "parameter" ) == (i ? controls.upper : controls.lower),
                                    "Missed analytical boundary" );
            for ( const auto &point : points )
                require( std::abs( point.at( "observable_rms" ).get<double>() - radius ) < 1e-6 &&
                         std::abs( point.at( "period" ).get<double>() - 2*pi/point.at( "parameter" ).get<double>() )
                         < 1e-6, "Continued the wrong orbit or period" );
        }
        return data.at( "trace_complete" ).get<bool>();
    };
    const std::array<double,3> seeds{ 1.04, 1.15, 1. };
    const auto result = batch.run( seeds, initialize, trace );
    require( result.discovery_complete && traces == 2 && result.launches.size() == 2 &&
             batch.catalog().size() == 2, "Expected two activated branches" );
    for ( unsigned i = 0; i < 2; ++i )
        require( result.launches[i].root_index == i &&
                 result.launches[i].preparation == periodic_orbit::chart_root_status::ready &&
                 result.launches[i].trace_complete == ( !cap_first || i != 0 ) &&
                 std::abs( f.physical->norm( batch.catalog().at(i).state() ) - (i+1) ) < 1e-6,
                 "Trace completion or catalog identity changed" );
    const auto replay = batch.run( seeds, initialize, trace );
    require( replay.discovery_complete && replay.attempts.size() == seeds.size() && replay.launches.empty() &&
             traces == 2, "Seed replay launched another branch" );
}
}

int main( int argc, char **argv )
{
    return section_tests::run( [&] {
        const auto directory = orbit_benchmarks::discovery_output( argc, argv );
        for ( unsigned m : { 1u, 4u } ) check( m, false, directory );
        check( 1, true, directory );
        require( guarded_memory::host_bytes == 0, "Analytical discovery or continuation transferred a state" );
    } );
}
