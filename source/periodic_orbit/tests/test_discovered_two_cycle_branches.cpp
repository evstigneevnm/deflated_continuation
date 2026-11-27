#include "benchmarks/discovered_branch.h"
#include "benchmarks/ode_continuation_fixture.h"
#include "two_cycle_problem.h"

namespace
{
using namespace ode_orbit_tests;
using setup = continuation_fixture<two_cycles>;
using flight = periodic_orbit::section_map<operations, step<two_cycles>, plane, policy, rate_provider<two_cycles>>;
using samples = periodic_orbit::verified_knot_samples<operations>;

auto radius_seeds( setup &f, X &scratch )
{
    return [&]( double radius, setup::state &z ) {
        for ( std::size_t i = 0; i < f.charts.size(); ++i )
        {
            f.physical->assign_mul( radius, f.layout->at( i ).section->center(), scratch );
            f.charts[i]->restrict( scratch, z.comp( i ) );
        }
    };
}

void check_target(
    setup &source, const flight::request_type &req, double h, orbit_benchmarks::archive_trace_settings controls,
    const samples &inner, const samples *outer
)
{
    const double p = inner.parameter();
    setup        f( source.physical, source.layout, p, 2 * pi / p, h, controls.budget );
    auto         rate = std::make_shared<rate_provider<two_cycles>>();
    rate->model       = f.model;
    flight map( f.physical, make_step( f.physical, h, f.model ), std::make_shared<policy>( f.physical ), {}, rate );
    orbit_benchmarks::discovered_branch target( f, map, req, h, controls, { 1e-6, 1e-6, 1e-6, 1e-5 }, 1e-6 );
    auto install = [&]( const samples &hits, std::size_t index, periodic_orbit::orbit_admission decision ) {
        require( hits.complete() && hits.size() == 1, "Unresolved analytical intersection" );
        const auto r = target.preload_verified( hits, source );
        require( r.complete() && r.entries.size() == 1, "Target preload failed" );
        const auto &e = r.entries.at( 0 );
        require(
            e.source.branch == hits.at( 0 ).source.branch && e.source.point == hits.at( 0 ).source.point &&
                e.source.segment == hits.at( 0 ).source.segment && e.admission.root_index == index &&
                e.admission.decision == decision,
            "Lost source-to-exclusion mapping"
        );
    };
    install( inner, 0, periodic_orbit::orbit_admission::inserted );
    install( inner, 0, periodic_orbit::orbit_admission::duplicate );
    if ( outer )
        install( *outer, 1, periodic_orbit::orbit_admission::inserted );
    auto       scratch = buffer( f.physical );
    const auto result  = target.run(
        std::array<double, 3>{ 1.15, 1., 2. }, radius_seeds( f, *scratch ),
        [&]( double lambda, std::size_t index, const setup::state &z ) {
            require( !outer && index == 1 && lambda == p, "Dispatched an imported orbit" );
            const auto data = target.record(
                { "target", "two_cycles", "ode:2", "omega", f.observable_label, "euclidean" }, lambda, index, z
            );
            for ( const auto &arm : data.at( "segments" ) )
                for ( const auto &point : arm.at( "points" ) )
                    require(
                        point.at( "verified" ) == true &&
                            std::abs( point.at( "observable_rms" ).get<double>() - 2. ) < 1e-6 &&
                            std::abs(
                                point.at( "period" ).get<double>() - 2 * pi / point.at( "parameter" ).get<double>()
                            ) < 1e-6,
                        "Target trace is not the outer cycle"
                    );
            return data.at( "trace_complete" ).get<bool>();
        }
    );
    require(
        result.discovery_complete && result.attempts.size() == 3 &&
            result.attempts.front().active_before == ( outer ? 2u : 1u ) && target.catalog().size() == 2 &&
            result.launches.size() == ( outer ? 0u : 1u ) && ( outer || result.launches.front().trace_complete ),
        "Wrong exclusions or launches at second knot"
    );
}

void check( unsigned m, bool cap_first, const std::filesystem::path &directory )
{
    constexpr double h = .06;
    const orbit_benchmarks::archive_trace_settings controls{ .95, 1.05, cap_first ? .02 : .008, 1000, 1e-6 };
    setup f( {}, two_cycles::seed, 2*pi, m, h, 0., controls.budget );
    flight map( f.physical, make_step<two_cycles>( f.physical, h ), std::make_shared<policy>( f.physical ),
                {}, std::make_shared<rate_provider<two_cycles>>() );
    flight::request_type req;
    req.target = make_plane( f.physical, 0, 0, 0, 1 );
    req.rule.minimum_crossing_speed = .01;
    orbit_benchmarks::discovered_branch batch( f, map, req, h, controls, { 1e-6, 1e-6, 1e-6, 1e-5 }, 1e-6 );
    auto scratch = buffer( f.physical );
    auto                                initialize = radius_seeds( f, *scratch );
    samples                             inner( f.physical, 1.025, 2 ), outer( f.physical, 1.025, 2 );
    unsigned traces = 0;
    auto trace = [&]( double p, std::size_t index, const setup::state &z ) {
        ++traces;
        const bool capped = cap_first && index == 0;
        const auto data   = batch.record(
            { cap_first ? "capped" : "discovery", "two_cycles", "ode:2", "omega", f.observable_label, "euclidean" }, p,
            index, z, capped ? 1 : 40, periodic_orbit::no_verified_point_action{},
            cap_first    ? nullptr
                                              : index == 0 ? &inner
                                                           : &outer
        );
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
    auto                       result = cap_first ? batch.run( seeds, initialize, trace )
                                                  : batch.run( std::array<double, 1>{ 1.04 }, initialize, trace );
    if ( !cap_first )
    {
        require(
            traces == 1 && result.discovery_complete && result.launches.at( 0 ).trace_complete,
            "Inner source branch failed"
        );
        check_target( f, req, h, controls, inner, nullptr );
        const auto more = batch.run( std::array<double, 2>{ 1.15, 1. }, initialize, trace );
        require( more.discovery_complete, "Second source batch failed" );
        result.launches.insert( result.launches.end(), more.launches.begin(), more.launches.end() );
        check_target( f, req, h, controls, inner, &outer );
    }
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
