#include "benchmarks/discovered_branch.h"
#include "benchmarks/ks1d_branch_flights.h"
#include "benchmarks/ks1d_branch_reference.h"

namespace
{
using namespace ks1d_orbit;
using flight = periodic_orbit::section_map<operations, rk, plane, policy, rate>;

void check( unsigned m, const std::filesystem::path &directory )
{
    branch_flights f( m, dt, 0., budget );
    auto rhs = provider( f.physical, p0 );
    flight::params limits;
    limits.maximum_steps = budget;
    flight map( f.physical, stepper( f.physical, rhs, dt ), std::make_shared<policy>( f.physical ), limits, rhs );
    flight::request_type req;
    req.target = f.layout->at(0).section;
    req.rule.minimum_crossing_speed = 1e-10;
    orbit_benchmarks::archive_trace_settings controls{ lower, upper, m == 1 ? .003 : .009, budget, 1e-7, 40, 1e-9 };
    orbit_benchmarks::discovered_branch batch( f, map, req, dt, controls, { 1e-9, 1e-7, 1e-7, 1e-6 }, 1e-9 );

    auto u = buffer( f.physical ), difference = buffer( f.physical );
    auto low = buffer( f.physical ), high = buffer( f.physical );
    fill_profile( *low, reference[0].coefficients );
    fill_profile( *high, reference[1].coefficients );
    const auto bytes = guarded_memory::host_bytes;
    unsigned endpoints = 0;
    auto on_point = [&]( auto, auto, double p, auto, const branch_flights::state &z, const auto &v ) {
        if ( p != lower && p != upper ) return;
        const auto i = p == upper ? 1 : 0;
        f.first->lift( z.comp(0), *u );
        f.physical->assign_mul( 1., *u, -1., i ? *high : *low, *difference );
        const double distance = f.physical->norm( *difference );
        std::cout << "M=" << m << " lambda=" << p << " state difference=" << distance
                  << " refined closure=" << v.closure_error << '\n';
        require( distance < 1e-7 && std::abs( v.period - reference[i].period ) < 1e-9 && v.closure_error < 1e-7,
                 "Discovered KS endpoint disagrees with the independent reference" );
        endpoints |= 1u << i;
    };
    auto trace = [&]( double p, std::size_t index, const branch_flights::state &z ) {
        const auto data = batch.record( { "discovery", "symmetric_KS1D", "N24:odd-R2C:a2:b4",
                                          "lambda", f.observable_label, "rms" }, p, index, z, 40, on_point );
        const auto &arms = data.at( "segments" );
        orbit_benchmarks::save_discovery_archive( directory, data );
        return data.at( "trace_complete" ) == true && data.at( "all_points_verified" ) == true && arms.size() == 2 &&
               arms.at(0).at( "endpoint" ) == "boundary_min" && arms.at(1).at( "endpoint" ) == "boundary_max";
    };
    auto initialize = []( double perturbation, branch_flights::state &z ) { perturb_sections<seed.size()>( perturbation, z ); };
    const auto result = batch.run( std::array<double,1>{ .001 }, initialize, trace );
    const auto &attempt = result.attempts.at(0);
    if ( attempt.outcome != periodic_orbit::seed_outcome::activated )
        std::cerr << "M=" << m << " discovery outcome=" << int( attempt.outcome )
                  << " crossing status=" << int( attempt.crossing ) << '\n';
    require( result.discovery_complete && result.launches.size() == 1 && result.launches[0].trace_complete &&
             result.launches[0].root_index == 0 && endpoints == 3, "KS catalog orbit did not reach both reference endpoints" );
    require( guarded_memory::host_bytes == bytes, "KS discovery or continuation transferred a state" );
}
}

int main( int argc, char **argv )
{
    return section_tests::run( [&] {
        const auto directory = orbit_benchmarks::discovery_output( argc, argv );
        for ( unsigned m : { 1u, 4u } ) check( m, directory );
    } );
}
