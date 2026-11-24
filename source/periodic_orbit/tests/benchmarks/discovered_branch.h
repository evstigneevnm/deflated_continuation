#ifndef PERIODIC_ORBIT_TESTS_DISCOVERED_BRANCH_H
#define PERIODIC_ORBIT_TESTS_DISCOVERED_BRANCH_H

#include "branch_archive_trace.h"
#include <periodic_orbit/knot_discovery.h>
#include <periodic_orbit/matching/rechart_orbit.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace orbit_benchmarks
{
inline std::filesystem::path discovery_output( int argc, char **argv )
{
    if ( argc == 1 ) return {};
    if ( argc != 3 || std::string( argv[1] ) != "--write" || !std::filesystem::create_directory( argv[2] ) )
        throw std::invalid_argument( "Expected --write NEW_DIRECTORY" );
    return argv[2];
}

inline void save_discovery_archive( const std::filesystem::path &directory, const nlohmann::json &data )
{
    if ( directory.empty() ) return;
    std::ofstream out;
    out.exceptions( std::ios::failbit | std::ios::badbit );
    out.open( directory / (data.at( "id" ).get<std::string>() + ".json") );
    out << data << '\n';
    out.close();
}

// Use a dedicated knot-bound flight: continuation may rebind f.op to an endpoint.
template <class Fixture, class Flight>
class discovered_branch
{
    using space = typename Fixture::space;
    using orbit = typename Fixture::orbit;
    using state = typename Fixture::state;
    using comparison = periodic_orbit::orbit_comparison<section_tests::operations, Flight>;
    Fixture &f, check;
    Flight &flight;
    archive_trace_settings settings;
    comparison cmp;
    periodic_orbit::knot_discovery<comparison, space, orbit> knot;
    branch_corrections<Fixture> solvers;
    using deflated_op = periodic_orbit::chart_deflated_operator<space, orbit>;
    std::shared_ptr<typename section_tests::newton_types<space, deflated_op, deflation::deflated_residual_norm>::solver> deflated;
    nmfd::detail::vector_wrap<section_tests::operations, true, true> physical, scratch;
    decltype( std::declval<Fixture &>().workspace() ) trial, solved, residual;

public:
    discovered_branch( Fixture &fixture, Flight &map, typename Flight::request_type section,
                       double h, archive_trace_settings controls, typename comparison::params tolerances,
                       double matching_tol )
        : f( fixture ), check( f.physical, f.layout, f.pilot_parameter, f.period, h, controls.budget ), flight( map ),
          settings( controls ), cmp( f.physical, flight, f.pilot_parameter, section, tolerances ),
          knot( cmp, f.coords, f.op, 4, matching_tol, tolerances.period_tolerance, 1e-7 ),
          solvers( f, 1e-10, 16 ), deflated( section_tests::make_newton( f.coords, knot.deflated_operator(),
              2e-10, 16, deflation::deflated_residual_norm{} ) ), physical( *f.physical ), scratch( *f.physical ),
          trial( f.workspace() ), solved( f.workspace() ), residual( check.workspace() )
    {}
    const auto &catalog() const { return knot.catalog(); }

    template <class Seeds, class Prepare, class Trace>
    auto run( const Seeds &seeds, Prepare prepare, Trace trace )
    {
        const double p = f.pilot_parameter;
        auto plain = [&]( const auto &seed ) {
            prepare( seed, *solved );
            return solvers.ordinary.solve( f.op.get(), *solved, p );
        };
        auto search = [&]( const auto &seed ) {
            prepare( seed, *trial );
            return periodic_orbit::try_orbit_solve<double>( [&]{ return deflated->solve( *trial, *solved ); } );
        };
        auto verify = [&]( auto &anchor ) -> std::optional<double> {
            const auto v = orbit_measurement( check, *solved, p, *residual, anchor, settings );
            if ( !v || !v->valid() ) return {};
            return v->period;
        };
        auto rechart = [&]( const auto &crossing, double period, state &z ) {
            f.physical->assign( crossing, *scratch );
            return periodic_orbit::rechart_orbit( flight, *f.layout, f.charts, *scratch, period, z );
        };
        return knot.run_and_continue( seeds, *physical, plain, search, verify, rechart, trace );
    }

    template <class OnVerified = periodic_orbit::no_verified_point_action>
    auto record( container::branch_identity id, double p, std::size_t index, const state &z,
                 unsigned max_steps = 40, OnVerified action = {} )
    {
        auto verify = [&]( double lambda, const state &point ) {
            return orbit_measurement( check, point, lambda, *residual, *physical, settings );
        };
        id.id += "-M" + std::to_string( f.layout->size() ) + "-root" + std::to_string( index );
        periodic_orbit::periodic_curve_recorder recorder( id, "RK45", f.layout->size(), verify, action );
        recorder.set_discovery_source( p, index );
        auto controls = settings;
        controls.max_steps = max_steps;
        const auto result = drive_archive( f, solvers, recorder, z, p, controls );
        std::stringstream stream;
        recorder.write( stream, result );
        return container::read_periodic_branch( stream );
    }
};
}
#endif
