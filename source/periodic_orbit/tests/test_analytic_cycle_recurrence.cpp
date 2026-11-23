#include "benchmarks/branch_archive_trace.h"
#include "benchmarks/ode_continuation_fixture.h"
#include "parameterized_cycle.h"
#include <periodic_orbit/orbit_comparison.h>
#include <periodic_orbit/recurrence_seed.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace
{
using namespace ode_orbit_tests;
using model = cycle_tests::parameterized_cycle;
using setup = continuation_fixture<model>;
using search = periodic_orbit::recurrence_seed_search<operations>;
using flight = periodic_orbit::section_map<operations, step<model>, plane, policy, rate_provider<model>>;
using comparison = periodic_orbit::orbit_comparison<operations, flight>;
constexpr double h = .05;
constexpr std::size_t budget = 2000;

comparison::crossing trace( const std::shared_ptr<operations> &space, const periodic_orbit::orbit_seed<operations> &seed,
                            std::size_t m, const std::filesystem::path &directory, comparison &cmp )
{
    const model rhs{ seed.parameter };
    auto layout = backend_layout( space, make_step( space, h, rhs ), rhs, seed.state, seed.period, m, 0., budget );
    setup f( space, layout, seed.parameter, seed.period, h, budget );
    setup check( space, layout, seed.parameter, seed.period, h, budget );
    orbit_benchmarks::branch_corrections<setup> solvers( f );
    auto z = f.workspace(), residual = check.workspace();
    auto initial = buffer( space );
    double p = seed.parameter;
    f.coords->assign_scalar( 0., *z );
    const orbit_benchmarks::archive_trace_settings settings{ .95, 1.05, .02, budget, 1e-6 };
    auto verify = [&]( double lambda, const setup::state &point ) {
        return orbit_benchmarks::orbit_measurement( check, point, lambda, *residual, *initial, settings );
    };
    require( solvers.ordinary.solve( f.op.get(), *z, p ), "Recurrence seed failed fixed-parameter correction" );
    auto anchor = verify( p, *z );
    require( anchor && anchor->valid(), "Corrected seed failed independent closure" );
    auto hit = cmp.make_crossing();
    require( cmp.prepare( *initial, p, anchor->period, hit ).status == periodic_orbit::crossing_preparation::ready,
             "Verified anchor has no unique comparison crossing" );

    const container::branch_identity id{ "recurrence-M" + std::to_string( m ), "analytic_cycle", "ode:2",
                                         "lambda", setup::observable_label, "euclidean" };
    periodic_orbit::periodic_curve_recorder recorder( id, "RK45", m, verify );
    const auto result = orbit_benchmarks::drive_archive( f, solvers, recorder, *z, p, settings );
    std::stringstream archive;
    recorder.write( archive, result );
    const auto data = container::read_periodic_branch( archive );
    require( result.complete() && data.at( "all_points_verified" ) && data.at( "segments" ).size() == 2,
             "Recurrence branch or its verification was incomplete" );
    for ( unsigned i = 0; i < 2; ++i )
    {
        const auto &arm = data.at( "segments" ).at( i );
        const auto &points = arm.at( "points" );
        require( arm.at( "endpoint" ) == (i ? "boundary_max" : "boundary_min") &&
                 points.back().at( "parameter" ) == (i ? settings.upper : settings.lower),
                 "Branch did not reach Stage 1 boundary" );
        for ( const auto &row : points )
        {
            const double lambda = row.at( "parameter" );
            require( std::abs( row.at( "period" ).get<double>() - 2 * pi / model{lambda}.omega() ) < 1e-6 &&
                     std::abs( row.at( "observable_rms" ).get<double>() - std::sqrt( lambda ) ) < 1e-6,
                     "Wrong period or radius on recovered branch" );
        }
    }
    if ( !directory.empty() )
    {
        std::ofstream out;
        out.exceptions( std::ios::badbit | std::ios::failbit );
        out.open( directory / (id.id + ".json") );
        out << archive.str();
        out.close();
    }
    std::cout << "Verified recurrence branch M=" << m << ", corrected T=" << anchor->period
              << ", independent closure=" << anchor->closure_error << '\n';
    return hit;
}
}

int main( int argc, char **argv )
{
    return section_tests::run( [&] {
        std::filesystem::path directory;
        if ( argc > 2 ) throw std::invalid_argument( "Expected at most one new archive directory" );
        if ( argc == 2 )
        {
            directory = argv[1];
            if ( !std::filesystem::create_directory( directory ) )
                throw std::invalid_argument( "Archive directory must be new" );
        }
        const auto start_bytes = guarded_memory::host_bytes;
        auto space = std::make_shared<operations>( 2 );
        auto initial = buffer( space ), output = buffer( space );
        const model rhs{ 1. };
        auto rate = std::make_shared<rate_provider<model>>();
        rate->model = rhs;
        flight map( space, make_step( space, h, rhs ), std::make_shared<policy>( space ), {}, rate );
        flight::request_type request{ 0., 50., make_plane( space, 0, 0, 0, 1 ) };
        request.rule.minimum_crossing_speed = 1e-3;
        search detector( space );
        const search::params limits{ .1, 25., 1e-5, 8 };
        fill( *space, *output, 3, 4 );
        space->assign_scalar( 0., *initial );
        auto short_run = request;
        short_run.end_time = .5;
        const auto equilibrium = detector.find( map, short_run, *initial, rhs.lambda, limits, *output );
        require( !equilibrium.candidate && equilibrium.crossings == 0 &&
                 equilibrium.last_flight.status == status::flight_limit, "Stationary origin produced a return" );
        near_state( space, *output, 3, 4, 1e-14 );

        fill( *space, *initial, .001, 0 );
        auto budget_one = limits;
        budget_one.maximum_crossings = 1;
        const auto single = detector.find( map, request, *initial, rhs.lambda, budget_one, *output );
        require( !single.candidate && single.crossings == 1, "One crossing was called a recurrence" );
        auto long_only = limits;
        long_only.minimum_period = 10.;
        long_only.maximum_crossings = 4;
        const auto filtered = detector.find( map, request, *initial, rhs.lambda, long_only, *output );
        require( !filtered.candidate && filtered.crossings == 4, "Skipped a short return to accept a multiple" );
        auto grazing = request;
        grazing.rule.minimum_crossing_speed = 2.;
        const auto slow = detector.find( map, grazing, *initial, rhs.lambda, limits, *output );
        require( !slow.candidate && slow.last_flight.status == status::grazing, "Accepted a grazing crossing" );
        near_state( space, *output, 3, 4, 1e-14 );

        const auto allocations = guarded_memory::allocation_calls;
        const auto found = detector.find( map, request, *initial, rhs.lambda, limits, *output );
        require( guarded_memory::allocation_calls == allocations, "Recurrence search allocated backend vectors" );
        require( found.candidate && found.candidate->origin == periodic_orbit::seed_origin::section_recurrence &&
                 found.crossings > 2 && found.last_distance && *found.last_distance <= limits.distance_tolerance,
                 "No converged recurrence after transient returns" );
        std::cout << "Seed: crossings=" << found.crossings << ", return distance=" << *found.last_distance
                  << ", estimated T=" << found.candidate->period << '\n';
        comparison cmp( space, map, rhs.lambda, request, { 1e-6, 1e-6, 1e-6, 1e-5 } );
        auto one = trace( space, *found.candidate, 1, directory, cmp );
        auto four = trace( space, *found.candidate, 4, directory, cmp );
        require( cmp.compare( one, four ).relation == periodic_orbit::orbit_relation::same,
                 "Independent verified M=1 and M=4 corrections did not match" );
        require( guarded_memory::host_bytes == start_bytes, "Seed search or branch performed host transfers" );
    } );
}
