#include "benchmarks/branch_archive_trace.h"
#include "benchmarks/branch_restart_case.h"
#include "benchmarks/ode_continuation_fixture.h"
#include "parameterized_cycle.h"
#include <periodic_orbit/periodic_curve_recorder.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

namespace
{
using namespace ode_orbit_tests;
using model = cycle_tests::parameterized_cycle;
using setup = continuation_fixture<model>;
using corrections = orbit_benchmarks::branch_corrections<setup>;
using measurement = container::periodic_measurement;
using reason = container::curve_endpoint_reason;
constexpr container::curve_endpoint_reason boundaries[]{ reason::boundary_min, reason::boundary_max };
const container::branch_identity identity{ "cycle-1", "analytic_cycle", "ode:2", "lambda", setup::observable_label,
                                           "euclidean" };

// r'=(lambda-r^2)r, theta'=1+(lambda-1)/4. The nonzero orbit has
// constant physical norm sqrt(lambda), independent of section count.
void trace( const std::filesystem::path &archive, std::size_t m, bool capped )
{
    constexpr double h = .05, lo = .95, hi = 1.05;
    constexpr unsigned budget = 2000;
    setup f( {}, model::seed, model::period, m, h, 0, budget );
    // Independent assembly with the same fixed pilot; verification cannot rebind the Newton operator.
    setup check( {}, model::seed, model::period, m, h, 0, budget );
    corrections solvers( f );
    auto z = f.workspace(), residual = check.workspace();
    auto initial = buffer( check.physical );
    double p = model::anchor_parameter;
    f.coords->assign_scalar( 0, *z );
    require( solvers.ordinary.solve( f.op.get(), *z, p ), "Anchor correction failed" );
    const orbit_benchmarks::archive_trace_settings settings{ lo, hi, .02, budget, 1e-6, capped ? 1u : 30u };
    auto verify = [&]( double lambda, const setup::state &point ) {
        return orbit_benchmarks::orbit_measurement( check, point, lambda, *residual, *initial, settings );
    };
    periodic_orbit::periodic_curve_recorder recorder( identity, "RK45", m, verify );
    const auto bytes = guarded_memory::host_bytes;
    const auto result = orbit_benchmarks::drive_archive( f, solvers, recorder, *z, p, settings );
    {
        std::ofstream file( archive );
        file.exceptions( std::ios::badbit | std::ios::failbit );
        recorder.write( file, result );
        file.close();
    }
    require( guarded_memory::host_bytes == bytes, "Branch observation or scalar archive transferred vectors" );
    std::ifstream file( archive );
    const auto data = container::read_periodic_branch( file );
    require( data.at( "sections" ) == m && data.at( "observable" ) == setup::observable_label &&
             data.at( "trace_complete" ) == !capped && data.at( "all_points_verified" ) == true,
             "Wrong reopened branch metadata" );
    require( data.at( "segments" ).size() == 2, "Missing arm" );
    std::size_t index = 0;
    for ( unsigned i = 0; i < 2; ++i )
    {
        const auto &arm = data.at( "segments" ).at( i );
        const auto &points = arm.at( "points" );
        const auto end = container::to_string( capped ? reason::max_steps : boundaries[i] );
        require( arm.at( "id" ) == i + 1 && arm.at( "direction" ) == (i ? 1 : -1) &&
                 points.size() >= 2 && arm.at( "endpoint" ) == end && points.back().at( "endpoint" ) == end &&
                 points.front().at( "parameter" ) == p, "Wrong semicurve or endpoint" );
        if ( !capped )
            require( points.back().at( "parameter" ) == (i ? hi : lo), "Lost corrected boundary" );
        for ( const auto &row : points )
        {
            const double lambda = row.at( "parameter" ), radius = std::sqrt( lambda );
            require( row.at( "source_index" ) == index++ && row.at( "verified" ) == true &&
                     std::abs( row.at( "period" ).get<double>() - model::period / model{ lambda }.omega() ) < 1e-6,
                     "Incorrect archived period or source index" );
            for ( const char *key : { "observable_rms", "observable_min", "observable_max" } )
                require( std::abs( row.at( key ).get<double>() - radius ) < 1e-6, "Wrong physical observable" );
        }
    }
}

void check_fold_and_failure()
{
    // These synthetic emissions turn in parameter. Neither force_store nor a
    // failed measurement may remove or reorder rows, or create a valid height.
    auto verify = []( double, int n ) -> std::optional<measurement> {
        if ( n == 1 )
            return {};
        return measurement{ 1, 2, 2, 2, n == 2 ? std::numeric_limits<double>::infinity() : 0. };
    };
    unsigned captured = 0;
    auto capture = [&]( auto, auto, auto, auto, const auto &, const auto & ) { ++captured; };
    periodic_orbit::periodic_curve_recorder recorder( identity, "RK45", 1, verify, capture );
    recorder.start_new_segment();
    recorder.add( 1., 0, true, reason::none );
    recorder.add( 2., 1, true, reason::none );
    recorder.add( 1.5, 2, false, reason::none );
    require( captured == 1, "Snapshot hook received an unverified point" );
    recorder.set_last_endpoint_reason( reason::closed_return );
    continuation::continuation_curve_result<double> result;
    result.semicurves_started = 1;
    result.branch_closed = true;
    auto &arm = result.semicurves[0];
    arm.status = continuation::semicurve_status::complete;
    arm.segment_id = 1;
    arm.direction = 1;
    arm.accepted_points = 3;
    arm.endpoint_reason = reason::closed_return;
    std::stringstream out;
    recorder.write( out, result );
    auto data = container::read_periodic_branch( out );
    const auto &points = data.at( "segments" ).at( 0 ).at( "points" );
    require( data.at( "trace_complete" ) == true && data.at( "all_points_verified" ) == false &&
             points[0].at( "verified" ) == true && points[1].at( "parameter" ) == 2 &&
             points[2].at( "parameter" ) == 1.5 && points[2].at( "endpoint" ) == "closed_return",
             "Fold or verification status lost" );
    for ( int i : { 1, 2 } )
        require( points[i].at( "verified" ) == false && !points[i].contains( "observable_rms" ),
                 "Published a failed measurement" );
    data["version"] = 2;
    std::istringstream future( data.dump() );
    require_throws<std::invalid_argument>( [&] { container::read_periodic_branch( future ); }, "Unknown archive version" );
}
}

int main( int argc, char **argv )
{
    return section_tests::run( [&] {
        namespace fs = std::filesystem;
        if ( argc == 3 && (std::string( argv[1] ) == "--write" || std::string( argv[1] ) == "--restart") )
        {
            for ( const auto m : { 1u, 4u } )
            {
                auto id = identity;
                id.id += "-M" + std::to_string( m );
                orbit_benchmarks::branch_restart_case<setup>( fs::path( argv[2] ) / ("cycle-M" + std::to_string( m )),
                    std::string( argv[1] ) == "--restart", id, m, model::anchor_parameter, .05,
                    { .95, 1.05, .02, 2000, 1e-6 },
                    [=] { return std::make_unique<setup>( model{}, model::seed, model::period, m, .05, 0., 2000 ); } );
            }
            return;
        }
        const auto dir = argc == 2 ? fs::path( argv[1] ) : fs::temp_directory_path() /
            ("cycle-archive-" + std::to_string( std::chrono::steady_clock::now().time_since_epoch().count() ));
        if ( argc > 2 || !fs::create_directory( dir ) )
            throw std::invalid_argument( "Supply a new output directory, or no argument for a temporary run" );
        check_fold_and_failure();
        for ( const auto m : { 1u, 4u } )
            trace( dir / ("cycle-M" + std::to_string( m ) + ".json"), m, false );
        trace( dir / "cycle-capped.json", 1, true );
        if ( argc == 1 )
            fs::remove_all( dir );
    } );
}
