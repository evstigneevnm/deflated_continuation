#ifndef PERIODIC_ORBIT_BENCHMARK_BRANCH_RESTART_CASE_H
#define PERIODIC_ORBIT_BENCHMARK_BRANCH_RESTART_CASE_H
#include "branch_archive_trace.h"
#include <periodic_orbit/orbit_snapshot.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace orbit_benchmarks
{
namespace archive_fs = std::filesystem;
using archive_json = nlohmann::json;

// Explicit I/O only. Keep the numerical-path guard enabled even while these
// copies use the underlying backend directly, and account for both directions.
struct checkpoint_transport
{
    std::size_t downloaded = 0, uploaded = 0;
    auto download( const section_tests::X &x )
    {
        std::vector<double> values( x.size() );
        const auto bytes = values.size() * sizeof( double );
        section_tests::guarded_memory::parent::copy_to_host( bytes, x.raw_ptr(), values.data() );
        downloaded += bytes;
        return values;
    }
    void upload( const archive_json &array, section_tests::X &x )
    {
        const auto values = array.get<std::vector<double>>();
        if ( values.size() != std::size_t( x.size() ) )
            throw std::invalid_argument( "Snapshot vector size mismatch" );
        const auto bytes = values.size() * sizeof( double );
        section_tests::guarded_memory::parent::copy_from_host( bytes, values.data(), x.raw_ptr() );
        uploaded += bytes;
    }
};

inline void save_json( const archive_fs::path &path, const archive_json &data )
{
    std::ofstream out( path );
    out.exceptions( std::ios::badbit | std::ios::failbit );
    out << data << '\n';
    out.close();
}

// A restart creates a new trace; the source trace and its endpoints are read-only.
template <class Fixture, class Pilot>
void branch_restart_case( const archive_fs::path &dir, bool restart, container::branch_identity id, std::size_t m,
                          double p, double h, archive_trace_settings settings, Pilot pilot )
{
    using namespace section_tests;
    using reason = container::curve_endpoint_reason;
    using space = typename Fixture::space;
    constexpr auto n = Fixture::dimension;
    const auto output = restart ? dir / "restarted" : dir;
    if ( archive_fs::exists( output ) )
        throw std::invalid_argument( "Refusing to replace existing trace: " + output.string() );
    const archive_json config = { { "step", h }, { "start", 0 }, { "max_flight_steps", settings.budget },
        { "parameter_bounds", {settings.lower, settings.upper} }, { "arclength_step", settings.ds },
        { "max_branch_steps", settings.max_steps }, { "closure_tolerance", settings.closure_tol } };
    auto coordinates = std::make_shared<operations>( n - 1 );
    space state_space( coordinates, m );
    nmfd::detail::vector_wrap<space, true, true> z( state_space );
    checkpoint_transport io;
    std::unique_ptr<Fixture> f;
    std::size_t segment = 0, index = 0;
    double saved_period = 0;
    archive_json source;
    const auto source_id = id.id;
    if ( restart )
    {
        std::ifstream in( dir / "branch.json" );
        source = container::read_periodic_branch( in );
        const auto expected = container::periodic_branch_manifest( id, "RK45", m );
        for ( const auto &key : { "id", "model", "discretization", "parameter", "method", "sections",
                                  "observable", "normalization" } )
            require( source.at( key ) == expected.at( key ), "Incompatible source branch" );
        const auto &ref = source.at( "interior_snapshot" );
        require( ref.at( "file" ) == "interior.json", "Unknown interior snapshot file" );
        segment = ref.at( "segment" );
        index = ref.at( "source_index" );
        const auto &row = periodic_orbit::verified_snapshot_point( source, segment, index );
        p = row.at( "parameter" );
        saved_period = row.at( "period" );
        auto phys = std::make_shared<operations>( n );
        std::ifstream snapshot( dir / "interior.json" );
        auto layout = periodic_orbit::restore_orbit_snapshot(
            snapshot, source, config, segment, index, phys, *z, [&](const auto &a, X &v) { io.upload( a, v ); } );
        f = std::make_unique<Fixture>( phys, layout, p, saved_period, h, settings.budget );
        id.id += "-restart";
    }
    else
    {
        f = pilot();
        state_space.assign_scalar( 0, *z );
    }
    // Reuse immutable geometry only. Verification has its own RK, crossing and replay workspaces.
    Fixture check( f->physical, f->layout, p, f->period, h, settings.budget );
    auto residual = check.workspace(), selected = f->workspace();
    auto initial = buffer( f->physical ), low = buffer( f->physical ), high = buffer( f->physical );
    auto verify = [&]( double lambda, const typename Fixture::state &point ) {
        return orbit_measurement( check, point, lambda, *residual, *initial, settings );
    };
    const auto guarded = guarded_memory::host_bytes;
    const auto io_before = io.uploaded + io.downloaded;
    branch_corrections<Fixture> solvers( *f );
    if ( restart )
    {
        const auto value = verify( p, *z ); // Do not repair an inconsistent checkpoint with Newton first.
        const double saved_rms = periodic_orbit::verified_snapshot_point( source, segment, index ).at( "observable_rms" );
        require( value && std::abs( value->period - saved_period ) < 1e-9 &&
                 std::abs( value->rms - saved_rms ) < settings.closure_tol, "Restored orbit failed verification" );
    }
    else
        require( solvers.ordinary.solve( f->op.get(), *z, p ), "Seed correction failed" );
    bool selected_point = false, have_low = false, have_high = false;
    auto capture = [&]( std::size_t arm, std::size_t row, double lambda, reason why,
                        const typename Fixture::state &point, const auto & ) {
        if ( !restart && !selected_point && lambda != p && why == reason::none )
        {
            f->coords->assign( point, *selected );
            segment = arm;
            index = row;
            selected_point = true;
        }
        if ( why == reason::boundary_min || why == reason::boundary_max )
        {
            (why == reason::boundary_min ? have_low : have_high) = true;
            f->first->lift( point.comp( 0 ), why == reason::boundary_min ? *low : *high );
        }
    };
    periodic_orbit::periodic_curve_recorder recorder( id, "RK45", m, verify, capture );
    if ( restart )
        recorder.set_restart_source( source_id, segment, index );
    const auto result = drive_archive( *f, solvers, recorder, *z, p, settings );
    require( result.complete() && have_low && have_high, "Missing verified branch boundaries" );
    require( guarded_memory::host_bytes == guarded && io.uploaded + io.downloaded == io_before,
             "Vector transfer during numerical verification or continuation" );
    std::stringstream stream;
    recorder.write( stream, result );
    auto branch = container::read_periodic_branch( stream );
    require( branch.at( "all_points_verified" ) == true, "Unverified orbit on branch" );
    const auto endpoint_file = dir / "endpoint_reference.json";
    const std::size_t snapshot_bytes = m * (3 * n - 1) * sizeof( double );
    if ( !restart )
    {
        require( selected_point, "No verified interior orbit to restart" );
        if ( !archive_fs::create_directory( output ) )
            throw std::runtime_error( "Cannot create new archive directory" );
        std::ofstream out( dir / "interior.json" );
        out.exceptions( std::ios::badbit | std::ios::failbit );
        periodic_orbit::write_orbit_snapshot( out, branch, segment, index, config, n, *f->layout, *selected,
                                              [&]( const X &x ) { return io.download( x ); } );
        out.close();
        require( io.downloaded == snapshot_bytes, "Unexpected snapshot export size" );
        branch["interior_snapshot"] = { { "segment", segment }, { "source_index", index }, { "file", "interior.json" } };
        // Test references only: two lifted physical states, outside the trace.
        save_json( endpoint_file, { io.download( *low ), io.download( *high ) } );
    }
    else
    {
        require( io.uploaded == snapshot_bytes && io.downloaded == 0, "Unexpected snapshot import size" );
        std::ifstream endpoints( endpoint_file );
        const auto refs = archive_json::parse( endpoints );
        for ( unsigned i = 0; i < 2; ++i )
        {
            io.upload( refs.at( i ), *initial );
            f->physical->add_lin_comb( -1., i ? *high : *low, *initial );
            const auto &a = source.at( "segments" ).at( i ).at( "points" ).back();
            const auto &b = branch.at( "segments" ).at( i ).at( "points" ).back();
            const double delta = f->physical->norm( *initial );
            require( a.at( "parameter" ) == b.at( "parameter" ) && delta < 1e-7 &&
                     std::abs( a.at( "period" ).get<double>() - b.at( "period" ).get<double>() ) < 1e-8,
                     "Restart changed the phase-fixed physical endpoint or period" );
            std::cout << id.id << " M=" << m << " lambda=" << b.at( "parameter" ) << " state difference=" << delta
                      << " closure=" << b.at( "closure_error" ) << '\n';
        }
        if ( !archive_fs::create_directory( output ) )
            throw std::runtime_error( "Cannot create new restart directory" );
    }
    save_json( output / "branch.json", branch );
    std::cout << "core host bytes=0, explicit I/O bytes: download=" << io.downloaded << " upload=" << io.uploaded << '\n';
}
}
#endif
