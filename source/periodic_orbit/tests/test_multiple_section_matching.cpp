#include "analytic_cycle_support.h"
using namespace cycle_tests;

// Real section flights around an isolated attracting cycle. Zero coordinates
// are pilot centers; perturbed coordinates are deliberately not orbit solutions.
void check_cycle()
{
    for ( const std::size_t count : { 1u, 2u, 4u, 8u } )
    {
        fixture  f( count );
        matching residual( f.physical, f.space, f.layout, f.segments );
        auto     z = block_buffer( f.space ), r = block_buffer( f.space ), saved = block_buffer( f.space );
        f.space->assign_scalar( 0, *z );
        const auto bytes  = guarded_memory::host_bytes;
        const auto report = residual.evaluate( *z, *r );
        require( report.succeeded(), "Cycle matching flight failed" );
        require( std::abs( report.period - 2 * pi ) < 2e-6, "Cycle matching period" );
        require( f.space->norm( *r ) < 2e-6, "Cycle matching residual" );
        for ( const auto &segment : report.segments )
        {
            require( segment.flight.flight_time > 0 && segment.flight.steps > 0, "Invalid segment duration/work" );
            require( std::abs( segment.arrival_section_value ) < 1e-8, "Arrival missed physical section" );
            require( segment.gap_norm < 2e-6, "Cycle physical gap" );
        }
        f.space->assign( *r, *saved );
        residual.apply( *z, *r );
        f.space->add_lin_comb( -1, *saved, *r );
        require( f.space->norm( *r ) < 1e-13, "Repeated matching changed the residual" );
        for ( std::size_t i = 0; i < count; ++i )
            f.coordinates->assign_scalar( .02 * ( i + 1 ) / count, ( *z ).comp( i ) );
        require( residual.evaluate( *z, *r ).succeeded(), "Perturbed section flight failed" );
        require( f.space->norm( *r ) > 1e-4, "Non-orbit perturbation reported zero matching residual" );
        f.space->assign( *r, *saved );
        residual.apply( *z, *z );
        f.space->add_lin_comb( -1, *saved, *z );
        require( f.space->norm( *z ) < 1e-13, "Matching input/output aliasing" );
        require( guarded_memory::host_bytes == bytes, "Matching used host state transfers" );
    }
}

// A controlled map P(x)=2*x on a shared plane isolates cyclic indexing and
// transactional publication from integration accuracy, including a late failure.
struct controlled_map
{
    std::shared_ptr<operations> space;
    bool        failure = false, exception = false, off_section = false, invalid = false, zero_time = false;
    std::size_t calls = 0;
    periodic_orbit::map_report<double>
    evaluate( const X &in, const periodic_orbit::flight_request<double, plane> &request, X &out )
    {
        ++calls;
        space->assign_lin_comb( 2, in, out );
        if ( exception )
            throw std::runtime_error( "Injected flight exception" );
        if ( invalid )
            space->assign_scalar( std::numeric_limits<double>::infinity(), out );
        if ( off_section )
            space->add_lin_comb( 1, request.target->normal(), out );
        periodic_orbit::map_report<double> report;
        report.status      = failure ? status::flight_limit : status::located;
        report.time        = request.start_time + 1;
        report.flight_time = 1;
        if ( zero_time )
        {
            report.status      = status::initial_event;
            report.time        = request.start_time;
            report.flight_time = 0;
        }
        report.crossing_speed = 1;
        report.steps          = 2;
        return report;
    }
};

std::weak_ptr<controlled_map> check_transactions( std::size_t dimension )
{
    using controlled = periodic_orbit::multiple_section_residual<operations, operations, controlled_map>;
    auto physical    = std::make_shared<operations>( dimension );
    auto coordinates = std::make_shared<operations>( dimension - 1 );
    auto space       = std::make_shared<blocks>( coordinates, 3 );
    auto center = buffer( physical ), normal = buffer( physical );
    physical->assign_scalar( 0, *center );
    physical->assign_scalar( 1, *normal );
    auto                                     section = std::make_shared<plane>( physical, *center, *normal );
    std::vector<matching::layout_type::node> nodes;
    std::vector<controlled::segment>         segments;
    for ( std::size_t i = 0; i < 3; ++i )
    {
        nodes.push_back( { section, double( i ), 1, i * 2, 2 } );
        segments.push_back(
            { std::make_shared<controlled_map>( controlled_map{ physical } ),
              std::make_shared<chart>( physical, coordinates, section ) }
        );
    }
    auto       layout = std::make_shared<matching::layout_type>( nodes, section );
    controlled residual( physical, space, layout, segments );
    auto       z = block_buffer( space ), out = block_buffer( space ), expected = block_buffer( space );
    for ( std::size_t i = 0; i < 3; ++i )
    {
        coordinates->assign_scalar( double( i + 1 ), ( *z ).comp( i ) );
        coordinates->assign_scalar( 2 * double( i + 1 ) - double( ( i + 1 ) % 3 + 1 ), ( *expected ).comp( i ) );
    }
    residual.apply( *z, *out );
    space->add_lin_comb( -1, *expected, *out );
    require( space->norm( *out ) < 1e-9, "Cyclic successor indexing" );
    segments[1].flight->failure = true;
    const auto last_calls       = segments[2].flight->calls;
    space->assign_scalar( 7, *out );
    const auto  saved_norm = space->norm( *out );
    const auto &failed     = residual.evaluate( *z, *out );
    require( failed.status == status::flight_limit && failed.failed_section == 1, "Lost failed-flight diagnostics" );
    require( failed.segments[0].flight.succeeded(), "Lost completed-flight diagnostics" );
    require( segments[2].flight->calls == last_calls, "Matching continued after a failed flight" );
    require( space->norm( *out ) == saved_norm, "Partial matching residual was published" );
    try
    {
        residual.apply( *z, *out );
        require( false, "Void residual silently ignored failed flight" );
    }
    catch ( const periodic_orbit::matching_evaluation_error<double> &e )
    {
        require(
            e.section == 1 && e.status == status::flight_limit && e.flight.steps == 2, "Failure exception report"
        );
    }
    const auto input_norm = space->norm( *z );
    residual.evaluate( *z, *z );
    require( space->norm( *z ) == input_norm, "Failed aliased evaluation changed input" );
    segments[1].flight->failure   = false;
    segments[1].flight->exception = true;
    require_throws<std::runtime_error>( [&] { residual.apply( *z, *out ); }, "Callback exception swallowed" );
    require( space->norm( *out ) == saved_norm, "Callback exception changed output" );
    segments[1].flight->exception   = false;
    segments[1].flight->off_section = true;
    require( !residual.evaluate( *z, *out ).succeeded(), "Restriction concealed off-section arrival" );
    require( space->norm( *out ) == saved_norm, "Off-section failure changed output" );
    segments[1].flight->off_section = false;
    segments[1].flight->invalid     = true;
    require( !residual.evaluate( *z, *out ).succeeded(), "Nonfinite arrival accepted" );
    segments[1].flight->invalid   = false;
    segments[1].flight->zero_time = true;
    require( !residual.evaluate( *z, *out ).succeeded(), "Immediate initial-event return accepted as a flight" );
    require( space->norm( *out ) == saved_norm, "Zero-duration failure changed output" );
    segments[1].flight->zero_time = false;
    typename blocks::vector_type malformed;
    require( residual.evaluate( malformed, *out ).status == status::invalid_input, "Wrong block count accepted" );
    require( space->norm( *out ) == saved_norm, "Malformed input changed output" );
    coordinates->assign_scalar( std::numeric_limits<double>::quiet_NaN(), ( *z ).comp( 1 ) );
    require( !residual.evaluate( *z, *out ).succeeded(), "Nonfinite coordinate accepted" );
    coordinates->assign_scalar( 2, ( *z ).comp( 1 ) );
    require( residual.evaluate( *z, *out ).succeeded(), "Failure poisoned repeated matching" );
    std::weak_ptr<controlled_map> owner = segments[0].flight;
    segments.clear();
    layout.reset();
    require( !owner.expired() && residual.evaluate( *z, *out ).succeeded(), "Matching lost retained dependencies" );
    auto                             different_closing = std::make_shared<plane>( physical, *center, *normal );
    auto                             relative = std::make_shared<matching::layout_type>( nodes, different_closing );
    std::vector<controlled::segment> dummy( 3 );
    require_throws<std::invalid_argument>(
        [&] { controlled wrong( physical, space, relative, dummy ); }, "Transformed closure silently treated as PO"
    );
    return owner;
}

void test()
{
    check_cycle();
    require( check_transactions( 2 ).expired(), "Matching dependencies have an ownership cycle" );
    require( check_transactions( 16384 ).expired(), "Large matching dependencies have an ownership cycle" );
    require( guarded_memory::host_bytes == 0, "Matching transferred vector data to host" );
}
int main()
{
    return run( test );
}
