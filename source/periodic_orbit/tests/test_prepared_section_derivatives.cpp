#include "prepared_cycle_support.h"
using namespace cycle_tests;

// A quarter-cycle flight has analytic radial derivative exp(-pi). An initial
// angular displacement only changes arrival time: dTau/dy0=-1, DP*(0,1)=0.
void check_flight( const std::string &method, std::size_t dimension = 2 )
{
    auto ops     = std::make_shared<operations>( dimension );
    auto initial = buffer( ops ), arrival = buffer( ops ), direction = buffer( ops ), image = buffer( ops ),
         plus = buffer( ops ), minus = buffer( ops ), finite_difference = buffer( ops );
    fill( *ops, *initial, 1, 0 );
    auto native = std::make_shared<step<analytic_cycle>>(
        ops, std::make_shared<problem<analytic_cycle>>( problem<analytic_cycle>{ ops, {} } ),
        std::make_shared<fixed>( fixed::params{ .11 } ), step<analytic_cycle>::params{ method }
    );
    recorded_map               map( ops, native, std::make_shared<rate_provider<analytic_cycle>>() );
    recorded_map::request_type request;
    request.start_time                  = .4;
    request.end_time                    = 3;
    request.target                      = make_plane( ops, 0, 0, 1, 0 );
    request.rule.direction              = periodic_orbit::crossing_direction::decreasing;
    request.rule.minimum_crossing_speed = .1;
    const auto report                   = map.evaluate( *initial, request, *arrival );
    require( report.succeeded(), "Prepared quarter-cycle flight" );
    auto rate            = std::make_shared<cycle_rate>();
    using replay         = periodic_orbit::explicit_flight_replay<operations, cycle_rate>;
    using prepared       = periodic_orbit::prepared_explicit_section<operations, cycle_rate, plane>;
    auto       workspace = std::make_shared<replay>( ops, rate, map.table().dense_outout_stage_count() );
    const auto schedule  = map.records();
    prepared   flight( ops, workspace, map.table(), schedule, *initial, *arrival, request.target, .1, 1e-8 );
    fill( *ops, *direction, 1, 0 );
    flight.apply( *direction, *image );
    near_state( ops, *image, 0, std::exp( -pi ), 2e-6 );
    fill( *ops, *direction, 0, 1 );
    flight.apply( *direction, *image );
    require(
        ops->norm( *image ) < 2e-6 && std::abs( flight.last_time_increment() + 1 ) < 2e-6,
        "Missing dense-output event-time correction"
    );
    fill( *ops, *direction, .3, -.4 );
    flight.apply( *direction, *image );
    const double epsilon = 1e-5;
    ops->assign_lin_comb( 1, *initial, epsilon, *direction, *plus );
    require( map.evaluate( *plus, request, *plus ).succeeded(), "Positive finite-difference flight" );
    ops->assign_lin_comb( 1, *initial, -epsilon, *direction, *minus );
    require( map.evaluate( *minus, request, *minus ).succeeded(), "Negative finite-difference flight" );
    ops->assign_lin_comb( .5 / epsilon, *plus, -.5 / epsilon, *minus, *finite_difference );
    ops->add_lin_comb( -1, *image, *finite_difference );
    require(
        ops->norm( *finite_difference ) < 3e-6, "Prepared derivative disagrees with event-map finite differences"
    );
    flight.apply( *direction, *plus );
    ops->add_lin_comb( -1, *image, *plus );
    require( ops->norm( *plus ) < 1e-13, "Trial flight overwrote prepared data" );
    require(
        flight.records().size() == schedule.size() && flight.records().back().theta == schedule.back().theta,
        "Prepared mesh changed during Jv"
    );
    ops->assign( *direction, *plus );
    flight.apply( *plus, *plus );
    ops->add_lin_comb( -1, *image, *plus );
    require( ops->norm( *plus ) < 1e-13, "Prepared derivative aliasing" );
    rate->fail_jacobian = true;
    ops->assign_scalar( 7, *plus );
    const auto norm       = ops->norm( *plus );
    const auto saved_time = flight.last_time_increment();
    require_throws<std::runtime_error>( [&] { flight.apply( *direction, *plus ); }, "Tangent failure ignored" );
    require( ops->norm( *plus ) == norm, "Failed tangent published a partial output" );
    require( flight.last_time_increment() == saved_time, "Failed tangent published a flight-time increment" );
    rate->fail_jacobian = false;
    const auto calls    = rate->rate_calls;
    require_throws<std::invalid_argument>(
        [&] {
            prepared unsupported(
                ops, workspace, nmfd::time_steppers::runge_kutta::make_butcher_table( "RK4" ), schedule, *initial,
                *arrival, request.target, .1, 1e-8
            );
        },
        "Hermite replay was silently inferred"
    );
    require( calls == rate->rate_calls, "Unsupported replay evaluated the RHS" );
    auto invalid_schedule         = schedule;
    invalid_schedule.back().theta = 0;
    require_throws<std::invalid_argument>(
        [&] {
            prepared invalid(
                ops, workspace, map.table(), invalid_schedule, *initial, *arrival, request.target, .1, 1e-8
            );
        },
        "Invalid frozen mesh accepted"
    );
    require_throws<std::runtime_error>(
        [&] {
            prepared grazing(
                ops, workspace, map.table(), schedule, *initial, *arrival, request.target, 2 * ops->norm( *arrival ),
                1e-8
            );
        },
        "Grazing dense derivative accepted"
    );
}

// Time-dependent rotation theta'=t checks stage times and adaptive recording.
// For a quarter-turn starting at t0, tau=sqrt(t0^2+pi)-t0 and dTau/dy0=-1/(t0+tau).
struct time_rotation
{
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double t ) const
    {
        return t * ( i % 2 ? u[i - 1] : -u[i + 1] );
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *, const double *v, double t ) const
    {
        return rhs( i, v, t );
    }
};
void check_adaptive_time()
{
    auto ops     = std::make_shared<operations>( 2 );
    auto initial = buffer( ops ), arrival = buffer( ops ), v = buffer( ops ), out = buffer( ops );
    fill( *ops, *initial, 1, 0 );
    fill( *ops, *v, 0, 1 );
    using adaptive_step = step<time_rotation, adaptive>;
    adaptive::params p;
    p.initial_step       = .1;
    p.maximum_step       = .3;
    p.relative_tolerance = p.absolute_tolerance = 1e-10;
    auto native                                 = std::make_shared<adaptive_step>(
        ops, std::make_shared<problem<time_rotation>>( problem<time_rotation>{ ops, {} } ),
        std::make_shared<adaptive>( ops, p ), adaptive_step::params{ "RK45" }
    );
    using map_type =
        periodic_orbit::recorded_section_map<operations, adaptive_step, plane, rate_provider<time_rotation>>;
    map_type               map( ops, native, std::make_shared<rate_provider<time_rotation>>() );
    map_type::request_type r;
    r.start_time                  = .4;
    r.end_time                    = 3;
    r.target                      = make_plane( ops, 0, 0, 1, 0 );
    r.rule.direction              = periodic_orbit::crossing_direction::decreasing;
    r.rule.minimum_crossing_speed = .1;
    require( map.evaluate( *initial, r, *arrival ).succeeded(), "Adaptive non-autonomous flight" );
    require(
        map.records().size() > 2 && std::any_of(
                                        map.records().begin(), map.records().end(),
                                        [&]( const auto &s ) { return s.dt != map.records().front().dt; }
                                    ),
        "Adaptive mesh was not recorded"
    );
    using rate_type = tangent_rate<time_rotation>;
    auto rate       = std::make_shared<rate_type>();
    using replay    = periodic_orbit::explicit_flight_replay<operations, rate_type>;
    auto work       = std::make_shared<replay>( ops, rate, map.table().dense_outout_stage_count() );
    periodic_orbit::prepared_explicit_section<operations, rate_type, plane> flow(
        ops, work, map.table(), map.records(), *initial, *arrival, r.target, .1, 1e-8
    );
    flow.apply( *v, *out );
    require( ops->norm( *out ) < 1e-7, "Non-autonomous event derivative" );
    require(
        std::abs( flow.last_time_increment() + 1 / std::sqrt( .16 + pi ) ) < 1e-7,
        "Non-autonomous event-time sensitivity"
    );
}
void test()
{
    check_flight( "RK45" );
    check_flight( "DOP853" );
    check_flight( "DOP853", 16384 );
    check_adaptive_time();
    require( guarded_memory::host_bytes == 0, "Prepared derivatives transferred vector data" );
}
int main()
{
    return run( test );
}
