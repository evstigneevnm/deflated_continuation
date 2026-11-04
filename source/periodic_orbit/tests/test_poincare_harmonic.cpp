#include "section_test_support.h"
using namespace section_tests;

void test()
{
    auto ops     = std::make_shared<operations>( 2 );
    auto initial = buffer( ops ), out = buffer( ops );
    fill( *ops, *initial, 1, 0 );
    auto single       = make_step<harmonic>( ops );
    auto state_policy = std::make_shared<policy>( ops );
    using map         = periodic_orbit::section_map<operations, step<harmonic>, plane, policy, rate_provider<harmonic>>;
    map flight( ops, single, state_policy, {}, std::make_shared<rate_provider<harmonic>>() );
    periodic_orbit::flight_request<double, plane> request;
    request.start_time                  = .7;
    request.end_time                    = .7 + 8;
    request.target                      = make_plane( ops, 0, 0, 1, 0 );
    request.rule.direction              = periodic_orbit::crossing_direction::decreasing;
    request.rule.minimum_crossing_speed = .5;
    auto report                         = flight.evaluate( *initial, request, *out );
    require( report.succeeded() && std::abs( report.time - ( .7 + pi / 2 ) ) < 2e-9, "Forward crossing time" );
    near_state( ops, *out, 0, 1, 2e-9 );
    require( report.crossing_speed < -.999 && single->modified == 1, "Crossing orientation/shortened commit" );
    request.end_time = .7 - 8;
    report           = flight.evaluate( *initial, request, *out );
    require(
        report.succeeded() && std::abs( report.time - ( .7 - 3 * pi / 2 ) ) < 3e-9, "Backward physical-time orientation"
    );
    near_state( ops, *out, 0, 1, 3e-9 );
    request.target         = make_plane( ops, 0, 0, 0, 1 );
    request.rule.direction = periodic_orbit::crossing_direction::increasing;
    request.end_time       = .7 + 8;
    report                 = flight.evaluate( *initial, request, *initial );
    require( report.succeeded() && std::abs( report.flight_time - 2 * pi ) < 4e-9, "Leave-and-return or aliasing" );
    near_state( ops, *initial, 1, 0, 4e-9 );
    fill( *ops, *initial, 1, 0 );
    request.rule.initial = periodic_orbit::initial_event_policy::report_initial;
    report               = flight.evaluate( *initial, request, *out );
    require(
        report.status == status::initial_event && report.steps == 0 && report.flight_time == 0,
        "Initial event reporting"
    );
    request.rule.initial             = periodic_orbit::initial_event_policy::leave_and_return;
    request.rule.minimum_flight_time = 7;
    request.end_time                 = .7 + 14;
    report                           = flight.evaluate( *initial, request, *out );
    require( report.succeeded() && std::abs( report.flight_time - 4 * pi ) < 1e-8, "Minimum flight time" );

    // Repeated independent oscillator pairs exercise a genuinely large state.
    auto large = std::make_shared<operations>( 32768 );
    auto u = buffer( large ), v = buffer( large );
    fill( *large, *u, 1, 0 );
    auto                                                           s = make_step<harmonic>( large, .1 );
    periodic_orbit::section_map<operations, step<harmonic>, plane> large_map(
        large, s, std::make_shared<policy>( large )
    );
    request                = {};
    request.end_time       = 2;
    request.target         = make_plane( large, 0, 0, 1, 0 );
    request.rule.direction = periodic_orbit::crossing_direction::decreasing;
    const auto bytes       = guarded_memory::host_bytes;
    report                 = large_map.evaluate( *u, request, *v );
    require( report.succeeded() && std::abs( report.time - pi / 2 ) < 2e-8, "Large backend crossing" );
    require( guarded_memory::host_bytes == bytes, "Section flight transferred state data to/from host" );
    near_state( large, *v, 0, 1, 2e-8 );
}
int main()
{
    return run( test );
}
