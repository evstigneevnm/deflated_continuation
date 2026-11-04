#include "section_test_support.h"
using namespace section_tests;

// Rotations commute with harmonic flow. A flight of duration T<2*pi
// closes against g(T)*x0, not x0. Supply the transformed final plane;
// Stage 1 neither estimates the drift nor solves a relative-periodic orbit.
void test()
{
    auto ops = std::make_shared<operations>( 2 );
    auto u   = buffer( ops );
    fill( *ops, *u, 1, 0 );
    auto s      = make_step<harmonic>( ops, .025 );
    auto rate   = std::make_shared<rate_provider<harmonic>>();
    auto p      = std::make_shared<policy>( ops );
    auto metric = std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( ops );
    periodic_orbit::automatic_layout<operations, step<harmonic>, rate_provider<harmonic>> builder(
        ops, s, rate, p, metric
    );
    const double duration = 1.8;
    const auto   target =
        make_plane( ops, std::cos( duration ), std::sin( duration ), -std::sin( duration ), std::cos( duration ) );
    periodic_orbit::placement_params<double> params;
    params.sections   = 4;
    const auto result = builder.build( *u, { .3, .3 + duration, target }, params );
    require( result.succeeded(), "Transformed closing target layout failed" );
    require( result.layout->closing_target() == target, "Relative closing target was lost" );
    require( std::abs( result.flight.time - ( .3 + duration ) ) < 1e-9, "Relative closing flight time" );
    const auto ordinary = builder.build( *u, { .3, .3 + duration, nullptr }, params );
    require( !ordinary.succeeded(), "Nonperiodic pilot accepted without transformed closure" );
}
int main()
{
    return run( test );
}
