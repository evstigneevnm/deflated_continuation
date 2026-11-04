#include "section_test_support.h"
using namespace section_tests;

// On the attracting unit circle, theta'=1+epsilon*cos(theta), so the
// period is 2*pi/sqrt(1-epsilon^2). Nonuniform angular speed and adaptive
// timesteps distinguish equal accepted-step work from equal physical time.
struct variable_speed
{
    double                epsilon = .7;
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        const auto   base = i - i % 2;
        const double x = u[base], y = u[base + 1], radial = 1 - x * x - y * y, speed = 1 + epsilon * x;
        return i % 2 ? radial * y + speed * x : radial * x - speed * y;
    }
};

void test()
{
    auto ops = std::make_shared<operations>( 2 );
    auto u   = buffer( ops );
    fill( *ops, *u, 1, 0 );
    using adaptive_step = step<variable_speed, adaptive>;
    adaptive::params cp;
    cp.initial_step       = .01;
    cp.maximum_step       = .4;
    cp.relative_tolerance = cp.absolute_tolerance = 1e-9;
    auto rhs = std::make_shared<problem<variable_speed>>( problem<variable_speed>{ ops, {} } );
    auto s   = std::make_shared<adaptive_step>(
        ops, rhs, std::make_shared<adaptive>( ops, cp ), adaptive_step::params{ "RK45" }
    );
    auto rate   = std::make_shared<rate_provider<variable_speed>>();
    auto p      = std::make_shared<policy>( ops );
    auto metric = std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( ops );
    periodic_orbit::automatic_layout<operations, adaptive_step, rate_provider<variable_speed>> builder(
        ops, s, rate, p, metric
    );
    const double                period = 2 * pi / std::sqrt( 1 - .7 * .7 );
    plane::evaluation_workspace work( ops );
    for ( const auto count : { 1u, 2u, 4u, 8u } )
    {
        periodic_orbit::placement_params<double> params;
        params.sections                  = count;
        params.maximum_endpoint_distance = 2e-7;
        const auto result                = builder.build( *u, { .4, .4 + period, nullptr }, params );
        if ( !result.succeeded() )
            std::cerr << "layout failure=" << int( result.status ) << " section=" << result.failed_section << '\n';
        require( result.succeeded() && result.layout->size() == count, "Variable-speed layout failed" );
        double      total = 0, min_time = period, max_time = 0;
        std::size_t smallest = result.pilot_steps, largest = 0;
        for ( std::size_t i = 0; i < count; ++i )
        {
            const auto &node = result.layout->at( i );
            total += node.expected_flight;
            smallest = std::min( smallest, node.expected_steps );
            largest  = std::max( largest, node.expected_steps );
            min_time = std::min( min_time, node.expected_flight );
            max_time = std::max( max_time, node.expected_flight );
            require( node.expected_steps >= params.minimum_steps_per_segment, "Too few steps between sections" );
            require(
                node.section->bind( work ).value( node.pilot_time, node.section->center() ) == 0,
                "Pilot center is not on section"
            );
        }
        require( std::abs( total - period ) < 1e-12 && largest - smallest <= 1, "Pilot work was not balanced" );
        if ( count >= 4 )
            require( max_time > 1.2 * min_time, "Layout incorrectly used uniform physical-time samples" );
        const auto repeated = builder.build( *u, { .4, .4 + period, nullptr }, params );
        require( repeated.succeeded() && repeated.pilot_steps == result.pilot_steps, "Pilot replay is not repeatable" );
    }
    fill( *ops, *u, 0, 0 );
    require(
        !builder.build( *u, { 0, period, nullptr }, {} ).succeeded(), "Equilibrium accepted as a transverse orbit"
    );
}
int main()
{
    return run( test );
}
