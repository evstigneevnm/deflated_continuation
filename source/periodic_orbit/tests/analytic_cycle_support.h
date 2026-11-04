#ifndef PERIODIC_ORBIT_ANALYTIC_CYCLE_SUPPORT_H
#define PERIODIC_ORBIT_ANALYTIC_CYCLE_SUPPORT_H

#include "section_test_support.h"
#include <periodic_orbit/matching/multiple_section_residual.h>

namespace cycle_tests
{
using namespace section_tests;

// x'=(1-r^2)x-y, y'=(1-r^2)y+x. The isolated attracting orbit r=1
// has period 2*pi and transverse multiplier exp(-4*pi). Off the orbit,
// r(t)=r0/sqrt(r0^2+(1-r0^2)*exp(-2*t)), theta(t)=theta0+t.
struct analytic_cycle
{
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        const auto base = i - i % 2;
        const auto x = u[base], y = u[base + 1], radial = 1 - x * x - y * y;
        return i % 2 ? radial * y + x : radial * x - y;
    }
};
using chart = periodic_orbit::affine_chart<operations>;
using map = periodic_orbit::section_map<operations, step<analytic_cycle>, plane, policy, rate_provider<analytic_cycle>>;
using matching   = periodic_orbit::multiple_section_residual<operations, operations, map>;
using blocks     = matching::space_type;
using block_wrap = nmfd::detail::vector_wrap<blocks, false, false>;

inline block_wrap block_buffer( const std::shared_ptr<blocks> &space )
{
    block_wrap x( *space );
    x.init();
    x.start_use();
    return x;
}

struct fixture
{
    std::shared_ptr<operations>                  physical    = std::make_shared<operations>( 2 );
    std::shared_ptr<operations>                  coordinates = std::make_shared<operations>( 1 );
    std::shared_ptr<blocks>                      space;
    std::shared_ptr<const matching::layout_type> layout;
    std::vector<matching::segment>               segments;
    double                                       dt, start;
    std::size_t                                  maximum_steps;
    std::string                                  method;

    std::shared_ptr<step<analytic_cycle>> make_stepper( double h ) const
    {
        auto rhs = std::make_shared<problem<analytic_cycle>>( problem<analytic_cycle>{ physical, {} } );
        return std::make_shared<step<analytic_cycle>>(
            physical, rhs, std::make_shared<fixed>( fixed::params{ h } ), step<analytic_cycle>::params{ method }
        );
    }
    fixture(
        std::size_t count, double h = .08, std::string scheme = "RK45", double t = .37, std::size_t budget = 20000
    )
        : space( std::make_shared<blocks>( coordinates, count ) ), dt( h ), start( t ), maximum_steps( budget ),
          method( std::move( scheme ) )
    {
        auto initial = buffer( physical );
        fill( *physical, *initial, 1, 0 );
        auto rate         = std::make_shared<rate_provider<analytic_cycle>>();
        auto state_policy = std::make_shared<policy>( physical );
        auto metric       = std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( physical );
        periodic_orbit::automatic_layout<operations, step<analytic_cycle>, rate_provider<analytic_cycle>> builder(
            physical, make_stepper( h ), rate, state_policy, metric
        );
        periodic_orbit::placement_params<double> p;
        p.sections                  = count;
        p.maximum_pilot_steps       = maximum_steps;
        p.maximum_endpoint_distance = 1e-6;
        auto result                 = builder.build( *initial, { start, start + 2 * pi, nullptr }, p );
        if ( !result.succeeded() )
            throw std::runtime_error(
                "Analytical-cycle pilot layout failed at section " + std::to_string( result.failed_section )
            );
        layout   = std::move( result.layout );
        segments = make_segments( h );
    }
    std::vector<matching::segment> make_segments( double h ) const
    {
        std::vector<matching::segment> result;
        map::params                    p;
        p.maximum_steps = maximum_steps;
        for ( std::size_t i = 0; i < layout->size(); ++i )
            result.push_back(
                { std::make_shared<map>(
                      physical, make_stepper( h ), std::make_shared<policy>( physical ), p,
                      std::make_shared<rate_provider<analytic_cycle>>()
                  ),
                  std::make_shared<chart>( physical, coordinates, layout->at( i ).section ) }
            );
        return result;
    }
};
}
#endif
