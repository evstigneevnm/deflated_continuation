#ifndef PERIODIC_ORBIT_TESTS_KS1D_ORBIT_SUPPORT_H
#define PERIODIC_ORBIT_TESTS_KS1D_ORBIT_SUPPORT_H
#include "../ode_orbit_support.h"
#include "ks1d_evolution.h"
#if defined( TEST_VECTOR_BACKEND_CUDA )
#    include <external_libraries/fft_facade_cufft.h>
using fft = external_libraries::fft::cufft_backend;
#elif defined( TEST_VECTOR_BACKEND_HIP )
#    include <external_libraries/fft_facade_hipfft.h>
using fft = external_libraries::fft::hipfft_backend;
#else
#    include <external_libraries/fft_facade_fftw.h>
using fft = external_libraries::fft::fftw_backend;
#endif

namespace ks1d_orbit
{
using namespace ode_orbit_tests;
using rate     = ks1d_orbit::evolution<operations, fft>;
using rk       = nmfd::time_steppers::runge_kutta::explicit_time_step<operations, rate, fixed, true>;
using map      = periodic_orbit::recorded_section_map<operations, rk, plane, rate>;
using matching = periodic_orbit::multiple_section_problem<operations, operations, map, rate>;

inline auto stepper( const std::shared_ptr<operations> &space, const std::shared_ptr<rate> &rhs, double dt )
{
    return std::make_shared<rk>( space, rhs, std::make_shared<fixed>( fixed::params{ dt } ), rk::params{ "RK45" } );
}

// Seed of the N=24 odd, 2*pi-periodic semidiscretization, in unnormalized
// imaginary R2C coefficients. Independent LSODA integration, sampled at a
// maximum of mode 1. This is a seed, not a spatially converged PDE reference.
constexpr std::array<double, 11> seed{ 3.4713118913,  -3.0269991860, 52.9326530721, 4.7307271608,
                                       -3.2242998383, 15.8220336620, 1.7723260826,  -.7814574934,
                                       1.8946420033,  .3659630652,   -.1943024199 };
constexpr double                 period = .015153690425, dt = .00002;
constexpr std::size_t            budget = 4096;

inline auto provider( const std::shared_ptr<operations> &space, double lambda = 68. )
{
    return std::make_shared<rate>( space, 24, 2., 4., lambda );
}

inline auto make_layout(
    const std::shared_ptr<operations> &space, const std::shared_ptr<rate> &trial, const X &initial, double pilot_period,
    std::size_t count, double h, double start, std::size_t maximum_steps
)
{
    periodic_orbit::automatic_layout<operations, rk, rate> builder(
        space, stepper( space, trial, h ), trial, std::make_shared<policy>( space ),
        std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( space )
    );
    periodic_orbit::placement_params<double> p;
    p.sections                  = count;
    p.maximum_pilot_steps       = maximum_steps;
    p.maximum_endpoint_distance = 1e-5;
    const auto pilot            = builder.build( initial, { start, start + pilot_period, nullptr }, p );
    require( pilot.succeeded(), "KS pilot layout/itinerary failed" );

    return pilot.layout;
}

inline auto make_segments(
    const std::shared_ptr<operations> &space, const std::shared_ptr<operations> &coords,
    const std::shared_ptr<const matching::layout_type> &layout, const std::shared_ptr<rate> &trial, double h,
    std::size_t maximum_steps
)
{
    std::vector<matching::segment> segments;
    map::map_type::params          flight;
    flight.maximum_steps = maximum_steps;
    for ( std::size_t i = 0; i < layout->size(); ++i )
        segments.push_back(
            { std::make_shared<map>( space, stepper( space, trial, h ), trial, flight ),
              std::make_shared<chart>( space, coords, layout->at( i ).section ) }
        );
    return segments;
}
}
#endif
