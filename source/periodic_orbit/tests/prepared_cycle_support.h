#ifndef PERIODIC_ORBIT_PREPARED_CYCLE_SUPPORT_H
#define PERIODIC_ORBIT_PREPARED_CYCLE_SUPPORT_H

#include "analytic_cycle_support.h"
#include <periodic_orbit/matching/multiple_section_problem.h>

namespace cycle_tests
{
template <class Model>
struct jacobian_kernel
{
    Model               model;
    double              time;
    const double       *state, *direction;
    double             *out;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out[i] = model.jacobian( i, state, direction, time );
    }
};

struct differentiable_cycle : analytic_cycle
{
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double ) const
    {
        const auto b = i - i % 2;
        const auto x = u[b], y = u[b + 1], a = 1 - x * x - y * y;
        return i % 2 ? ( 1 - 2 * x * y ) * v[b] + ( a - 2 * y * y ) * v[b + 1]
                     : ( a - 2 * x * x ) * v[b] + ( -1 - 2 * x * y ) * v[b + 1];
    }
};

template <class Model>
struct tangent_rate
{
    Model               model{};
    bool                fail_rate = false, fail_jacobian = false;
    mutable std::size_t rate_calls = 0, tangent_calls = 0;
    bool                evaluate( double time, const X &in, X &out ) const
    {
        ++rate_calls;
        if ( fail_rate )
            return false;
        backend::for_each_type{}( rhs_kernel<Model>{ in.raw_ptr(), out.raw_ptr(), time, model }, out.size() );
        return true;
    }
    bool apply_jacobian( double time, const X &state, const X &v, X &out ) const
    {
        ++tangent_calls;
        if ( fail_jacobian )
            return false;
        backend::for_each_type{}(
            jacobian_kernel<Model>{ model, time, state.raw_ptr(), v.raw_ptr(), out.raw_ptr() }, out.size()
        );
        return true;
    }
};
using cycle_rate = tangent_rate<differentiable_cycle>;
using recorded_map =
    periodic_orbit::recorded_section_map<operations, step<analytic_cycle>, plane, rate_provider<analytic_cycle>>;
using orbit_problem = periodic_orbit::multiple_section_problem<operations, operations, recorded_map, cycle_rate>;

inline std::vector<orbit_problem::segment> recorded_segments( const fixture &f )
{
    std::vector<orbit_problem::segment>     segments;
    typename recorded_map::map_type::params p;
    p.maximum_steps = f.maximum_steps;
    for ( std::size_t i = 0; i < f.layout->size(); ++i )
        segments.push_back(
            { std::make_shared<recorded_map>(
                  f.physical, f.make_stepper( f.dt ), std::make_shared<rate_provider<analytic_cycle>>(), p
              ),
              std::make_shared<chart>( f.physical, f.coordinates, f.layout->at( i ).section ) }
        );
    return segments;
}
}
#endif
