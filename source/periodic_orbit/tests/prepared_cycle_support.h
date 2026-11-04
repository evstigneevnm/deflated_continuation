#ifndef PERIODIC_ORBIT_PREPARED_CYCLE_SUPPORT_H
#define PERIODIC_ORBIT_PREPARED_CYCLE_SUPPORT_H

#include "analytic_cycle_support.h"
#include "variational_test_support.h"
#include <periodic_orbit/matching/multiple_section_problem.h>

namespace cycle_tests
{

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
