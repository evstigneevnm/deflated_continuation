#ifndef PERIODIC_ORBIT_VAIDYANATHAN_PERIODIC_PROBLEM_H
#define PERIODIC_ORBIT_VAIDYANATHAN_PERIODIC_PROBLEM_H

#include <array>
#include <cstddef>
#include <string>
#include <scfd/utils/device_tag.h>

namespace ode_orbit_tests
{
// Five-dimensional Vaidyanathan system from the local PO_tests formulation:
// x'=a*(y-x)+w+v, y'=c*x-x*z-y, z'=x*y-b*z,
// w'=-x*z+p*w, v'=q*x; a=10, b=8/3, c=28, p=1.3, q=2.5.
// A chaotic-trajectory recurrence was refined by phase-constrained shooting
// with SciPy DOP853 (rtol=atol=3e-14). Independent closure is below 2.1e-13.
// The non-equilibrium orbit has unstable multipliers 0.795584+/-1.975593*i
// and 1.009622. Its phase has one increasing flow-normal return per period.
// This is seeded orbit correction, not general chaotic-orbit discovery.
struct vaidyanathan_periodic
{
    static constexpr const char                        *name = "vaidyanathan_periodic";
    static constexpr double                             a = 10, b = 8. / 3, c = 28, p = 1.3, q = 2.5;
    inline static constexpr std::array<const char *, 5> parameter_names{ "a", "b", "c", "p", "q" };
    inline static constexpr std::array<double, 5>       parameters{ a, b, c, p, q };
    static constexpr double                             period = 2.4064400379694417;
    inline static constexpr std::array<double, 5>       seed{
        1.6056697979945038, -5.404348869605473, 24.3673777180209, 112.34762925394274, -3.4191430134752285
    };
    static double default_step( const std::string &method )
    {
        return method == "RK45" ? .002 : .01;
    }
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        const auto j = i - i % 5;
        switch ( i % 5 )
        {
        case 0:
            return a * ( u[j + 1] - u[j] ) + u[j + 3] + u[j + 4];
        case 1:
            return c * u[j] - u[j] * u[j + 2] - u[j + 1];
        case 2:
            return u[j] * u[j + 1] - b * u[j + 2];
        case 3:
            return -u[j] * u[j + 2] + p * u[j + 3];
        default:
            return q * u[j];
        }
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double ) const
    {
        const auto j = i - i % 5;
        switch ( i % 5 )
        {
        case 0:
            return a * ( v[j + 1] - v[j] ) + v[j + 3] + v[j + 4];
        case 1:
            return ( c - u[j + 2] ) * v[j] - u[j] * v[j + 2] - v[j + 1];
        case 2:
            return u[j + 1] * v[j] + u[j] * v[j + 1] - b * v[j + 2];
        case 3:
            return -u[j + 2] * v[j] - u[j] * v[j + 2] + p * v[j + 3];
        default:
            return q * v[j];
        }
    }
};
}
#endif
