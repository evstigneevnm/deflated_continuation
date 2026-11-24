#ifndef PERIODIC_ORBIT_TEST_TWO_CYCLE_PROBLEM_H
#define PERIODIC_ORBIT_TEST_TWO_CYCLE_PROBLEM_H

#include <array>
#include <cstddef>
#include <scfd/utils/device_tag.h>

namespace ode_orbit_tests
{
// u' = g(|u|^2)u + omega*Ju. The positive isolated radii are 1
// (repelling) and 2 (attracting); both primitive periods are 2*pi/omega.
struct two_cycles
{
    static constexpr double anchor_parameter = 1;
    inline static constexpr std::array<double, 2> seed{ 1, 0 };
    double omega = 1;
    two_cycles with_parameter( double p ) const { return { p }; }
    __DEVICE_TAG__ static double g( double s ) { return .03*(s-1)*(4-s)/(1+s); }
    __DEVICE_TAG__ double parameter_rhs( std::ptrdiff_t i, const double *u, double ) const
    { return i ? u[0] : -u[1]; }
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double t ) const
    { return g( u[0]*u[0]+u[1]*u[1] )*u[i] + omega*parameter_rhs( i, u, t ); }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double t ) const
    {
        const double s = u[0]*u[0]+u[1]*u[1], dg = .03*(-1+10/((1+s)*(1+s)));
        return g(s)*v[i] + 2*dg*(u[0]*v[0]+u[1]*v[1])*u[i] + omega*parameter_rhs( i, v, t );
    }
};
}
#endif
