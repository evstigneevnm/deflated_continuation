#ifndef PERIODIC_ORBIT_TEST_PARAMETERIZED_CYCLE_H
#define PERIODIC_ORBIT_TEST_PARAMETERIZED_CYCLE_H
#include <array>
#include <cstddef>
#include <scfd/utils/device_tag.h>
namespace cycle_tests
{
// r'=(lambda-r^2)r, angle'=omega. At r=sqrt(lambda) the transverse
// multiplier is exp(-2*lambda*T), with T=2*pi/omega.
struct parameterized_cycle
{
    static constexpr const char *name = "analytic_cycle";
    static constexpr double anchor_parameter = 1, period = 6.2831853071795864769;
    inline static constexpr std::array<double, 2> seed{ 1., 0. };
    double lambda = anchor_parameter;
    parameterized_cycle with_parameter( double p ) const { return { p }; }
    __DEVICE_TAG__ double parameter_rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return u[i] + .25 * ( i ? u[0] : -u[1] );
    }
    __DEVICE_TAG__ double omega() const
    {
        return 1 + .25 * ( lambda - 1 );
    }
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return ( lambda - u[0] * u[0] - u[1] * u[1] ) * u[i] + omega() * ( i ? u[0] : -u[1] );
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double ) const
    {
        return ( lambda - u[0] * u[0] - u[1] * u[1] ) * v[i] - 2 * u[i] * ( u[0] * v[0] + u[1] * v[1] ) +
               omega() * ( i ? v[0] : -v[1] );
    }
};
}
#endif
