#ifndef PERIODIC_ORBIT_ROSSLER_PERIODIC_PROBLEM_H
#define PERIODIC_ORBIT_ROSSLER_PERIODIC_PROBLEM_H

#include <array>
#include <cstddef>
#include <string>
#include <scfd/utils/device_tag.h>

namespace ode_orbit_tests
{
// Rossler: x'=-y-z, y'=x+a*y, z'=b+z*(x-c), a=b=0.1, c=6.
// The seed selects a nontrivial periodic orbit, not an equilibrium or a general
// discovery algorithm. It was refined from the historical example's seed using
// SciPy DOP853 (rtol=3e-13, atol=3e-14) and phase-constrained shooting.
struct rossler_periodic
{
    static constexpr const char                        *name = "rossler_periodic";
    static constexpr double                             a = .1, b = .1, c = 6;
    inline static constexpr std::array<const char *, 3> parameter_names{ "a", "b", "c" };
    inline static constexpr std::array<double, 3>       parameters{ a, b, c };
    static constexpr double                             period = 6.0309863678295104;
    inline static constexpr std::array<double, 3> seed{ 5.4075195831062954, -7.955995200747136, .036837201516959343 };
    static double                                 default_step( const std::string &method )
    {
        return method == "RK45" ? .01 : .04;
    }
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        const auto j = i - i % 3;
        return i % 3 == 0 ? -u[j + 1] - u[j + 2] : i % 3 == 1 ? u[j] + a * u[j + 1] : b + u[j + 2] * ( u[j] - c );
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double ) const
    {
        const auto j = i - i % 3;
        return i % 3 == 0   ? -v[j + 1] - v[j + 2]
               : i % 3 == 1 ? v[j] + a * v[j + 1]
                            : u[j + 2] * v[j] + ( u[j] - c ) * v[j + 2];
    }
};
}
#endif
