#ifndef PERIODIC_ORBIT_LORENZ_PERIODIC_PROBLEM_H
#define PERIODIC_ORBIT_LORENZ_PERIODIC_PROBLEM_H

#include <array>
#include <cstddef>
#include <string>
#include <scfd/utils/device_tag.h>

namespace ode_orbit_tests
{
// The historical cubic-damped Lorenz formulation, not the undamped textbook
// system: x'=sigma*(y-x)-epsilon*x^3, y'=rho*x-y-x*z+delta,
// z'=x*y-beta*z-delta. The orbit is an unstable, non-equilibrium short cycle.
// Reference obtained by SciPy DOP853 phase-constrained shooting (z0=27,
// rtol=atol=3e-14), continuing the short undamped orbit to epsilon=0.0055.
// The seed is shifted by 3/8 of a period so its flow-normal plane has exactly
// one increasing return per period; the original z0=27 plane had two.
struct lorenz_periodic
{
    static constexpr const char                        *name  = "lorenz_periodic";
    static constexpr double                             sigma = 10, rho = 28, beta = 8. / 3, epsilon = .0055, delta = 0;
    inline static constexpr std::array<const char *, 5> parameter_names{ "sigma", "rho", "beta", "epsilon", "delta" };
    inline static constexpr std::array<double, 5>       parameters{ sigma, rho, beta, epsilon, delta };
    static constexpr double                             period = 1.6443075739726336;
    inline static constexpr std::array<double, 3> seed{ 3.1769997853693805, 5.6626243224125057, 11.597900660565699 };
    static double                                 default_step( const std::string &method )
    {
        return method == "RK45" ? .002 : .01;
    }
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        const auto j = i - i % 3;
        const auto x = u[j], y = u[j + 1], z = u[j + 2];
        return i % 3 == 0   ? sigma * ( y - x ) - epsilon * x * x * x
               : i % 3 == 1 ? rho * x - y - x * z + delta
                            : x * y - beta * z - delta;
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double ) const
    {
        const auto j = i - i % 3;
        return i % 3 == 0   ? ( -sigma - 3 * epsilon * u[j] * u[j] ) * v[j] + sigma * v[j + 1]
               : i % 3 == 1 ? ( rho - u[j + 2] ) * v[j] - v[j + 1] - u[j] * v[j + 2]
                            : u[j + 1] * v[j] + u[j] * v[j + 1] - beta * v[j + 2];
    }
};
}
#endif
