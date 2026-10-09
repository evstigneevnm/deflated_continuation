#ifndef NMFD_TIME_STEPPERS_TESTS_EXPLICIT_LORENZ_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_EXPLICIT_LORENZ_PROBLEM_H

#include <cstddef>
#include <scfd/utils/device_tag.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
template<class VectorOperations>
struct explicit_lorenz_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    // epsilon controls cubic damping in the spontaneous relaminarization model.
    // Setting epsilon to zero recovers the classical Lorenz system.
    scalar_type sigma = 10, rho = 28, beta = scalar_type(8) / 3, epsilon = 0*scalar_type(.0055L), delta = 0;

    struct mapping
    {
        vector_type in, out;
        scalar_type sigma, rho, beta, epsilon, delta;

        __DEVICE_TAG__ void operator()(std::ptrdiff_t) const
        {
            out(0) = -sigma * in(0) + sigma * in(1) - epsilon * in(0) * in(0) * in(0);
            out(1) = rho * in(0) - in(1) - in(0) * in(2) + delta;
            out(2) = -beta * in(2) + in(0) * in(1) - delta;
        }
    };

    void apply(const vector_type& in, vector_type& out) const
    {
        const mapping map{in, out, sigma, rho, beta, epsilon, delta};
        if constexpr (VectorOperations::memory_type::is_host_visible)
        {
            map(0);
        }
        else
        {
            // One complete small-system mapping, no per-component branches.
            typename VectorOperations::for_each_type execute;
            execute(map, 1);
            execute.wait();
        }
    }
};
}
}
}

#endif
