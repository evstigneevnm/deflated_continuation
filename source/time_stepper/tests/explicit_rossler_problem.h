#ifndef NMFD_TIME_STEPPERS_TESTS_EXPLICIT_ROSSLER_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_EXPLICIT_ROSSLER_PROBLEM_H

#include <cstddef>
#include <scfd/utils/device_tag.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
template<class VectorOperations>
struct explicit_rossler_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    scalar_type a = scalar_type(.2L), b = scalar_type(.2L), c = scalar_type(5.7L);
    struct mapping
    {
        vector_type in, out;
        scalar_type a, b, c;
        __DEVICE_TAG__ void operator()(std::ptrdiff_t) const
        {
            out(0) = -in(1)-in(2);
            out(1) = in(0)+a*in(1);
            out(2) = b+in(2)*(in(0)-c);
        }
    };
    void apply(const vector_type& in, vector_type& out) const
    {
        const mapping map{in, out, a, b, c};
        if constexpr (VectorOperations::memory_type::is_host_visible)
            map(0);
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
