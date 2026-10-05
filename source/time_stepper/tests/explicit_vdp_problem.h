#ifndef NMFD_TIME_STEPPERS_TESTS_EXPLICIT_VDP_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_EXPLICIT_VDP_PROBLEM_H

#include <cstddef>
#include <scfd/utils/device_tag.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
template<class VectorOperations>
struct explicit_vdp_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    scalar_type mu = 1;
    struct mapping
    {
        vector_type in, out;
        scalar_type mu;
        __DEVICE_TAG__ void operator()(std::ptrdiff_t) const
        {
            out(0) = in(1);
            out(1) = mu*(1-in(0)*in(0))*in(1)-in(0);
        }
    };
    void apply(const vector_type& in, vector_type& out) const
    {
        const mapping map{in, out, mu};
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
