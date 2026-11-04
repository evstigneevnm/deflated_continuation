#ifndef PERIODIC_ORBIT_VARIATIONAL_TEST_SUPPORT_H
#define PERIODIC_ORBIT_VARIATIONAL_TEST_SUPPORT_H

#include "section_test_support.h"

namespace section_tests
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
}
#endif
