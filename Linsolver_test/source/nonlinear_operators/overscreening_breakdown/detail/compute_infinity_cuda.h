#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_DETAIL_COMPUTE_INFINITY_CUDA_H__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_DETAIL_COMPUTE_INFINITY_CUDA_H__

#include <utils/device_tag.h>

namespace detail
{


    /*
        0x7f800000 = infinity

        0xff800000 = -infinity

        0x7ff0000000000000 = infinity

        0xfff0000000000000 = -infinity
     */
    // infinit spec



template <class T>
__DEVICE_TAG__ inline T compute_infinity()
{}

template <>
__DEVICE_TAG__ inline float compute_infinity()
{
    return 0x7f800000;
}
template <>
__DEVICE_TAG__ inline double compute_infinity()
{
    return 0x7ff0000000000000;
}


}

#endif 