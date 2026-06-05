#ifndef __NMFD_KERNELS_DENSE_VECTOR_SPACE_H__
#define __NMFD_KERNELS_DENSE_VECTOR_SPACE_H__

#include "scfd/utils/scalar_traits.h"
#include <cmath>
#include <cstddef>
#include <utility>

#include <scfd/utils/device_tag.h>

/************************************************************
 * Backend independent kernels implementing vector operations
 * to run on heterogenous devices.
 ************************************************************/
namespace nmfd
{
namespace operations
{
namespace kernels
{

template <class Scalar, class VectorType>
struct scalar_prod
{
    const Scalar *x;
    const Scalar *y;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = x[idx] * y[idx];
    }
};

template <class Scalar>
struct assign_scalar
{
    Scalar  scalar;
    Scalar *x;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        x[idx] = scalar;
    }
};

template <class Scalar>
struct add_mul_scalar
{
    Scalar  scalar;
    Scalar  mul_x;
    Scalar *x;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        x[idx] = mul_x * x[idx] + scalar;
    }
};

template <class Scalar, class VectorType>
struct norm_inf
{
    const Scalar *x;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = scfd::utils::scalar_traits<Scalar>::abs( x[idx] );
    }
};

template <class Scalar, class VectorType>
struct sum
{
    const Scalar *x;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = x[idx];
    }
};

template <class Scalar, class VectorType>
struct asum
{
    const Scalar *x;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = scfd::utils::scalar_traits<Scalar>::abs( x[idx] );
    }
};

template <class Scalar>
struct assign
{
    const Scalar *x;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = x[idx];
    }
};

template <class Scalar>
struct assign_lin_comb_1
{
    Scalar        mul_x;
    const Scalar *x;
    Scalar       *y;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        y[idx] = mul_x * x[idx];
    }
};

template <class Scalar>
struct assign_lin_comb_2
{
    Scalar        mul_x;
    const Scalar *x;
    Scalar        mul_y;
    const Scalar *y;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = mul_x * x[idx] + mul_y * y[idx];
    }
};

template <class Scalar>
struct add_lin_comb_1
{
    Scalar        mul_x;
    const Scalar *x;
    Scalar       *y;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        y[idx] += mul_x * x[idx];
    }
};

template <class Scalar>
struct add_lin_comb_2
{
    Scalar        mul_x;
    const Scalar *x;
    Scalar        mul_y;
    Scalar       *y;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        y[idx] = mul_x * x[idx] + mul_y * y[idx];
    }
};

template <class Scalar>
struct add_lin_comb_3
{
    Scalar        mul_x;
    const Scalar *x;
    Scalar        mul_y;
    const Scalar *y;
    Scalar        mul_z;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = mul_x * x[idx] + mul_y * y[idx] + mul_z * z[idx];
    }
};

template <class Scalar>
struct make_abs_copy
{
    const Scalar *x;
    Scalar       *y;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        y[idx] = scfd::utils::scalar_traits<Scalar>::abs( x[idx] );
    }
};

template <class Scalar>
struct make_abs
{
    Scalar *x;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        x[idx] = scfd::utils::scalar_traits<Scalar>::abs( x[idx] );
    }
};

template <class Scalar>
struct max_pointwise
{
    Scalar        sc;
    const Scalar *x;
    Scalar       *y;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        y[idx] = ( x[idx] > y[idx] ) ? ( ( x[idx] > sc ) ? x[idx] : sc ) : ( ( y[idx] > sc ) ? y[idx] : sc );
    }
};

template <class Scalar>
struct min_pointwise
{
    Scalar        sc;
    const Scalar *x;
    Scalar       *y;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        y[idx] = ( x[idx] < y[idx] ) ? ( ( x[idx] < sc ) ? x[idx] : sc ) : ( ( y[idx] < sc ) ? y[idx] : sc );
    }
};

template <class Scalar>
struct mul_pointwise
{
    Scalar        mul_x;
    const Scalar *x;
    Scalar        mul_y;
    const Scalar *y;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = ( mul_x * x[idx] ) * ( mul_y * y[idx] );
    }
};

template <class Scalar>
struct div_pointwise
{
    Scalar        mul_x;
    const Scalar *x;
    Scalar        mul_y;
    const Scalar *y;
    Scalar       *z;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        z[idx] = ( mul_x * x[idx] ) / ( mul_y * y[idx] );
    }
};


template <class Scalar>
struct assign_random
{
    Scalar       from;
    Scalar       range;
    Scalar      *x;
    unsigned int seed;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        unsigned int h = seed ^ static_cast<unsigned int>( idx * 2654435761u );
        h              = ( ( h >> 16 ) ^ h ) * 0x45d9f3bu;
        h              = ( ( h >> 16 ) ^ h ) * 0x45d9f3bu;
        h              = ( h >> 16 ) ^ h;
        x[idx]         = from + range * static_cast<Scalar>( h ) / static_cast<Scalar>( 0xFFFFFFFFu );
    }
};

} // namespace kernels
} // namespace operations
} // namespace nmfd

#endif
