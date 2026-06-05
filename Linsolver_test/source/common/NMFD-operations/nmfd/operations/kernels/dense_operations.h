#ifndef __NMFD_KERNELS_DENSE_OPERATIONS_H__
#define __NMFD_KERNELS_DENSE_OPERATIONS_H__

#include <scfd/utils/device_tag.h>

namespace nmfd
{
namespace operations
{
namespace kernels
{

template <class MatrixType>
struct matrix_transpose_2d
{
    const MatrixType src;
    MatrixType       dst;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        dst( idx[1], idx[0] ) = src( idx[0], idx[1] );
    }
};

template <class Scalar, class MatrixType>
struct matrix_assign_scalar_2d
{
    const Scalar   value;
    MatrixType     dst;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        dst( idx ) = value;
    }
};

template <class MatrixType>
struct matrix_assign_2d
{
    const MatrixType src;
    MatrixType       dst;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        dst( idx ) = src( idx );
    }
};

template <class Scalar, class MatrixType>
struct matrix_sum_2d
{
    const Scalar     alpha;
    const MatrixType a;
    const Scalar     beta;
    const MatrixType b;
    MatrixType       c;
    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        c( idx ) = alpha * a( idx ) + beta * b( idx );
    }
};

template <class MatrixType>
struct matrix_sq_2d
{
    const MatrixType src;
    MatrixType       dst;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        dst( idx ) = src( idx ) * src( idx );
    }
};

template <class Scalar, class MatrixType>
struct matrix_extract_diag_2d
{
    const MatrixType src;
    MatrixType       dst;
    const bool       invert;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        const auto i = idx[0];
        const auto j = idx[1];
        if ( i == j )
        {
            const auto d = src( i, j );
            dst( idx )   = invert ? ( Scalar{ 1 } / d ) : d;
        }
        else
        {
            dst( idx ) = Scalar{ 0 };
        }
    }
};

template <class Scalar, class MatrixType>
struct matrix_diag_from_vec_2d
{
    const Scalar *const vec;
    MatrixType          dst;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        const auto i = idx[0];
        const auto j = idx[1];
        dst( idx )   = ( i == j ) ? vec[i] : Scalar{ 0 };
    }
};

template <class Scalar, class MatrixType>
struct matrix_scalar_diag_2d
{
    const Scalar val;
    MatrixType   dst;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        const auto i = idx[0];
        const auto j = idx[1];
        dst( idx )   = ( i == j ) ? val : Scalar{ 0 };
    }
};

template <class Scalar, class MatrixType>
struct matrix_diag_extract
{
    const MatrixType mat;
    Scalar *const    vec;
    const bool       invert;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        const Scalar d = mat( idx, idx );
        vec[idx]       = invert ? ( Scalar{ 1 } / d ) : d;
    }
};

} // namespace kernels
} // namespace operations
} // namespace nmfd

#endif
