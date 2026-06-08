#ifndef __VECTOR_SPACE_H__
#define __VECTOR_SPACE_H__

#include <utility>

#include <scfd/utils/device_tag.h>

/************************************************************
 * Backend independent kernels implementing vector operations
 * to run on heterogenous devices.
 ************************************************************/

namespace kernels
{

template <class IdxND, class Scalar, class VectorType, class ArrayNdType, int TensorDim>
struct shur_prod
{
    VectorType x, y;
    ArrayNdType z;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        Scalar local_sum = Scalar(0);
        for(int i = 0; i < TensorDim; i++)
        {
            local_sum += x(idx, i) * y(idx, i);
        }
        z(idx) = local_sum;
    }
};

template <class IdxND, class Scalar, class VectorType, class ArrayNdType, int TensorDim>
struct sum
{
    VectorType x;
    ArrayNdType z;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        Scalar local_sum = Scalar(0);
        for(int i = 0; i < TensorDim; i++)
        {
            local_sum += x(idx, i);
        }
        z(idx) = local_sum;
    }
};

template <class IdxND, class Scalar, class VectorType, int TensorDim>
struct assign_scalar
{
    Scalar      s;
    VectorType  x;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        for(int i = 0; i < TensorDim; i++)
        {
            x(idx, i) = s;
        }
    }
};

template <class IdxND, class Scalar, class VectorType, int TensorDim>
struct add_mul_scalar
{
    Scalar s, mul_x;
    VectorType    x;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        for(int i = 0; i < TensorDim; i++)
        {
            x(idx, i) = mul_x * x(idx, i) + s;
        }
    }
};


template <class IdxND, class Scalar, class VectorType, int TensorDim>
struct scale
{
    Scalar     s;
    VectorType x;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        for(int i = 0; i < TensorDim; i++)
        {
            x(idx, i) = s * x(idx, i);
        }
    }
};

template <class IdxND, class Scalar, class VectorType, int TensorDim>
struct assign
{
    VectorType x, y;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        for(int i = 0; i < TensorDim; i++)
        {
            y(idx, i) = x(idx, i);
        }
    }
};

template <class IdxND, class Scalar, class VectorType, int TensorDim>
struct assign_lin_1_comb
{
    Scalar    mul_x;
    VectorType x, y;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        for(int i = 0; i < TensorDim; i++)
        {
            y(idx, i) = mul_x * x(idx, i);
        }
    }
};

template <class IdxND, class Scalar, class VectorType, int TensorDim>
struct assign_lin_2_comb
{
    Scalar    mul_x, mul_y;
    VectorType x, y, z;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        for(int i = 0; i < TensorDim; i++)
        {
            z(idx, i) = mul_x * x(idx, i) + mul_y * y(idx, i);
        }
    }
};

template <class IdxND, class Scalar, class VectorType, int TensorDim>
struct add_lin_comb
{
    Scalar    mul_x;
    Scalar    mul_y;
    VectorType x, y;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        for(int i = 0; i < TensorDim; i++)
        {
            y(idx, i) = mul_x * x(idx, i) + mul_y * y(idx, i);
        }
    }
};

}// namespace kernels

#endif
