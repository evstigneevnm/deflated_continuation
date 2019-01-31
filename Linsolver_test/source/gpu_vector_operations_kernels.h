#ifndef __GPU_VECTOR_OPERATIONS_KERNELS_H__
#define __GPU_VECTOR_OPERATIONS_KERNELS_H__

#include <thrust/complex.h>

template<typename T>
extern void check_is_valid_number_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T *x, bool *result_d);

template<typename T>
extern void assign_scalar_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T scalar, T *x);

template<typename T>
extern void add_mul_scalar_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T scalar, const T mul_x, T *x);

template<typename T>
extern void assign_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T mul_x, const T *x, T *y);

template<typename T>
extern void assign_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t sz, const T mul_x, const T *x, const T mul_y, const T *y, T *z);

template<typename T>
extern void add_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t sz,const T mul_x, const  T*& x, const T mul_y, T*& y);

template<typename T>
extern void add_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t sz, const T  mul_x, const T*&  x, const T mul_y, const T*& y, const T mul_z, T*& z);

template<typename T>
extern void mul_pointwise_wrap(dim3 dimGrid, dim3 dimBlock, size_t sz, const T mul_x, const T* x, const T mul_y, const T* y, T* z);


#endif