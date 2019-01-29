#ifndef __GPU_VECTOR_OPERATIONS_KERNELS_H__
#define __GPU_VECTOR_OPERATIONS_KERNELS_H__


template<typename T>
extern void check_is_valid_number_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T *x, bool *result_d);




#endif