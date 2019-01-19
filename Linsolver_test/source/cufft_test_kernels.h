#include <cuda_runtime.h>
#include <cufft.h>

#ifndef __CUFFT_TEST_KERNELS_H__
#define __CUFFT_TEST_KERNELS_H__

#ifndef BLOCKSIZE
    #define BLOCKSIZE 128
#endif

template <typename T_R, typename T_C>
extern void set_values(size_t N, T_C* array);

template <typename T_R, typename T_C>
extern void check_values(size_t N, T_C* array1, T_C* array2);

template <typename T_R, typename T_C>
extern void convert_2R_values(size_t N, T_C* arrayC, T_R* arrayRR, T_R* arrayRI);

#endif