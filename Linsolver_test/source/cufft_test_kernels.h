#include <cuda_runtime.h>
#include <cufft.h>
#include "macros.h"
#include <thrust/complex.h>

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

template<typename T_C>
extern void gradient_Fourier(size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y);

template<typename T_C>
extern void gradient_Fourier(size_t Nx, size_t My, size_t Ny, T_C*& gradient_x, T_C*& gradient_y);

template <typename T_R, typename T_C>
extern void complexreal_to_real(size_t N, const T_C*& arrayC, T_R*& arrayRR);


#endif