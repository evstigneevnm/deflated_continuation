#ifndef __KURAMOTO_SIVASHINSKIY_2D_KER_H__
#define __KURAMOTO_SIVASHINSKIY_2D_KER_H__

#include <thrust/complex.h>
#include <macros.h>


template<typename T, typename T_C>
extern void gradient_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C *gradient_x, T_C *gradient_y);

template<typename T, typename T_C>
extern void Laplace_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C *gradient_x, T_C *gradient_y, T* Laplce);

template<typename T, typename T_C>
extern void biharmonic_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C *gradient_x, T_C *gradient_y, T *biharmonic);


#endif