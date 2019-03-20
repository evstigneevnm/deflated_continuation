#ifndef __KURAMOTO_SIVASHINSKIY_2D_KER_H__
#define __KURAMOTO_SIVASHINSKIY_2D_KER_H__

#include <thrust/complex.h>
#include <macros.h>


template<typename T_C>
extern void gradient_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y);

template<typename T_C>
extern void Laplace_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y, T_C*& Laplce);

template<typename T_C>
extern void biharmonic_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y, T_C*& biharmonic);

template<typename T, typename T_C>
extern void C2R_(unsigned int BLOCK_SIZE, size_t Nx, size_t My, T_C*& arrayC, T*& arrayR_im);

template<typename T, typename T_C>
extern void R2C_(unsigned int BLOCK_SIZE, size_t Nx, size_t My, T*& arrayR_im, T_C*& arrayC);


#endif