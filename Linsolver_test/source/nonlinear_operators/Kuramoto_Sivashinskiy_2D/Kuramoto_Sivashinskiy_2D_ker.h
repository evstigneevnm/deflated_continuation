#ifndef __KURAMOTO_SIVASHINSKIY_2D_KER_H__
#define __KURAMOTO_SIVASHINSKIY_2D_KER_H__

#include <thrust/complex.h>
#include <common/macros.h>


template<typename T_C>
extern void gradient_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y);

template<typename T_C>
extern void Laplace_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y, T_C*& Laplce);

template<typename T_C>
extern void biharmonic_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y, T_C*& biharmonic);

template<typename TC, typename T_vec_im, typename TC_vec>
extern void C2R_(unsigned int BLOCK_SIZE, size_t Nx, size_t My, TC_vec& arrayC, T_vec_im& arrayR_im);

template<typename TC, typename T_vec_im, typename TC_vec>
extern void R2C_(unsigned int BLOCK_SIZE, size_t Nx, size_t My, T_vec_im& arrayR_im, TC_vec& arrayC);


#endif