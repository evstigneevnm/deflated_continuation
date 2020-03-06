#ifndef __KURAMOTO_SIVASHINSKIY_2D_KER_H__
#define __KURAMOTO_SIVASHINSKIY_2D_KER_H__



template<typename T>
extern void function(dim3 dimGrid, dim3 dimBlock, size_t Nx, T R_, T *x, T lambda, T *f);

template<typename T>
extern void jacobian_x_kernel(dim3 dimGrid, dim3 dimBlock, size_t Nx,  T R_, T *x0, T lambda0, T* dx, T *df);

template<typename T_C>
extern void jacobian_lambda(dim3 dimGrid, dim3 dimBlock, size_t Nx, T R_, T *x0, T lambda0, T *dlambda);


#endif