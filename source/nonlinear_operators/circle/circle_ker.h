#ifndef __CIRCLE_TEST_KER_H__
#define __CIRCLE_TEST_KER_H__



template<typename T>
extern void function(dim3 dimGrid, dim3 dimBlock, size_t Nx, const T R_, const T*& x, const T lambda, T*& f);

template<typename T>
extern void jacobian_x(dim3 dimGrid, dim3 dimBlock, size_t Nx,  const T R_, const T*& x0, const T lambda0, const T*& dx, T*& df);

template<typename T>
extern void jacobian_lambda(dim3 dimGrid, dim3 dimBlock, size_t Nx, const T R_, const T*& x0, const T lambda0, T*& dlambda);


#endif