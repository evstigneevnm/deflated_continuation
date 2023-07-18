#ifndef __ROSSLER_OPERATOR_IMPL_CUH__
#define __ROSSLER_OPERATOR_IMPL_CUH__

#include "rossler_operator_ker.h"

template<class T, class T_vec>
__global__ void F_kernel(T_vec in_p, T a, T b, T c, T_vec out_p)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=3) return;
    out_p[0] = -in_p[1]-in_p[2];
    out_p[1] = in_p[0]+a*in_p[1];
    out_p[2] = b + in_p[2]*(in_p[0]-c);
}
template<class T, class T_vec, unsigned int BlockSize>
void nonlinear_operators::rossler_operator_ker<T, T_vec, BlockSize>::F(const T_vec& in_p, const std::tuple<T,T,T>& params, T_vec& out_p )const
{
    T a,b,c;
    std::tie(a, b, c) = params;
    F_kernel<T, T_vec><<<dimGrid, dimBlock>>>(in_p, a, b, c, out_p);
}




template<class T, class T_vec>
__global__ void set_initial_kernel(T_vec x0)
{
    x0[0] = 2.2;
    x0[1] = 0.0;
    x0[2] = 0.0;
}
template<class T, class T_vec, unsigned int BlockSize>
void nonlinear_operators::rossler_operator_ker<T, T_vec, BlockSize>::set_initial(T_vec& x0)const
{
    set_initial_kernel<T, T_vec><<<dimGrid, dimBlock>>>(x0);
}

template<class T, class T_vec>
__global__ void set_period_point_kernel(T_vec x0)
{
    x0[0] = 5.28061710527368;
    x0[1] = -7.74063775648652;
    x0[2] = 0.0785779366135108;
}
template<class T, class T_vec, unsigned int BlockSize>
void nonlinear_operators::rossler_operator_ker<T, T_vec, BlockSize>::set_period_point(T_vec& x0)const
{
    set_period_point_kernel<T, T_vec><<<dimGrid, dimBlock>>>(x0);
}


template<class T, class T_vec>
__global__ void jacobian_u_kernel(T_vec x0, T a, T b, T c, T_vec x_in_p, T_vec x_out_p)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=3) return;    
    x_out_p[0] = -x_in_p[1]-x_in_p[2];
    x_out_p[1] = x_in_p[0]+a*x_in_p[1];
    x_out_p[2] = x0[2]*x_in_p[0]+x0[0]*x_in_p[2]-c*x_in_p[2];   
}
template<class T, class T_vec, unsigned int BlockSize>
void nonlinear_operators::rossler_operator_ker<T, T_vec, BlockSize>::jacobian_u(const T_vec& x0, const std::tuple<T,T,T>& params, const T_vec& x_in_p, T_vec& x_out_p)const
{
    T a,b,c;
    std::tie(a, b, c) = params;
    jacobian_u_kernel<T, T_vec><<<dimGrid, dimBlock>>>(x0, a, b, c, x_in_p, x_out_p);
}





#endif