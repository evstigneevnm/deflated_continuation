#include <nonlinear_operators/circle/circle_ker.h>


template<typename T>
__global__ void function_kernel(int Nx, T R_, T *x, T lambda, T *f)
{

    int j=blockDim.x * blockIdx.x + threadIdx.x;
    
    if(j>=Nx) return;

    f[0] = x[0]*x[0]+lambda*lambda-R_*R_;

}


template<typename T>
__global__ void jacobian_x_kernel(int Nx, T R_, T *x0, T lambda0, T* dx, T *df)
{

    int j=blockDim.x * blockIdx.x + threadIdx.x;
    
    if(j>=Nx) return;
    // [d(x^2+lambda^2-R^2)/dx]*dx = 2x0*dx
    df[0] = T(2)*x0[0]*dx[0];

}

template<typename T>
__global__ void jacobian_lambda_kernel(int Nx, T R_, T *x0, T lambda0, T *dlambda)
{

    int j=blockDim.x * blockIdx.x + threadIdx.x;
    
    if(j>=Nx) return;

    dlambda[0] = T(2)*lambda0;

}

template<typename T>
void function(dim3 dimGrid, dim3 dimBlock, size_t Nx, const T R_, const T*& x, const T lambda, T* &f)
{
    function_kernel<T><<<dimGrid, dimBlock>>>(Nx, (T)R_,  (T*&)x, (T)lambda, f);
}


template<typename T>
void jacobian_x(dim3 dimGrid, dim3 dimBlock, size_t Nx,  const T R_, const T*& x0, const T lambda0, const T*& dx, T*& df)
{
    jacobian_x_kernel<T><<<dimGrid, dimBlock>>>(Nx, R_, (T*&) x0, (T) lambda0, (T*&) dx, (T*) df);
}

template<typename T>
void jacobian_lambda(dim3 dimGrid, dim3 dimBlock, size_t Nx, const T R_, const T*& x0, const T lambda0, T*& dlambda)
{
    jacobian_lambda_kernel<T><<<dimGrid, dimBlock>>>(Nx, R_, (T*&) x0, lambda0, dlambda);
}



//explicit instantiation
template void function<float>(dim3 dimGrid, dim3 dimBlock, size_t Nx, const float R_, const float*& x, const float lambda, float*& f);
template void jacobian_x<float>(dim3 dimGrid, dim3 dimBlock, size_t Nx, const float R_, const float*& x0, const float lambda0, const float*& dx, float*& df);
template void jacobian_lambda<float>(dim3 dimGrid, dim3 dimBlock, size_t Nx, const float R_, const float*& x0, const float lambda0, float*& dlambda);

template void function<double>(dim3 dimGrid, dim3 dimBlock, size_t Nx, const double R_, const double*& x, const double lambda, double*& f);
template void jacobian_x<double>(dim3 dimGrid, dim3 dimBlock, size_t Nx, const double R_, const double*& x0, const double lambda0, const double*& dx, double*& df);
template void jacobian_lambda<double>(dim3 dimGrid, dim3 dimBlock, size_t Nx, const double R_, const double*& x0, const double lambda0, double*& dlambda);
