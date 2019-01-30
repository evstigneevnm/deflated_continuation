#include <cuda_runtime.h>
#include <gpu_vector_operations_kernels.h>
//debug
#include <cstdio>

template<typename T>
__global__ void check_is_valid_number_kernel(size_t N, const T *x, bool *result)
{
    
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    if(!isfinite(x[j]))
    {
        result[0]=false;
        return;
    }

}

template<typename T>    
void check_is_valid_number_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T *x, bool *result_d)
{
    check_is_valid_number_kernel<T><<<dimGrid, dimBlock>>>(N, x, result_d);
}

//===
template<typename T> 
__global__ void assign_scalar_kernel(size_t N, const T scalar, T *x)
{

    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    x[j]=T(scalar);

}


template<typename T> 
void assign_scalar_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T scalar, T *x)
{
    assign_scalar_kernel<T><<<dimGrid, dimBlock>>>(N, scalar, x);
}
//===


template<typename T>
__global__ void add_mul_scalar_kernel(size_t N, const T scalar, const T mul_x, T *x)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    x[j]=mul_x*x[j] + scalar;
}


template<typename T>
void add_mul_scalar_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T scalar, const T mul_x, T *x)
{
    add_mul_scalar_kernel<T><<<dimGrid, dimBlock>>>(N, scalar, mul_x, x);
}
//===

template<typename T>
__global__ void assign_mul_kernel(size_t N, const T mul_x, const T *x, T *y)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    y[j]=mul_x*x[j];

}

template<typename T>
void assign_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T mul_x, const T *x, T *y)
{
    assign_mul_kernel<T><<<dimGrid, dimBlock>>>(N, mul_x, x, y);
}
//===


template<typename T>
__global__ void assign_mul_kernel(size_t N, const T mul_x, const T *x, const T mul_y, const T *y, T *z)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = mul_x*x[j] + mul_y*y[j];   

}

template<typename T>
void assign_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t sz, const T mul_x, const T *x, const T mul_y, const T *y, T *z)
{
    assign_mul_kernel<T><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y, z);
}
//===

template<typename T>
__global__ void add_mul_kernel(size_t N,const T mul_x, const T*& x, const T mul_y, T*& y)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    y[j] = mul_x*x[j] + mul_y*y[j];
}


template<typename T>
void add_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t N,const T mul_x, const T*& x, const T mul_y, T*& y)
{
    add_mul_kernel<T><<<dimGrid, dimBlock>>>(N,mul_x, x, mul_y, y);
}
//===

template<typename T>
__global__ void add_mul_kernel(size_t N, const T  mul_x, const T*&  x, const T mul_y, const T*& y, const T mul_z, T*& z)
{

    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = mul_x*x[j] + mul_y*y[j] + mul_z*z[j];
}

template<typename T>
void add_mul_wrap(dim3 dimGrid, dim3 dimBlock, size_t N, const T  mul_x, const T*&  x, const T mul_y, const T*& y, const T mul_z, T*& z)
{
    add_mul_kernel<T><<<dimGrid, dimBlock>>>(N, mul_x, x, mul_y, y, mul_z, z);
}



//explicit instantiation
template void check_is_valid_number_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t N, const double *x, bool *result_d);
template void check_is_valid_number_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t N, const float *x, bool *result_d);

template void assign_scalar_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t N, const float scalar, float *x);
template void assign_scalar_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t N, const double scalar, double *x);
template void assign_scalar_wrap< thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t N, const thrust::complex<float> scalar, thrust::complex<float> *x);
template void assign_scalar_wrap< thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t N, const thrust::complex<double> scalar, thrust::complex<double> *x);

template void add_mul_scalar_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t N, const float scalar, const float mul_x, float *x);
template void add_mul_scalar_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t N, const double scalar, const double mul_x, double *x);
template void add_mul_scalar_wrap< thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t N, const thrust::complex<float> scalar, const thrust::complex<float> mul_x, thrust::complex<float> *x);
template void add_mul_scalar_wrap< thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t N, const thrust::complex<double> scalar, const thrust::complex<double> mul_x, thrust::complex<double> *x);

template void assign_mul_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t N, const float mul_x, const float *x, float *y);
template void assign_mul_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t N, const double mul_x, const double *x, double *y);
template void assign_mul_wrap< thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t N, const thrust::complex<float> mul_x, const thrust::complex<float> *x, thrust::complex<float> *y);
template void assign_mul_wrap< thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t N, const thrust::complex<double> mul_x, const thrust::complex<double> *x, thrust::complex<double> *y);

template void assign_mul_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t sz, const float mul_x, const float *x, const float mul_y, const float *y, float *z);
template void assign_mul_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t sz, const double mul_x, const double *x, const double mul_y, const double *y, double *z);
template void assign_mul_wrap< thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t sz, const thrust::complex<float> mul_x, const thrust::complex<float> *x, const thrust::complex<float> mul_y, const thrust::complex<float> *y, thrust::complex<float> *z);
template void assign_mul_wrap< thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t sz, const  thrust::complex<double> mul_x, const  thrust::complex<double> *x,  const thrust::complex<double> mul_y, const  thrust::complex<double> *y,  thrust::complex<double> *z);

template void add_mul_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t N,const float mul_x, const float*& x, const float mul_y, float*& y);
template void add_mul_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t N,const double mul_x, const double*& x, const double mul_y, double*& y);
template void add_mul_wrap< thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t N,const thrust::complex<float> mul_x, const thrust::complex<float>*& x, const thrust::complex<float> mul_y, thrust::complex<float>*& y);
template void add_mul_wrap< thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t N,const thrust::complex<double> mul_x, const thrust::complex<double>*& x, const thrust::complex<double> mul_y, thrust::complex<double>*& y);

template void add_mul_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t sz, const float  mul_x, const float*&  x, const float mul_y, const float*& y, const float mul_z, float*& z);
template void add_mul_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t sz, const double  mul_x, const double*&  x, const double mul_y, const double*& y, const double mul_z, double*& z);
template void add_mul_wrap< thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t sz, const thrust::complex<float>  mul_x, const thrust::complex<float>*&  x, const thrust::complex<float> mul_y, const thrust::complex<float>*& y, const thrust::complex<float> mul_z, thrust::complex<float>*& z);
template void add_mul_wrap< thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t sz, const thrust::complex<double>  mul_x, const thrust::complex<double>*&  x, const thrust::complex<double> mul_y, const thrust::complex<double>*& y, const thrust::complex<double> mul_z, thrust::complex<double>*& z);