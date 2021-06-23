#ifndef __gpu_vector_operations_impl_CUH__
#define __gpu_vector_operations_impl_CUH__

#include <cuda_runtime.h>
#include <utils/cuda_support.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <utils/curand_safe_call.h>
#include <thrust/complex.h>


template<class T, class TC>
__global__ void set_complex_vector_random_parts(size_t sz, T* vec_real, T* vec_imag, TC* vec)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=sz) return;

    vec[j] = TC(vec_real[j],vec_imag[j]);

}


template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::curandGenerateUniformDistribution(curandGenerator_t gen, vector_type& vector)
{
}

template <>
void gpu_vector_operations<float, 1024>::curandGenerateUniformDistribution(curandGenerator_t gen, vector_type& vector)
{
    CURAND_SAFE_CALL( curandGenerateUniform(gen, vector, sz) );
}
template <>
void gpu_vector_operations<double, 1024>::curandGenerateUniformDistribution(curandGenerator_t gen, vector_type& vector)
{
    CURAND_SAFE_CALL( curandGenerateUniformDouble(gen, vector, sz) );
}
template <>
void gpu_vector_operations<thrust::complex<float>, 1024>::curandGenerateUniformDistribution(curandGenerator_t gen, vector_type& vector)
{
    
    float* vector_real;
    float* vector_imag;
    vector_real = device_allocate<float>(sz);
    vector_imag = device_allocate<float>(sz);
    try
    {
        CURAND_SAFE_CALL( curandGenerateUniform(gen, vector_real, sz) );
    }
    catch(std::runtime_error& e)
    {
        device_deallocate(vector_real);
        device_deallocate(vector_imag);
        throw std::runtime_error("curandGenerateUniformDistribution<complex float>: error in curandGenerateUniformReal."); 
    }
    try
    {
        CURAND_SAFE_CALL( curandGenerateUniform(gen, vector_imag, sz) );
    }
    catch(std::runtime_error& e)
    {
        device_deallocate(vector_real);
        device_deallocate(vector_imag);
        throw std::runtime_error("curandGenerateUniformDistribution<complex float>: error in curandGenerateUniformImag."); 
    }
    set_complex_vector_random_parts<float, thrust::complex<float> ><<<dimGrid, dimBlock>>>(sz, vector_real, vector_imag, vector);
    device_deallocate(vector_real);
    device_deallocate(vector_imag);
}
template <>
void gpu_vector_operations<thrust::complex<double>, 1024>::curandGenerateUniformDistribution(curandGenerator_t gen, vector_type& vector)
{
    double* vector_real;
    double* vector_imag;
    vector_real = device_allocate<double>(sz);
    vector_imag = device_allocate<double>(sz);
    try
    {
        CURAND_SAFE_CALL( curandGenerateUniformDouble(gen, vector_real, sz) );
    }
    catch(std::runtime_error& e)
    {
        device_deallocate(vector_real);
        device_deallocate(vector_imag);
        throw std::runtime_error("curandGenerateUniformDistribution<complex float>: error in curandGenerateUniformReal."); 
    }
    try
    {
        CURAND_SAFE_CALL( curandGenerateUniformDouble(gen, vector_imag, sz) );
    }
    catch(std::runtime_error& e)
    {
        device_deallocate(vector_real);
        device_deallocate(vector_imag);
        throw std::runtime_error("curandGenerateUniformDistribution<complex float>: error in curandGenerateUniformImag."); 
    }
    set_complex_vector_random_parts<double, thrust::complex<double> ><<<dimGrid, dimBlock>>>(sz, vector_real, vector_imag, vector);
    device_deallocate(vector_real);
    device_deallocate(vector_imag);
}

template<class T, class T_vec>
__global__ void scale_complex_vector(size_t sz, T a, T b, T_vec vec)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=sz) return;

    vec[j] = T(a.real() + (b.real() - a.real())*vec[j].real(), a.imag() + (b.imag() - a.imag())*vec[j].imag());

}
template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::scale_adapter(scalar_type a, scalar_type b, vector_type& vec)
{
    add_mul_scalar(a, (b-a), vec);
}
template <>
void gpu_vector_operations<thrust::complex<double>, 1024>::scale_adapter(scalar_type a, scalar_type b, vector_type& vec)
{
    scale_complex_vector<thrust::complex<double>, vector_type><<<dimGrid, dimBlock>>>(sz, a, b, vec);
}
template<>
void gpu_vector_operations<thrust::complex<float>, 1024>::scale_adapter(scalar_type a, scalar_type b, vector_type& vec)
{
    scale_complex_vector<thrust::complex<float>, vector_type><<<dimGrid, dimBlock>>>(sz, a, b, vec);
}


template<typename T>
__global__ void check_is_valid_number_kernel(int N, const T* x, bool* result);

template<>
__global__ void check_is_valid_number_kernel(int N, const float* x, bool* result)
{
    
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    if(!isfinite(x[j]))
    {
        result[0]=false;
        return;
    }

}

template<>
__global__ void check_is_valid_number_kernel(int N, const double* x, bool* result)
{
    
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    if(!isfinite(x[j]))
    {
        result[0]=false;
        return;
    }

}

template<>
__global__ void check_is_valid_number_kernel(int N, const thrust::complex<double>* x, bool* result)
{
    
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    if(!isfinite(x[j].real()+x[j].imag()))
    {
        result[0]=false;
        return;
    }

}

template<>
__global__ void check_is_valid_number_kernel(int N, const thrust::complex<float>* x, bool* result)
{
    
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    if(!isfinite(x[j].real()+x[j].imag()))
    {
        result[0]=false;
        return;
    }

}

template <typename T, int BLOCK_SIZE>
bool gpu_vector_operations<T, BLOCK_SIZE>::check_is_valid_number(const vector_type& x)const
{
    bool result_h=true, *result_d;
    result_d=device_allocate<bool>(1);
    host_2_device_cpy<bool>(result_d, &result_h, 1);
    check_is_valid_number_kernel<T><<<dimGrid, dimBlock>>>(sz, x, result_d);
    device_2_host_cpy<bool>(&result_h, result_d, 1);
    CUDA_SAFE_CALL(cudaFree(result_d));
    return result_h;
}
//===
template<typename T> 
__global__ void assign_scalar_kernel(size_t N, const T scalar, T* x)
{

    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    x[j]=T(scalar);

}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::assign_scalar(const scalar_type scalar, vector_type& x)const
{
    assign_scalar_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, scalar, x);
}
//===
template<typename T>
__global__ void add_mul_scalar_kernel(size_t N, const T scalar, const T mul_x, T* x)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    x[j]=mul_x*x[j] + scalar;
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)const
{
    add_mul_scalar_kernel<T><<<dimGrid, dimBlock>>>(sz, scalar, mul_x, x);    
}
//===
template<typename T>
__global__ void assign_mul_kernel(size_t N, const T mul_x, const T* x, T* y)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    y[j]=mul_x*x[j];

}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::assign_mul(const scalar_type mul_x, const vector_type& x, vector_type& y)const
{
    assign_mul_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, y);
}
//===
template<typename T>
__global__ void assign_mul_kernel(size_t N, const T mul_x, const T* x, const T mul_y, const T* y, T* z)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = mul_x*x[j] + mul_y*y[j];   
}


template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::assign_mul(scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                           vector_type& z)const
{
    assign_mul_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y, z);
}
//===

template<typename T>
__global__ void add_mul_kernel(size_t N,const T mul_x, const T* x, const T mul_y, T* y)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    y[j] = mul_x*x[j] + mul_y*y[j];
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y)const
{
    add_mul_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y);
}
//===

template<typename T>
__global__ void add_mul_kernel(size_t N, const T  mul_x, const T*  x, const T mul_y, const T* y, const T mul_z, T* z)
{

    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = mul_x*x[j] + mul_y*y[j] + mul_z*z[j];
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                        const scalar_type mul_z, vector_type& z)const
{
    add_mul_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y, mul_z, z);
}
//===

template<typename T>
__global__ void mul_pointwise_kernel(size_t N, const T mul_x, const T* x, const T mul_y, const T* y, T* z)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = (mul_x*x[j])*(mul_y*y[j]);
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::mul_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                    vector_type& z)const
{
    mul_pointwise_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y, z);
}
//===

template<typename T>
__global__ void mul_pointwise_kernel(size_t N, T* x, const T mul_y, const T* y)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    x[j] *= (mul_y*y[j]);
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::mul_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
{
    mul_pointwise_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, x, mul_y, y);
}
//===

template<typename T>
__global__ void div_pointwise_kernel(size_t N, const T mul_x, const T* x, const T mul_y, const T* y, T* z)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = (mul_x*x[j])/(mul_y*y[j]);
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                    vector_type& z)const
{
    div_pointwise_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y, z);
}
//===

template<typename T>
__global__ void div_pointwise_kernel(size_t N, T* x, const T mul_y, const T* y)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    x[j] /= (mul_y*y[j]);
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::div_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
{
    div_pointwise_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, x, mul_y, y);
}
//===
template<typename T>
__global__ void add_mul_kernel(size_t N, const T  mul_x, const T*  x, const T mul_y, const T* y, const T mul_w, const T* w, const T mul_z, T* z)
{

    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = mul_x*x[j] + mul_y*y[j] + mul_w*w[j] + mul_z*z[j];
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                        const scalar_type mul_w, const vector_type& w, const scalar_type mul_z, vector_type& z)const
{
    add_mul_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y, mul_w, w, mul_z, z);
}
//===
template<typename T>
__global__ void assign_mul_kernel(size_t N, const T mul_x, const T* x, const T mul_y, const T* y, 
                                    const T mul_v, const T* v, const T mul_w, const T* w, T* z)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    z[j] = mul_x*x[j] + mul_y*y[j] + mul_v*v[j] + mul_w*w[j];
}


template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::assign_mul(scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                                        scalar_type mul_v, const vector_type& v, const scalar_type mul_w, const vector_type& w, vector_type& z)const
{
    assign_mul_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, mul_x, x, mul_y, y, mul_v, v, mul_w, w, z);
}
//===
template<typename T>
__global__ void set_value_at_point_kernel(size_t N, T val_x, size_t at, T* x)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;
    if(at>=N) return;
    x[at] = T(val_x);
}


template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::set_value_at_point(scalar_type val_x, size_t at, vector_type& x)
{
    set_value_at_point_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, val_x, at, x);
}

template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::set_value_at_point(scalar_type val_x, size_t at, vector_type& x, size_t sz_l)
{
    set_value_at_point_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz_l, val_x, at, x);
}
//===
template<typename T>
__global__ void get_value_at_point_kernel(size_t N, size_t at, T* x, T* val_x)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;
    if(at>=N) return;
    val_x[0] = x[at];
}


template <typename T, int BLOCK_SIZE>
T gpu_vector_operations<T, BLOCK_SIZE>::get_value_at_point(size_t at, vector_type& x)
{
    
    T* val_x_d;
    val_x_d = device_allocate<T>(1);
    get_value_at_point_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz, at, x, val_x_d);
    T val_x = T(0.0);
    device_2_host_cpy<T>(&val_x, val_x_d, 1);
    return val_x;
}

//===
template <typename T, int BLOCK_SIZE>
void gpu_vector_operations<T, BLOCK_SIZE>::calculate_cuda_grid()
{
    dim3 dimBlock_s(BLOCK_SIZE);
    unsigned int blocks_x=floor(sz/( BLOCK_SIZE ))+1;
    dim3 dimGrid_s(blocks_x);
    dimBlock=dimBlock_s;
    dimGrid=dimGrid_s;
}





#endif