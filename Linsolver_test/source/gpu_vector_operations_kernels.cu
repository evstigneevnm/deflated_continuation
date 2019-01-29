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

//explicit instantiation
template void check_is_valid_number_wrap<double>(dim3 dimGrid, dim3 dimBlock, size_t N, const double *x, bool *result_d);
template void check_is_valid_number_wrap<float>(dim3 dimGrid, dim3 dimBlock, size_t N, const float *x, bool *result_d);
