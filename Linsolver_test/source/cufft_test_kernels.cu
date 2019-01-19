#include <cufft_test_kernels.h>




template <typename T_R, typename T_C>
__global__ void kernel_set_values(size_t N, T_C *array)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        array[j].x=sin(T_R(j)/N);
        array[j].y=T_R(0.0);
    }

}

template <typename T_R, typename T_C>
__global__ void kernel_check_values(size_t N, T_C *array1, T_C *array2)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        array1[j].x-=array2[j].x/N;
        array1[j].y-=array2[j].y/N;


    }

}

template <typename T_R, typename T_C>
__global__ void kernel_convert_2R_values(size_t N, T_C* arrayC, T_R* arrayRR, T_R* arrayRI)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        arrayRR[j] = arrayC[j].x;
        arrayRI[j] = arrayC[j].y;


    }

}


//========= kernels wraps ============
template <typename T_R, typename T_C>
void set_values(size_t N, T_C* array)
{

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlock(BLOCKSIZE, 1);
    dim3 dimGrid( k1, 1 );
    kernel_set_values<T_R,T_C><<<dimGrid, dimBlock>>>(N, array);

}


template <typename T_R, typename T_C>
void check_values(size_t N, T_C* array1, T_C* array2)
{

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlock(BLOCKSIZE, 1);
    dim3 dimGrid( k1, 1 );
    kernel_check_values<T_R,T_C><<<dimGrid, dimBlock>>>(N, array1, array2);

}


template <typename T_R, typename T_C>
void convert_2R_values(size_t N, T_C* arrayC, T_R* arrayRR, T_R* arrayRI)
{

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlock(BLOCKSIZE, 1);
    dim3 dimGrid( k1, 1 );
    kernel_convert_2R_values<T_R,T_C><<<dimGrid, dimBlock>>>(N, arrayC, arrayRR, arrayRI);

}



//all precompile variants:
template void set_values<double, cufftDoubleComplex>(size_t N, cufftDoubleComplex* t);
template void set_values<float, cufftComplex>(size_t N, cufftComplex* t);
template void check_values<double, cufftDoubleComplex>(size_t N, cufftDoubleComplex* t1, cufftDoubleComplex* t2);
template void check_values<float, cufftComplex>(size_t N, cufftComplex* t1, cufftComplex* t2);
template void convert_2R_values<double, cufftDoubleComplex>(size_t N, cufftDoubleComplex* t, double* t1, double* t2);
template void convert_2R_values<float, cufftComplex>(size_t N, cufftComplex* t, float* t1, float* t2);