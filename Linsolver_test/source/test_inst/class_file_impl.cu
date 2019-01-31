#include "class_file_impl.cuh"

template<typename T>
__global__ void add_vectors_kernel(int N, const T*& x, T*& y)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    y[j]+=x[j];

}


