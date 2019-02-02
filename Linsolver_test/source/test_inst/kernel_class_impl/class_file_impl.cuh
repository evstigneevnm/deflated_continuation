#pragma once

#include <cuda_runtime.h> 
#include "class_file.h"


template<typename T>
__global__ void add_vectors_kernel(int N, const T*& x, T*& y)
{

    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N) return;

    y[j]+=x[j];

}



template<typename T, int BLOCK_SIZE>
void test_class::class_file<T, BLOCK_SIZE>::add_vectors(const T*& x, T*& y)
{
    dim3 grid(floor(sz/BLOCK_SIZE) + 1);
    dim3 block(BLOCK_SIZE);
    add_vectors_kernel<scalar_type><<<grid, block>>>(sz, x, y);
}


