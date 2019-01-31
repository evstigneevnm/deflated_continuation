#pragma once

#include <cuda_runtime.h> 
#include "class_file.h"

template<typename T, int BLOCK_SIZE>
void class_file<T, BLOCK_SIZE>::add_vectors(const T*& x, T*& y)
{
    dim3 grid(floor(sz/BLOCK_SIZE) + 1);
    dim3 block(BLOCK_SIZE);
    add_vectors_kernel<T><<<grid, block>>>(sz, x, y);
}


template class class_file<float>;
template class class_file<double>;