/*
* This file is part of the Lattice Boltzmann multiple GPU distribution. 
(https://github.com/evstigneevnm/LBM_D3Q19_mGPU).
* Copyright (c) 2017-2018 Evstigneev Nikolay Mikhaylovitch and Ryabkov Oleg Igorevich.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 2 only.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef __CUDA_SUPPORT_H__
#define __CUDA_SUPPORT_H__

#include <algorithm>
#include <cctype>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <stdarg.h>
#include <string>
#include <utils/cuda_safe_call.h>
#include <cuda_runtime.h>
#include <scfd/backend/copy/cuda.h>
#include <scfd/utils/init_cuda.h>


inline int init_cuda(int PCI_ID)
{
    return scfd::utils::init_cuda(PCI_ID);
}

inline int init_cuda_dev_num(int device_number = 0)
{
    return scfd::utils::init_cuda(-2, device_number);
}

inline int init_cuda_best_memory()
{
    return scfd::utils::init_cuda_persistent();
}

inline int init_cuda_auto()
{
    return init_cuda_dev_num(0);
}

inline int init_cuda_from_string(const std::string& device_selector)
{
    if((device_selector.empty())||(device_selector == "auto"))
    {
        return init_cuda_auto();
    }
    if(device_selector == "best_mem")
    {
        return init_cuda_best_memory();
    }
    bool is_integer = std::all_of(device_selector.begin(), device_selector.end(), [](unsigned char c)
    {
        return std::isdigit(c) != 0;
    });
    if(is_integer)
    {
        return init_cuda_dev_num(std::stoi(device_selector));
    }
    return scfd::utils::init_cuda_str(device_selector);
}

//
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
//
//__host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) 
//

template <class T>
void host_2_device_cpy(T* device, T* host, size_t size);

template <class T>
void device_2_host_cpy(T* host, T* device, size_t size);

template <class T>
void host_2_device_cpy(T* device, T* host, int Nx, int Ny, int Nz)
{
    host_2_device_cpy(device, host, static_cast<size_t>(Nx)*static_cast<size_t>(Ny)*static_cast<size_t>(Nz));
}

template <class T>
void device_2_host_cpy(T* host, T* device, size_t size)
{
    scfd::cuda_copy<size_t>()(size, device, host);
}

template <class T>
void host_2_device_cpy(T* device, T* host, size_t size)
{
    scfd::cuda_copy<size_t>()(size, host, device);
}

template <class T>
void device_2_device_cpy(const T* device_from, T* device_to, size_t size)
{
    scfd::cuda_copy<size_t>()(size, device_from, device_to);
}


template <class T>
void device_2_host_cpy(T* host, T* device, int Nx, int Ny, int Nz)
{
    device_2_host_cpy(host, device, static_cast<size_t>(Nx)*static_cast<size_t>(Ny)*static_cast<size_t>(Nz));
}


template <class T>
T* device_allocate(int Nx, int Ny, int Nz)
{
    T* m_device;
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_device, mem_size));
    return m_device;    
}


template <class T>
T* device_allocate(size_t size)
{
    T* m_device;
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_device, sizeof(T)*size));
    return m_device;    
}

template <class T>
T* device_allocate_host(int Nx, int Ny, int Nz)
{
    T* m_device;
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&m_device, mem_size));
    return m_device;    
}


template <class T>
T* device_allocate_host(size_t size)
{
    T* m_device;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&m_device, sizeof(T)*size));
    return m_device;    
}

template <class T>
void device_allocate_all(int Nx, int Ny, int Nz, int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T** value=va_arg(ap, T**); /* Increments ap to the next argument. */
        T* temp=device_allocate<T>(Nx, Ny, Nz);
        value[0]=temp;      
    }
    va_end(ap);

}

template <class T>
void device_deallocate(T* array)
{
    CUDA_SAFE_CALL(cudaFree(array));
}

template <class T>
void device_deallocate_host(T* array)
{
    CUDA_SAFE_CALL(cudaFreeHost(array));
}


template <class T>
void device_deallocate_all(int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T* value=va_arg(ap, T*); /* Increments ap to the next argument. */
        CUDA_SAFE_CALL(cudaFree(value));
    }
    va_end(ap);
}


// host operations
template <class T>
T* host_allocate(int Nx, int Ny, int Nz)
{
    
    int size=(Nx)*(Ny)*(Nz);
    T* array;
    array=(T*)malloc(sizeof(T)*size);
    if ( !array )
    {
        throw std::runtime_error(std::string("host memory allocation failed"));
    }
    for(int j=0;j<size;j++)
        array[j]=(T)0;

    return array;
}

// host operations
template <class T>
T* host_allocate(size_t size)
{
    
    T* array;
    array=(T*)malloc(sizeof(T)*size);
    if ( !array )
    {
        throw std::runtime_error(std::string("host memory allocation failed"));
    }
    for(int j=0;j<size;j++)
        array[j]=(T)0;

    return array;
}


template <class T>
void host_allocate_all(int Nx, int Ny, int Nz, int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T** value= va_arg(ap, T**); /* Increments ap to the next argument. */
        T* temp=host_allocate<T>(Nx, Ny, Nz);
        value[0]=temp;      
    }
    va_end(ap);

}


template <class T>
void host_deallocate(T* array)
{
    free(array);
}

template <class T>
void host_deallocate_all(int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T* value= va_arg(ap, T*); /* Increments ap to the next argument. */
        free(value);
    }
    va_end(ap);

}


#endif
