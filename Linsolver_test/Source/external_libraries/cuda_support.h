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

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <stdarg.h>
#include <cuda_runtime.h>

#include "cuda_safe_call.h"
//TODO: put throw everywhere!!!


int InitCUDA(int GPU_number=-1);



template <class T>
void host_2_device_cpy(T* device, T* host, int Nx, int Ny, int Nz)
{
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice));

}

template <class T>
void device_2_host_cpy(T* host, T* device, size_t size)
{
    CUDA_SAFE_CALL(cudaMemcpy(host, device, sizeof(T)*size, cudaMemcpyDeviceToHost));
}

template <class T>
void host_2_device_cpy(T* device, T* host, size_t size)
{
    CUDA_SAFE_CALL(cudaMemcpy(device, host, sizeof(T)*size, cudaMemcpyHostToDevice));
}


template <class T>
void device_2_host_cpy(T* host, T* device, int Nx, int Ny, int Nz)
{
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost));
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

