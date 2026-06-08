#ifndef __GPU_REDUCTION_IMPL_OGITA_CUH__
#define __GPU_REDUCTION_IMPL_OGITA_CUH__

#include <common/ogita/gpu_reduction_ogita_impl_shmem.cuh>
#include <common/ogita/gpu_reduction_ogita_impl_functions.cuh>
#include <common/ogita/gpu_reduction_ogita.h>

namespace gpu_reduction_ogita_gpu_kernels
{


template <class T, class T_vec, unsigned int blockSize, bool nIsPow2, bool first_run>
__global__ void reduce_asum_ogita_kernel(const T_vec g_idata, T_vec g_odata, T_vec err_data, int n)
{

    using TR = typename gpu_reduction_ogita_type::type_complex_cast<T>::T;
    T *sdata = __GPU_REDUCTION_OGITA_H__SharedMemory<T>();
    T *cdata = &__GPU_REDUCTION_OGITA_H__SharedMemory<T>()[blockSize];


    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T main_sum = T(0.0);
    T error_sum = T(0.0);
    T error_local = T(0.0);
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {

        if(first_run)
        {
            err_data[i] = T(0.0);
        }
        //main_sum += g_idata[i];
        

        main_sum = __GPU_REDUCTION_OGITA_H__two_asum_device(error_local, main_sum, g_idata[i] );
        error_sum += error_local + err_data[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
        {
            if(first_run)
            {
                err_data[i + blockSize] = T(0.0);
            }
            main_sum = __GPU_REDUCTION_OGITA_H__two_asum_device(error_local, main_sum, g_idata[i+blockSize] );
            error_sum += error_local + err_data[i+blockSize];            
            //main_sum += g_idata[i+blockSize];

        }
                

        i += gridSize;
    }
    // each thread puts its local sum into shared memory
    sdata[tid] = main_sum;
    cdata[tid] = error_sum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 512]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 512];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 512]);
            
            // printf("1024\n");
            

        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            // main_sum = main_sum + sdata[tid + 256];
            // sdata[tid] = main_sum;
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 256]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 256];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 256]);

        }

        __syncthreads();
        
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            // main_sum = main_sum + sdata[tid + 128];
            // sdata[tid] = main_sum;
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 128]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 128]; //__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 128]);

        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            // main_sum = main_sum + sdata[tid +  64];
            // sdata[tid] = main_sum;
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 64]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 64];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 64]);
 
        }

        __syncthreads();
    }

    if (tid < 32)
    {

#if (__CUDA_ARCH__ >= 300 )

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64)
        {
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 32]);
            error_sum = error_sum + (error_local + cdata[tid + 32]);
        }
        // Reduce final warp using shuffle
        // assume threadsize = 32
        for (int offset = 32/2; offset > 0; offset >>= 1) 
        {
            main_sum =  __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, shuffle(main_sum, offset) );
            error_sum = error_sum + (error_local + shuffle(error_sum, offset));

        }

#else
        if (blockSize >=  64)
        {
            // main_sum = main_sum + sdata[tid + 32];
            // sdata[tid] = main_sum;

            sdata[tid] =  main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 32]);
            
            
            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 32];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 32]);          
        }
        __syncthreads();
        if (blockSize >=  32)
        {
            // main_sum = main_sum + sdata[tid + 16];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 16]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 16];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 16]);  
        }
        __syncthreads();
        if (blockSize >=  16)
        {
            // main_sum = main_sum + sdata[tid +  8];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 8]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 8];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 8]);                
        }
        __syncthreads();
        if (blockSize >=   8)
        {
            // main_sum = main_sum + sdata[tid +  4];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 4]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 4];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 4]);     
        }
        __syncthreads();
        if (blockSize >=   4)
        {
            // main_sum = main_sum + sdata[tid +  2];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 2]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 2];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 2]); 
        }
        __syncthreads();
        if (blockSize >=   2)
        {
            // main_sum = main_sum + sdata[tid +  1];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 1]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 1];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 1]); 
  
        }
        __syncthreads();
#endif
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = main_sum;
        err_data[blockIdx.x] = error_sum;
    }

}

template <class T, class T_vec, unsigned int blockSize, bool nIsPow2, bool first_run>
__global__ void reduce_sum_ogita_kernel(const T_vec g_idata, T_vec g_odata, T_vec err_data, int n)
{
    T *sdata = __GPU_REDUCTION_OGITA_H__SharedMemory<T>();
    T *cdata = &__GPU_REDUCTION_OGITA_H__SharedMemory<T>()[blockSize];


    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T main_sum = T(0.0);
    T error_sum = T(0.0);
    T error_local = T(0.0);
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {

        if(first_run)
        {
            err_data[i] = T(0.0);
        }
        //main_sum += g_idata[i];
        main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, g_idata[i]);
        error_sum += error_local + err_data[i];

        // printf("%i %i %le %le %le %le\n", blockIdx.x, i, main_sum, g_idata[i], error_local, error_sum);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
        {
            if(first_run)
            {
                err_data[i + blockSize] = T(0.0);
            }

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, g_idata[i+blockSize]);
            error_sum += error_local + err_data[i+blockSize];            
            //main_sum += g_idata[i+blockSize];
            // printf("%i %i %le %le %le %le\n", blockIdx.x, i + blockSize, main_sum, g_idata[i+blockSize], error_local, error_sum);
            //printf("i+bs = %i\n", i+blockSize);
        }
        // else
            // printf("i = %i\n", i);
                

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    //printf("%i %.24le\n", n, error_sum);
    sdata[tid] = main_sum;
    cdata[tid] = error_sum;
    __syncthreads();
    // printf("%i %le %le\n", tid, sdata[tid], cdata[tid]);
    // __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 512]);
            

            cdata[tid] =  error_sum = error_sum + error_local + cdata[tid + 512];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 512]);

            

        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 256]);
           

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 256];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 256]);
        }

        __syncthreads();
        // printf("%i %le %le\n", tid, main_sum, error_sum);
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            // main_sum = main_sum + sdata[tid + 128];
            // sdata[tid] = main_sum;
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 128]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 128]; //__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 128]);
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            // main_sum = main_sum + sdata[tid +  64];
            // sdata[tid] = main_sum;
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 64]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 64];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 64]);
    
        }

        __syncthreads();
    }

    if (tid < 32)
    {
#if (__CUDA_ARCH__ >= 300 )

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64)
        {
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 32]);
            error_sum = error_sum + (error_local + cdata[tid + 32]);
        }
        // Reduce final warp using shuffle
        // assume threadsize = 32
        for (int offset = 32/2; offset > 0; offset >>= 1) 
        {
            main_sum =  __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, shuffle(main_sum, offset) );
            error_sum = error_sum + (error_local + shuffle(error_sum, offset));

        }

#else
        if (blockSize >=  64)
        {
            // main_sum = main_sum + sdata[tid + 32];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 32]);
            
            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 32];//__GPU_REDUCTIO_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 32]);
            __syncthreads();
        }

        if (blockSize >=  32)
        {
            // main_sum = main_sum + sdata[tid + 16];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 16]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 16];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 16]);

            __syncthreads();
        }

        if (blockSize >=  16)
        {
            // main_sum = main_sum + sdata[tid +  8];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 8]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 8];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 8]);          
            __syncthreads();
        }

        if (blockSize >=   8)
        {
            // main_sum = main_sum + sdata[tid +  4];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 4]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 4];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 4]);
            __syncthreads();
        }

        if (blockSize >=   4)
        {
            // main_sum = main_sum + sdata[tid +  2];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 2]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 2];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 2]);
            __syncthreads();
        }

        if (blockSize >=   2)
        {
            // main_sum = main_sum + sdata[tid +  1];
            // sdata[tid] = main_sum;

            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 1]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 1];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 1]);     
            __syncthreads();
        }
#endif
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] =  main_sum;
        err_data[blockIdx.x] = error_sum;
        // printf("=====\n");
        // printf("tid = %i %le %le\n", tid, g_odata[blockIdx.x], err_data[blockIdx.x]);
    }
}


template <class T, class T_vec, unsigned int blockSize, bool nIsPow2, bool first_run>
__global__ void reduce_dot_ogita_kernel(const T_vec g_idata1, const T_vec g_idata2, T_vec g_odata, T_vec err_data, int n)
{
    T *sdata = __GPU_REDUCTION_OGITA_H__SharedMemory<T>();
    T *cdata = &__GPU_REDUCTION_OGITA_H__SharedMemory<T>()[blockSize];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T main_sum = T(0.0);
    T error_sum = T(0.0);
    T error_local = T(0.0);
    T error_local_prod = T(0.0);
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        if(first_run)
        {
            err_data[i] = T(0.0);
        }

        // main_sum += g_idata1[i]*g_idata2[i];
        T res_l = __GPU_REDUCTION_OGITA_H__two_prod_device(error_local_prod, g_idata1[i], g_idata2[i]);
        main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, res_l);
        
        error_sum += error_local_prod + error_local + err_data[i];        
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays

        if (nIsPow2 || i + blockSize < n)
        {

            if(first_run)
            {
                err_data[i+blockSize] = T(0.0);
            }
            //main_sum += g_idata1[i+blockSize]*g_idata2[i+blockSize];
            T res_l = __GPU_REDUCTION_OGITA_H__two_prod_device(error_local_prod, g_idata1[i+blockSize], g_idata2[i+blockSize]);
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, res_l);            
            error_sum += error_local_prod + error_local + err_data[i+blockSize];               

        }
        i += gridSize;
    }
    // each thread puts its local sum into shared memory
    // print_var(error_local_prod);
    sdata[tid] = main_sum;
    cdata[tid] = error_sum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid + 512];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 512]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 512];
                     
        }

        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid + 256];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 256]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 256];
                       
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid + 128];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 128]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 128];
                       
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid +  64];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 64]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 64];
                        
        }

        __syncthreads();
    }

    if (tid < 32)
    {
#if (__CUDA_ARCH__ >= 300 )

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64)
        {
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 32]);
            error_sum = error_sum + (error_local + cdata[tid + 32]);
        }
        // Reduce final warp using shuffle
        // assume threadsize = 32
        for (int offset = 32/2; offset > 0; offset >>= 1) 
        {
            main_sum =  __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, shuffle(main_sum, offset) );
            error_sum = error_sum + (error_local + shuffle(error_sum, offset));

        }

#else

        if (blockSize >=  64)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid + 32];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 32]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 32];
            
            __syncthreads();
        }

        if (blockSize >=  32)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid + 16];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 16]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 16];
                       
            __syncthreads();
        }

        if (blockSize >=  16)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid +  8];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 8]);

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 8];
                        
            __syncthreads();
        }

        if (blockSize >=   8)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid +  4];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 4]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 4];
                        
            __syncthreads();
        }

        if (blockSize >=   4)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid +  2];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 2]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 2];
                        
            __syncthreads();
        }

        if (blockSize >=   2)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid +  1];
            sdata[tid] = main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 1]);
            

            cdata[tid] = error_sum = error_sum + error_local + cdata[tid + 1];
                              
            __syncthreads();        
        }
#endif   
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] =  main_sum;
        err_data[blockIdx.x] = error_sum;
    }
}

}




template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::get_blocks_threads_shmem(int n, int maxBlocks, int &blocks, int &threads, int &sdataSize)
{

    const int use_double_shmem_ = 2;

    threads = (n < BLOCK_SIZE*2) ? nextPow2((n + 1)/ 2) : BLOCK_SIZE;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    sdataSize = (threads <= 32) ? use_double_shmem_*2 * threads * sizeof(T) : use_double_shmem_*threads * sizeof(T);
    blocks = (maxBlocks>blocks) ? blocks : maxBlocks;

}


template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_asum(int blocks, int threads, int sdataSize, const T_vec InputV, T_vec OutputV, T_vec errV, int N, bool first_run)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
        }       
    }
        
}



template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_sum(int blocks, int threads, int sdataSize, const T_vec InputV, T_vec OutputV, T_vec errV, int N, bool first_run)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV, OutputV, errV, N); 
                break;
        }       
    }
        
}



template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_dot(int blocks, int threads, int sdataSize, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec errV, int N, bool first_run)
{

    

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, true, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, true, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, false, true><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, false, false><<< dimGrid, dimBlock, sdataSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
        }       
    }
        
}




template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
T gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::reduction_sum(int N, const T_vec InputV, T_vec OutputV, T_vec Output, T_vec errV, T_vec err, bool use_abs_)
{
    T gpu_result = T(0.0);
    T gpu_err = T(0.0);
    int threads = 0, blocks = 0, sdataSize=0;

    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, sdataSize);

    //perform reduction
    // printf("s = %i, threads=%i, blocks=%i, shmem size=%i\n",N,threads, blocks, sdataSize);
    if(use_abs_)
        wrapper_reduce_asum(blocks, threads, sdataSize, InputV, OutputV, errV, N, true);
    else
        wrapper_reduce_sum(blocks, threads, sdataSize, InputV, OutputV, errV, N, true);
    
    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, sdataSize);
        // printf("s = %i, threads=%i, blocks=%i, shmem size=%i\n",s, threads, blocks, sdataSize);
        if(use_abs_)
            wrapper_reduce_asum(blocks, threads, sdataSize, OutputV, OutputV, errV, s, false);
        else
            wrapper_reduce_sum(blocks, threads, sdataSize, OutputV, OutputV, errV, s, false);

        s = (s + (threads*2-1)) / (threads*2);
    }
    
    if (s > 1)
    {
        // printf("s= %i >1, threads=%i, blocks=%i, shmem size=%i\n",s, threads, blocks, sdataSize);
        device_2_host_cpy<T>(Output, OutputV, s);
        device_2_host_cpy<T>(err, errV, s);

        T tt = T(0.0);
        for (int i=0; i < s; i++)
        {
            gpu_result = two_sum_(tt, gpu_result, Output[i]);
            gpu_err += tt + err[i];
        }
        needReadBack = false;
    }
    if (needReadBack)
    {
        // printf("s = %i == 1, needReadBack.\n",s);
        device_2_host_cpy<T>(&gpu_result, OutputV, 1);
        device_2_host_cpy<T>(&gpu_err, errV, 1);
    }

    T gpu_res_long = gpu_result + gpu_err;
    // printf(" gpu_result = %.24le\n gpu_err = %.24le\n gpu_lsum= %.24le\n", gpu_result, gpu_err, gpu_res_long);
    return(gpu_res_long);
}



template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
T gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::reduction_dot(int N, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec Output, T_vec errV, T_vec err)
{
    T gpu_result = T(0.0);
    T gpu_err = T(0.0);
    int threads = 0, blocks = 0, sdataSize=0;

    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, sdataSize);
    // perform reduction
    // printf(" s = %i, threads=%i, blocks=%i, shmem size=%i\n",N,threads, blocks, sdataSize);
    wrapper_reduce_dot(blocks, threads, sdataSize, InputV1, InputV2, OutputV, errV, N, true);
    
    // device_2_host_cpy<T>(err, errV, N);
    // for(int jj = 0;jj<N;jj++)
    // {
    //     std::cout << err[jj] << std::endl;
    // }

    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, sdataSize);
        // printf(" s = %i, threads=%i, blocks=%i, shmem size=%i\n", s, threads, blocks, sdataSize);
        wrapper_reduce_sum(blocks, threads, sdataSize, OutputV, OutputV, errV, s, false);
        s = (s + (threads*2-1)) / (threads*2);
    }
    
    if (s > 1)
    {
        // printf(" s= %i >1, threads=%i, blocks=%i, shmem size=%i\n", s, threads, blocks, sdataSize);
        
        device_2_host_cpy<T>(Output, OutputV, s);
        device_2_host_cpy<T>(err, errV, s);

        T tt = T(0.0);
        for (int i=0; i < s; i++)
        {
            gpu_result = two_sum_(tt, gpu_result, Output[i]);
            gpu_err += tt + err[i];
        }
        needReadBack = false;
    }
    if (needReadBack)
    {
        // printf(" s = %i == 1, needReadBack.\n",s);
        device_2_host_cpy<T>(&gpu_result, OutputV, 1);
        device_2_host_cpy<T>(&gpu_err, errV, 1);
    }

    // printf(" gpu_result = %.24le gpu_err = %.24le\n", (double)gpu_result, (double)gpu_err);
    // std::cout << "gpu_res = " << gpu_result << " gpu_err = " << gpu_err << std::endl;
    gpu_result = gpu_result + gpu_err;
    
    return(gpu_result);
}



// specialization for thrust::complex<T>
template<>
thrust::complex<float> gpu_reduction_ogita<thrust::complex<float>, thrust::complex<float>* >::two_prod_(thrust::complex<float> &t, thrust::complex<float> a, thrust::complex<float> b)
    {
        
        // p = a*b;
        // t = std::fma(a, b, -p);
        // return p;       

        T_real a_R = a.real();
        T_real a_I = a.imag();
        T_real b_R = b.real();
        T_real b_I = b.imag();

        T_real p_R1 = a_R*b_R;
        T_real t_R1 = std::fma(a_R, b_R, -p_R1);
        T_real p_R2 = a_I*b_I;
        T_real t_R2 = std::fma(a_I, b_I, -p_R2);
        T_real p_I1 = a_R*b_I;
        T_real t_I1 = std::fma(a_R, b_I, -p_I1);
        T_real p_I2 = -a_I*b_R;
        T_real t_I2 = std::fma(-a_I, b_R, -p_I2);

        t = thrust::complex<float>(t_R1 + t_R2,t_I1 + t_I2);
        thrust::complex<float> p = thrust::complex<float>(p_R1 + p_R2, p_I1 + p_I2);
        
        return p;
    }
template<>
thrust::complex<double> gpu_reduction_ogita<thrust::complex<double>, thrust::complex<double>* >::two_prod_(thrust::complex<double> &t, thrust::complex<double> a, thrust::complex<double> b)
    {
        
        // p = a*b;
        // t = std::fma(a, b, -p);
        // return p;       

        T_real a_R = a.real();
        T_real a_I = a.imag();
        T_real b_R = b.real();
        T_real b_I = b.imag();

        T_real p_R1 = a_R*b_R;
        T_real t_R1 = std::fma(a_R, b_R, -p_R1);
        T_real p_R2 = a_I*b_I;
        T_real t_R2 = std::fma(a_I, b_I, -p_R2);
        T_real p_I1 = a_R*b_I;
        T_real t_I1 = std::fma(a_R, b_I, -p_I1);
        T_real p_I2 = -a_I*b_R;
        T_real t_I2 = std::fma(-a_I, b_R, -p_I2);


        t = thrust::complex<double>(t_R1 + t_R2,t_I1 - t_I2);
        thrust::complex<double> p = thrust::complex<double>(p_R1 + p_R2, p_I1 - p_I2);
        
        return p;
    }




#endif