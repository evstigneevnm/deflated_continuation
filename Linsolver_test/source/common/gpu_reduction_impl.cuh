#ifndef __GPU_REDUCTION_IMPL_CUH__
#define __GPU_REDUCTION_IMPL_CUH__

#include <common/gpu_reduction.h>


namespace gpu_reduction_impl_gpu_kernels
{

template<class T>
__device__ void warp_reduce_max( volatile T smem[64])
{

    smem[threadIdx.x] = smem[threadIdx.x+32] > smem[threadIdx.x] ? 
                        smem[threadIdx.x+32] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+16] > smem[threadIdx.x] ? 
                        smem[threadIdx.x+16] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+8] > smem[threadIdx.x] ? 
                        smem[threadIdx.x+8] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+4] > smem[threadIdx.x] ? 
                        smem[threadIdx.x+4] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+2] > smem[threadIdx.x] ? 
                        smem[threadIdx.x+2] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+1] > smem[threadIdx.x] ? 
                        smem[threadIdx.x+1] : smem[threadIdx.x];

}

template<class T>
__device__ void warp_reduce_min( volatile T smem[64])
{

    smem[threadIdx.x] = smem[threadIdx.x+32] < smem[threadIdx.x] ? 
                        smem[threadIdx.x+32] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+16] < smem[threadIdx.x] ? 
                        smem[threadIdx.x+16] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+8] < smem[threadIdx.x] ? 
                        smem[threadIdx.x+8] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+4] < smem[threadIdx.x] ? 
                        smem[threadIdx.x+4] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+2] < smem[threadIdx.x] ? 
                        smem[threadIdx.x+2] : smem[threadIdx.x];

    smem[threadIdx.x] = smem[threadIdx.x+1] < smem[threadIdx.x] ? 
                        smem[threadIdx.x+1] : smem[threadIdx.x];

}

template<class T, class T_vec, int threads_r>
__global__ void find_min_max_dynamic_kernel(const T_vec in, T_vec out, int n, int start_adr, int num_blocks)
{

    volatile __shared__ T smem_min[64];
    volatile __shared__ T smem_max[64];
    

    int tid = threadIdx.x + start_adr;

    T max = -__GPU_REDUCTION_H__inf;
    T min = __GPU_REDUCTION_H__inf;
    T resval=0.0;


    // tail part
    int mult = 0;
    for(int i = 1; mult + tid < n; i++)
    {
        resval = in[tid + mult];
    
        min = resval < min ? resval : min;
        max = resval > max ? resval : max;

        mult = (i*threads_r);
    }

    // previously reduced MIN part
    mult = 0;
    int i;
    for(i = 1; mult+threadIdx.x < num_blocks; i++)
    {
        resval = out[threadIdx.x + mult];

        min = resval < min ? resval : min;
        
        mult = (i*threads_r);
    }

    // MAX part
    for(; mult+threadIdx.x < num_blocks*2; i++)
    {
        resval = out[threadIdx.x + mult];

        max = resval > max ? resval : max;
        
        mult = (i*threads_r);
    }


    if(threads_r == 32)
    {
        smem_min[threadIdx.x+32] = T(0.0);
        smem_max[threadIdx.x+32] = T(0.0);
    }
    
    smem_min[threadIdx.x] = min;
    smem_max[threadIdx.x] = max;

    __syncthreads();

    if(threadIdx.x < 32)
    {
        warp_reduce_min(smem_min);
        warp_reduce_max(smem_max);
    }
    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
        out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x]; 
    }


}

template<class T, class T_vec, int els_per_block, int threads_r>
__global__ void find_min_max_kernel(const T_vec in, T_vec out)
{
    volatile __shared__  T smem_min[64];
    volatile __shared__  T smem_max[64];

    int tid = threadIdx.x + blockIdx.x*els_per_block;

    T max = -__GPU_REDUCTION_H__inf;
    T min = __GPU_REDUCTION_H__inf;
    T resval;

    const int iters = els_per_block/threads_r;
    
#pragma unroll
        for(int i = 0; i < iters; i++)
        {

            resval = in[tid + i*threads_r];

            min = resval < min ? resval : min;
            max = resval > max ? resval : max;

        }
    
    
    if(threads_r == 32)
    {
        smem_min[threadIdx.x+32] = T(0.0);
        smem_max[threadIdx.x+32] = T(0.0);
    
    }
    
    smem_min[threadIdx.x] = min;
    smem_max[threadIdx.x] = max;


    __syncthreads();

    if(threadIdx.x < 32)
    {
        warp_reduce_min(smem_min);
        warp_reduce_max(smem_max);
    }
    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
        out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x]; 
    }

}

} //namespace

template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::findBlockSize(int* whichSize, int num_el)
{

    const T pretty_big_number = T(24.0f*1024.0f*1024.0f);

    T ratio = T(num_el)/pretty_big_number;


    if(ratio > T(0.8f))
        (*whichSize) =  5;
    else if(ratio > T(0.6f))
        (*whichSize) =  4;
    else if(ratio > T(0.4f))
        (*whichSize) =  3;
    else if(ratio > T(0.2f))
        (*whichSize) =  2;
    else
        (*whichSize) =  1;


}





template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::compute_reduction_min_max(const T_vec d_in, T_vec d_out, int num_els)
{

    int whichSize = -1; 
        
    findBlockSize(&whichSize, num_els);

    //whichSize = 5;

    int block_size = int(powf(2, whichSize-1))*BLOCK_SIZE;
    int num_blocks = num_els/block_size;
    int tail = num_els - num_blocks*block_size;
    int start_adr = num_els - tail;

    
    if(whichSize == 1)
        gpu_reduction_impl_gpu_kernels::find_min_max_kernel<T, T_vec, BLOCK_SIZE, threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
    else if(whichSize == 2)
        gpu_reduction_impl_gpu_kernels::find_min_max_kernel<T, T_vec, BLOCK_SIZE*2, threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
    else if(whichSize == 3)
        gpu_reduction_impl_gpu_kernels::find_min_max_kernel<T, T_vec, BLOCK_SIZE*4, threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
    else if(whichSize == 4)
        gpu_reduction_impl_gpu_kernels::find_min_max_kernel<T, T_vec, BLOCK_SIZE*8, threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
    else
        gpu_reduction_impl_gpu_kernels::find_min_max_kernel<T, T_vec, BLOCK_SIZE*16, threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 

    gpu_reduction_impl_gpu_kernels::find_min_max_dynamic_kernel<T, T_vec, threads_r><<< 1, threads_r>>>(d_in, d_out, num_els, start_adr, num_blocks);
    
}


namespace gpu_reduction_impl_gpu_kernels
{
//specializaton due to compiler error: declaration is incompatible with previous "__smem" in 'T=double'
template<class T>
struct __GPU_REDUCTION_H__SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
template<>
struct __GPU_REDUCTION_H__SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};
template<>
struct __GPU_REDUCTION_H__SharedMemory<float>
{
    __device__ inline operator       float *()
    {
        extern __shared__ float __smem_f[];
        return (float *)__smem_f;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ float __smem_f[];
        return (float *)__smem_f;
    }
};

template<class T>
__device__ T inline cuda_abs(T val)
{
}
template<>
__device__ double inline cuda_abs(double val)
{
    return( fabs(val) );
}
template<>
__device__ float inline cuda_abs(float val)
{
    return( fabsf(val) );
}


template <class T, class T_vec, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_asum_kernel(const T_vec g_idata, T_vec g_odata, int n)
{
    T *sdata = __GPU_REDUCTION_H__SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = T(0.0);

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += cuda_abs<T>(g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += cuda_abs<T>(g_idata[i+blockSize]);

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }

        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}


template <class T, class T_vec, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_sum_kernel(const T_vec g_idata, T_vec g_odata, int n)
{
    T *sdata = __GPU_REDUCTION_H__SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = T(0.0);

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }

        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <class T, class T_vec, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_dot_kernel(const T_vec g_idata1, const T_vec g_idata2, T_vec g_odata, int n)
{
    T *sdata = __GPU_REDUCTION_H__SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = T(0.0);
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata1[i]*g_idata2[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
        {

            mySum += g_idata1[i+blockSize]*g_idata2[i+blockSize];
        }
        i += gridSize;
    }
    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }

        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

} //namespace


template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::get_blocks_threads_shmem(int n, int maxBlocks, int &blocks, int &threads, int &smemSize)
{

    threads = (n < BLOCK_SIZE*2) ? nextPow2((n + 1)/ 2) : BLOCK_SIZE;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    blocks = (maxBlocks>blocks) ? blocks : maxBlocks;

}



template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_sum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, int N)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 1024, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 512:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 512, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 256:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 256, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 128:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 128, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 64:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 64, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 32:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 32, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 16:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 16, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  8:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 8, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  4:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 4, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  2:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 2, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  1:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 1, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 1024, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 512:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 512, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 256:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 256, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 128:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 128, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 64:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 64, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 32:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 32, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 16:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 16, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  8:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 8, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  4:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 4, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  2:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 2, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  1:
                gpu_reduction_impl_gpu_kernels::reduce_sum_kernel<T, T_vec, 1, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
        }       
    }
        
}

template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_asum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, int N)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 1024, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 512:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 512, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 256:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 256, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 128:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 128, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 64:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 64, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 32:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 32, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 16:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 16, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  8:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 8, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  4:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 4, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  2:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 2, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  1:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 1, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 1024, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 512:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 512, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 256:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 256, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 128:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 128, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 64:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 64, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 32:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 32, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 16:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 16, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  8:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 8, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  4:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 4, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  2:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 2, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  1:
                gpu_reduction_impl_gpu_kernels::reduce_asum_kernel<T, T_vec, 1, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
        }       
    }
        
}


template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_dot(int blocks, int threads, int smemSize, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, int N)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 1024, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 512:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 512, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 256:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 256, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 128:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 128, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 64:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 64, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 32:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 32, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 16:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 16, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  8:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 8, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  4:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 4, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  2:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 2, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  1:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 1, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 1024, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 512:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 512, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 256:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 256, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 128:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 128, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 64:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 64, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 32:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 32, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 16:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 16, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  8:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 8, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  4:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 4, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  2:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 2, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  1:
                gpu_reduction_impl_gpu_kernels::reduce_dot_kernel<T, T_vec, 1, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
        }       
    }
        
}

template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
T gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::reduction_sum(int N, const T_vec InputV, T_vec OutputV, T_vec Output, bool use_abs_)
{
    T gpu_result=0.0;
    int threads = 0, blocks = 0, smemSize=0;

    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, smemSize);

    //perform reduction
    //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
    if(use_abs_)
        wrapper_reduce_asum(blocks, threads, smemSize, InputV, OutputV, N);
    else
        wrapper_reduce_sum(blocks, threads, smemSize, InputV, OutputV, N);
    
    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, smemSize);
        //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
        wrapper_reduce_sum(blocks, threads, smemSize, OutputV, OutputV, s);
        s = (s + (threads*2-1)) / (threads*2);
    }
    if (s > 1)
    {
        //cudaMemcpy(Output, OutputV, s * sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(Output, OutputV, s);

        for (int i=0; i < s; i++)
        {
            gpu_result += Output[i];
        }
        needReadBack = false;
    }
    if (needReadBack)
    {
        //cudaMemcpy(&gpu_result, OutputV, sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(&gpu_result, OutputV, 1);
    }
    return gpu_result;  
}


template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
T gpu_reduction<T, T_vec, BLOCK_SIZE, threads_r>::reduction_dot(int N, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec Output)
{
    T gpu_result=0.0;
    int threads = 0, blocks = 0, smemSize=0;

    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, smemSize);

    //perform reduction
    //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
    wrapper_reduce_dot(blocks, threads, smemSize, InputV1, InputV2, OutputV, N);
    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, smemSize);
        //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
        wrapper_reduce_sum(blocks, threads, smemSize, OutputV, OutputV, s);
        s = (s + (threads*2-1)) / (threads*2);
    }
    if (s > 1)
    {
        //cudaMemcpy(Output, OutputV, s * sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(Output, OutputV, s);

        for (int i=0; i < s; i++)
        {
            gpu_result += Output[i];
        }
        needReadBack = false;
    }
    if (needReadBack)
    {
        //cudaMemcpy(&gpu_result, OutputV, sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(&gpu_result, OutputV, 1);
    }
    return gpu_result;  
}



#endif