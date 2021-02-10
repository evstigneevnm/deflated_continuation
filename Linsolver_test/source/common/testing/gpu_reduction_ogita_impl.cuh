#ifndef __GPU_REDUCTION_IMPL_OGITA_CUH__
#define __GPU_REDUCTION_IMPL_OGITA_CUH__

#include <common/testing/gpu_reduction_ogita.h>

namespace gpu_reduction_ogita_gpu_kernels
{

template<class T>
struct __GPU_REDUCTION_OGITA_H__SharedMemory
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

//Dynamic shared memory specialization
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory<double>
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
struct __GPU_REDUCTION_OGITA_H__SharedMemory<float>
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


template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory< thrust::complex<float> >
{
    __device__ inline operator       thrust::complex<float> *()
    {
        extern __shared__ thrust::complex<float>(__smem_C[]);
        return (thrust::complex<float> *) __smem_C;
    }

    __device__ inline operator const thrust::complex<float> *() const
    {
        extern __shared__ thrust::complex<float>(__smem_C[]);
        return (thrust::complex<float> *) __smem_C;
    }
};
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory< thrust::complex<double> >
{
    __device__ inline operator       thrust::complex<double> *()
    {
        extern __shared__ thrust::complex<double>(__smem_Z[]);
        return (thrust::complex<double> *) __smem_Z;
    }

    __device__ inline operator const thrust::complex<double> *() const
    {
        extern __shared__ thrust::complex<double>(__smem_Z[]);
        return (thrust::complex<double> *) __smem_Z;
    }
};



//this needs specialization for complex, since no fma exist for thrust/complex
template<class T>
__device__ inline T __GPU_REDUCTION_OGITA_H__two_prod_device(T &t, T a, T b)
{
    T p = a*b;
    t = fma(a, b, -p);
    return p;
}

template<class T>
__device__ inline T __GPU_REDUCTION_OGITA_H__two_sum_device(T &t, T a, T b)
{
    T s = a+b;
    T bs = s-a;
    T as = s-bs;
    t = (b-bs) + (a-as);
    return s;
}


template<>
__device__ inline thrust::complex<float> __GPU_REDUCTION_OGITA_H__two_prod_device(thrust::complex<float> &t, thrust::complex<float> a, thrust::complex<float> b)
{
    using T_real = float;
    using TC = typename thrust::complex<float>;
    // T p = a*b;
    // t = fma(a, b, -p);
    // return p;

    T_real a_R = a.real();
    T_real a_I = a.imag();
    T_real b_R = b.real();
    T_real b_I = b.imag();

    T_real p_R1 = a_R*b_R;
    T_real t_R1 = fma(a_R, b_R, -p_R1);
    T_real p_R2 = a_I*b_I;
    T_real t_R2 = fma(a_I, b_I, -p_R2);
    T_real p_I1 = a_R*b_I;
    T_real t_I1 = fma(a_R, b_I, -p_I1);
    T_real p_I2 = a_I*b_R;
    T_real t_I2 = fma(a_I, b_R, -p_I2);

    T_real t1 = T_real(0.0);
    T_real t2 = T_real(0.0);
    T_real p_R = __GPU_REDUCTION_OGITA_H__two_sum_device(t1, p_R1, p_R2);
    T_real p_I = __GPU_REDUCTION_OGITA_H__two_sum_device(t2, p_I1, -p_I2);
    
    TC p = TC(p_R, p_I);
    
    t = TC(t_R1 + t_R2 + t1, t_I1 + t_I2 + t2);

    return p;    
}
template<>
__device__ inline thrust::complex<double> __GPU_REDUCTION_OGITA_H__two_prod_device(thrust::complex<double> &t, thrust::complex<double> a, thrust::complex<double> b)
{
    using T_real = double;
    using TC = typename thrust::complex<double>;
    // T p = a*b;
    // t = fma(a, b, -p);
    // return p;

    T_real a_R = a.real();
    T_real a_I = a.imag();
    T_real b_R = b.real();
    T_real b_I = b.imag();

    T_real p_R1 = a_R*b_R;
    T_real t_R1 = fma(a_R, b_R, -p_R1);
    T_real p_R2 = a_I*b_I;
    T_real t_R2 = fma(a_I, b_I, -p_R2);
    T_real p_I1 = a_R*b_I;
    T_real t_I1 = fma(a_R, b_I, -p_I1);
    T_real p_I2 = a_I*b_R;
    T_real t_I2 = fma(a_I, b_R, -p_I2);

    T_real t1 = T_real(0.0);
    T_real t2 = T_real(0.0);
    T_real p_R = __GPU_REDUCTION_OGITA_H__two_sum_device(t1, p_R1, p_R2);
    T_real p_I = __GPU_REDUCTION_OGITA_H__two_sum_device(t2, p_I1, -p_I2);
 
    TC p = TC(p_R, p_I);
    
    t = TC(t_R1 + t_R2 + t1, t_I1 + t_I2 + t2);

    // printf("p_R1 = %le, p_R2 = %le\n", p_R1, p_R2 );
    return p;    
}



template<class T>
__device__ inline T __GPU_REDUCTION_OGITA_H__two_sum_device(T &t, T a, typename gpu_reduction_ogita_type::type_complex_cast<T>::T b)
{
    T s = a+T(b);
    T bs = s-a;
    T as = s-bs;
    t = (T(b)-bs) + (a-as);
    return s;
}



template<class T>
__device__ typename gpu_reduction_ogita_type::type_complex_cast<T>::T inline cuda_abs(T val)
{
    return( abs(val) );
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



template <class T, class T_vec, unsigned int blockSize, bool nIsPow2, bool first_run>
__global__ void reduce_asum_ogita_kernel(const T_vec g_idata, T_vec g_odata, T_vec err_data, int n)
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
        main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, cuda_abs<T>(g_idata[i]) );
        error_sum += error_local + err_data[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
        {
            if(first_run)
            {
                err_data[i + blockSize] = T(0.0);
            }

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, cuda_abs<T>(g_idata[i+blockSize]) );
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
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 512]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 512];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 512]);
            cdata[tid] = error_sum;//;// + error_l2;
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
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 256]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 256];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 256]);
            cdata[tid] = error_sum;// + error_l2;

        }

        __syncthreads();
        
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            // main_sum = main_sum + sdata[tid + 128];
            // sdata[tid] = main_sum;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 128]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 128]; //__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 128]);
            cdata[tid] = error_sum;// + error_l2;

        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            // main_sum = main_sum + sdata[tid +  64];
            // sdata[tid] = main_sum;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 64]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 64];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 64]);
            cdata[tid] = error_sum;// + error_l2;  
 
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        T *smem = sdata;
        T *cmem = cdata;

        if (blockSize >=  64)
        {
            // main_sum = main_sum + smem[tid + 32];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 32]);
            smem[tid] = main_sum;
            
            error_sum += error_local + cmem[tid + 32];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 32]);
            cmem[tid] = error_sum;// + error_l2;          
        }
        __syncthreads();
        if (blockSize >=  32)
        {
            // main_sum = main_sum + smem[tid + 16];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 16]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 16];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 16]);
            cmem[tid] = error_sum;// + error_l2;   
        }
        __syncthreads();
        if (blockSize >=  16)
        {
            // main_sum = main_sum + smem[tid +  8];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 8]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 8];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 8]);
            cmem[tid] = error_sum;// + error_l2;                
        }
        __syncthreads();
        if (blockSize >=   8)
        {
            // main_sum = main_sum + smem[tid +  4];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 4]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 4];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 4]);
            cmem[tid] = error_sum;// + error_l2;      
        }
        __syncthreads();
        if (blockSize >=   4)
        {
            // main_sum = main_sum + smem[tid +  2];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 2]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 2];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 2]);
            cmem[tid] = error_sum;// + error_l2;  
        }
        __syncthreads();
        if (blockSize >=   2)
        {
            // main_sum = main_sum + smem[tid +  1];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 1]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 1];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 1]);
            cmem[tid] = error_sum;// + error_l2;  
  
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] =  sdata[0];
        err_data[blockIdx.x] = cdata[0];
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
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 512]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 512];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 512]);
            cdata[tid] = error_sum;//;// + error_l2;
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
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 256]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 256];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 256]);
            cdata[tid] = error_sum;// + error_l2;
            // printf("512\n");
            // printf("tid = %i %le<%le %le<(%le+%le)\n", tid, main_sum, sdata[tid+ 256], error_sum, cdata[tid+ 256], error_local);
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
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 128]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 128]; //__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 128]);
            cdata[tid] = error_sum;// + error_l2;
            // printf("256\n");
            // printf("tid = %i %le<%le %le<(%le+%le)\n", tid, main_sum, sdata[tid+ 128], error_sum, cdata[tid+ 128], error_local);
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            // main_sum = main_sum + sdata[tid +  64];
            // sdata[tid] = main_sum;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 64]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 64];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 64]);
            cdata[tid] = error_sum;// + error_l2;  
            // printf("128\n"); 
            // printf("tid = %i %le<%le %le<(%le+%le)\n", tid, main_sum, sdata[tid+ 64], error_sum, cdata[tid+ 64], error_local);    
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        T *smem = sdata;
        T *cmem = cdata;

        if (blockSize >=  64)
        {
            // main_sum = main_sum + smem[tid + 32];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 32]);
            smem[tid] = main_sum;
            
            error_sum += error_local + cmem[tid + 32];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 32]);
            cmem[tid] = error_sum;// + error_l2;          
            // printf("64\n");
            // printf("tid = %i %le<%le %le<(%le+%le)\n", tid, main_sum, smem[tid+ 32], error_sum, cmem[tid+ 32], error_local);
            __syncthreads();
        }

        if (blockSize >=  32)
        {
            // main_sum = main_sum + smem[tid + 16];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 16]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 16];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 16]);
            cmem[tid] = error_sum;// + error_l2;   
            // printf("32\n");      
            // printf("tid = %i %le<%le %le<(%le+%le)\n", tid, main_sum, smem[tid+ 16], error_sum, cmem[tid+ 16], error_local);
            __syncthreads();
        }

        if (blockSize >=  16)
        {
            // main_sum = main_sum + smem[tid +  8];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 8]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 8];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 8]);
            cmem[tid] = error_sum;// + error_l2;      
            // printf("16\n");   
            // printf("tid = %i %le<%le %le<(%le+%le)\n", tid, main_sum, smem[tid+ 8], error_sum, cmem[tid+ 8], error_local);            
            __syncthreads();
        }

        if (blockSize >=   8)
        {
            // main_sum = main_sum + smem[tid +  4];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 4]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 4];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 4]);
            cmem[tid] = error_sum;// + error_l2;      
            // printf("8\n");   
            // printf("tid = %i %le<%le %le<(%le+%le) %le\n", tid, main_sum, smem[tid + 4], error_sum, cmem[tid + 4], error_local, cmem[tid]); 
            __syncthreads();
        }

        if (blockSize >=   4)
        {
            // main_sum = main_sum + smem[tid +  2];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 2]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 2];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 2]);
            cmem[tid] = error_sum;// + error_l2;  
            // printf("4\n");       
            // printf("tid = %i %le<%le %le<(%le+%le) %le\n", tid, main_sum, smem[tid + 2], error_sum, cmem[tid + 2], error_local, cmem[tid]);
            __syncthreads();
        }

        if (blockSize >=   2)
        {
            // main_sum = main_sum + smem[tid +  1];
            // smem[tid] = main_sum;

            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 1]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 1];//__GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 1]);
            cmem[tid] = error_sum;// + error_l2;  
            // printf("2\n"); 
            // printf("tid = %i %le<%le %le<(%le+%le) %le\n", tid, main_sum, smem[tid + 1], error_sum, cmem[tid + 1], error_local, cmem[tid]);      
            __syncthreads();
        }

    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] =  sdata[0];
        err_data[blockIdx.x] = cdata[0];
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
        // printf("%i : %le < %le %le %le %le \n", i, main_sum, g_idata1[i], g_idata2[i], error_local_prod, error_local);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        // printf("%i : %le %le\n", i, g_idata1[i], g_idata2[i]);
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
            // printf("%i : %le < %le %le %le %le \n", i+blockSize, main_sum, g_idata1[i+blockSize], g_idata2[i+blockSize], error_local_prod, error_local);
            // printf("%i : %le %le\n", i+blockSize, g_idata1[i+blockSize], g_idata2[i+blockSize]);
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
            // sdata[tid] = main_sum = main_sum + sdata[tid + 512];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 512]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 512];
            cdata[tid] = error_sum;         
        }

        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid + 256];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 256]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 256];
            cdata[tid] = error_sum;            
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid + 128];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 128]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 128];
            cdata[tid] = error_sum;            
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            // sdata[tid] = main_sum = main_sum + sdata[tid +  64];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, sdata[tid + 64]);
            sdata[tid] = main_sum;

            error_sum += error_local + cdata[tid + 64];
            cdata[tid] = error_sum;            
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        T *smem = sdata;
        T *cmem = cdata;

        if (blockSize >=  64)
        {
            // smem[tid] = main_sum = main_sum + smem[tid + 32];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 32]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 32];
            cmem[tid] = error_sum;
            __syncthreads();
        }

        if (blockSize >=  32)
        {
            // smem[tid] = main_sum = main_sum + smem[tid + 16];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 16]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 16];
            cmem[tid] = error_sum;            
            __syncthreads();
        }

        if (blockSize >=  16)
        {
            // smem[tid] = main_sum = main_sum + smem[tid +  8];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 8]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 8];
            cmem[tid] = error_sum;            
            __syncthreads();
        }

        if (blockSize >=   8)
        {
            // smem[tid] = main_sum = main_sum + smem[tid +  4];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 4]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 4];
            cmem[tid] = error_sum;            
            __syncthreads();
        }

        if (blockSize >=   4)
        {
            // smem[tid] = main_sum = main_sum + smem[tid +  2];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 2]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 2];
            cmem[tid] = error_sum;            
            __syncthreads();
        }

        if (blockSize >=   2)
        {
            // smem[tid] = main_sum = main_sum + smem[tid +  1];
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, smem[tid + 1]);
            smem[tid] = main_sum;

            error_sum += error_local + cmem[tid + 1];
            cmem[tid] = error_sum;                  
            __syncthreads();        
        }
        
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] =  sdata[0];
        err_data[blockIdx.x] = cdata[0];
    }
}

}




template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::get_blocks_threads_shmem(int n, int maxBlocks, int &blocks, int &threads, int &smemSize)
{

    const int use_double_shmem_ = 2;

    threads = (n < BLOCK_SIZE*2) ? nextPow2((n + 1)/ 2) : BLOCK_SIZE;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    smemSize = (threads <= 32) ? use_double_shmem_*2 * threads * sizeof(T) : use_double_shmem_*threads * sizeof(T);
    blocks = (maxBlocks>blocks) ? blocks : maxBlocks;

}


template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_asum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, T_vec errV, int N, bool first_run)
{

    // std::cout << "smemSize = " << smemSize << " threads = " << threads<< std::endl;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    // printf("smemSize = %i, threads = %i, first_run = %d\n", smemSize, threads, first_run);
    
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1024, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 512, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 256, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 128, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 64, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 32, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 16, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 8, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 4, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 2, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_asum_ogita_kernel<T, T_vec, 1, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
        }       
    }
        
}



template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_sum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, T_vec errV, int N, bool first_run)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, errV, N); 
                break;
        }       
    }
        
}



template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_dot(int blocks, int threads, int smemSize, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec errV, int N, bool first_run)
{

    // std::cout << "smemSize = " << smemSize << " threads = " << threads<< std::endl;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    // printf("smemSize = %i, threads = %i, first_run = %d\n", smemSize, threads, first_run);
    
     if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, true, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, true, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1024, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 512:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 512, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 256:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 256, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 128:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 128, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 64:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 64, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 32:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 32, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case 16:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 16, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  8:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 8, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  4:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 4, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  2:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 2, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
            case  1:
                if(first_run)
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, false, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                else
                    gpu_reduction_ogita_gpu_kernels::reduce_dot_ogita_kernel<T, T_vec, 1, false, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, errV, N); 
                break;
        }       
    }
        
}




template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
T gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::reduction_sum(int N, const T_vec InputV, T_vec OutputV, T_vec Output, T_vec errV, T_vec err, bool use_abs_)
{
    T gpu_result = T(0.0);
    T gpu_err = T(0.0);
    int threads = 0, blocks = 0, smemSize=0;

    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, smemSize);

    //perform reduction
    // printf("s = %i, threads=%i, blocks=%i, shmem size=%i\n",N,threads, blocks, smemSize);
    if(use_abs_)
        wrapper_reduce_asum(blocks, threads, smemSize, InputV, OutputV, errV, N, true);
    else
        wrapper_reduce_sum(blocks, threads, smemSize, InputV, OutputV, errV, N, true);
    
    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, smemSize);
        // printf("s = %i, threads=%i, blocks=%i, shmem size=%i\n",s, threads, blocks, smemSize);
        if(use_abs_)
            wrapper_reduce_asum(blocks, threads, smemSize, OutputV, OutputV, errV, s, false);
        else
            wrapper_reduce_sum(blocks, threads, smemSize, OutputV, OutputV, errV, s, false);

        s = (s + (threads*2-1)) / (threads*2);
    }
    
    if (s > 1)
    {
        // printf("s= %i >1, threads=%i, blocks=%i, shmem size=%i\n",s, threads, blocks, smemSize);
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
    int threads = 0, blocks = 0, smemSize=0;

    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, smemSize);

    // perform reduction
    // printf(" s = %i, threads=%i, blocks=%i, shmem size=%i\n",N,threads, blocks, smemSize);
    wrapper_reduce_dot(blocks, threads, smemSize, InputV1, InputV2, OutputV, errV, N, true);
    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, smemSize);
        // printf(" s = %i, threads=%i, blocks=%i, shmem size=%i\n", s, threads, blocks, smemSize);
        wrapper_reduce_sum(blocks, threads, smemSize, OutputV, OutputV, errV, s, false);
        s = (s + (threads*2-1)) / (threads*2);
    }
    
    if (s > 1)
    {
        // printf(" s= %i >1, threads=%i, blocks=%i, shmem size=%i\n", s, threads, blocks, smemSize);
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

    gpu_result = gpu_result + gpu_err;
    // printf(" gpu_result = %.24le\n gpu_err = %.24le\n gpu_lsum= %.24Le\n", gpu_result, gpu_err, gpu_res_long);
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
        T_real p_R2 = -a_I*b_I;
        T_real t_R2 = std::fma(-a_R, b_R, p_R2);
        T_real p_I1 = a_R*b_I;
        T_real t_I1 = std::fma(a_R, b_I, -p_I1);
        T_real p_I2 = a_I*b_R;
        T_real t_I2 = std::fma(a_I, b_R, -p_I2);

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
        T_real p_R2 = -a_I*b_I;
        T_real t_R2 = std::fma(-a_R, b_R, p_R2);
        T_real p_I1 = a_R*b_I;
        T_real t_I1 = std::fma(a_R, b_I, -p_I1);
        T_real p_I2 = a_I*b_R;
        T_real t_I2 = std::fma(a_I, b_R, -p_I2);


        t = thrust::complex<double>(t_R1 + t_R2,t_I1 + t_I2);
        thrust::complex<double> p = thrust::complex<double>(p_R1 + p_R2, p_I1 + p_I2);
        
        return p;
    }




#endif