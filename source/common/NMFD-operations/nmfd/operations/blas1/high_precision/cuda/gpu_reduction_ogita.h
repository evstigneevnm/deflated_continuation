#ifndef __GPU_REDUCTION_OGITA_H__
#define __GPU_REDUCTION_OGITA_H__

#include <utility>
#include <cstddef>
#include <cmath>
#include <common/macros.h>
#include <cuda_runtime.h>
#include <scfd/utils/cuda_safe_call.h>
#include <thrust/complex.h>
#include <nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_type.h>



template<class T, class T_vec, int BLOCK_SIZE = BLOCK_SIZE_1D, int threads_r = 64>
class gpu_reduction_ogita
{
private:
    using T_real = typename gpu_reduction_ogita_type::type_complex_cast<T>::T;

public:

    gpu_reduction_ogita(size_t vec_size_):
    vec_size(vec_size_)
    {
        vec_helper = allocate_host(vec_size);
        vec_helper_d = allocate_device(vec_size);
        err_helper = allocate_host(vec_size);
        err_helper_d = allocate_device(vec_size);

    }
    ~gpu_reduction_ogita()
    {
        if(vec_helper != nullptr)
        {
            free_host(vec_helper);
        }
        if(vec_helper_d != nullptr)
        {
            free_device(vec_helper_d);
        }
        if(err_helper != nullptr)
        {
            free_host(err_helper);
        }
        if(err_helper_d != nullptr)
        {
            free_device(err_helper_d);
        }        
    }


    T sum(const T_vec d_in)
    {
        T res = reduction_sum(vec_size, d_in, vec_helper_d, vec_helper, err_helper_d, err_helper, false);
        return res;
    }
    T_real asum(const T_vec d_in)
    {
        gpu_reduction_ogita_type::return_real<T> get_real;
        T res = reduction_sum(vec_size, d_in, vec_helper_d, vec_helper, err_helper_d, err_helper, true);
        return get_real.get_real(res);
    }
    T dot(const T_vec d1_in, const T_vec d2_in)
    {
        T res = reduction_dot(vec_size, d1_in, d2_in, vec_helper_d, vec_helper, err_helper_d, err_helper);
        return res;
    }

    T_real norm(const T_vec d_in)
    {
        gpu_reduction_ogita_type::return_real<T> get_real;
        T res = reduction_dot(vec_size, d_in, d_in, vec_helper_d, vec_helper, err_helper_d, err_helper);
        return std::sqrt( get_real.get_real(res) );
    }

private:

    const int maxBlocks = std::pow<int>(2,31) - 1;// sm_30 and greater.

    size_t vec_size;
    T_vec vec_helper_d = nullptr;
    T_vec vec_helper = nullptr;
    T_vec err_helper_d = nullptr;
    T_vec err_helper = nullptr;

    static T_vec allocate_device(size_t size)
    {
        void* ptr = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&ptr, sizeof(T)*size));
        return static_cast<T_vec>(ptr);
    }

    static T_vec allocate_host(size_t size)
    {
        void* ptr = nullptr;
        if(size != 0)
        {
            CUDA_SAFE_CALL(cudaMallocHost(&ptr, sizeof(T)*size, cudaHostAllocDefault));
        }
        return static_cast<T_vec>(ptr);
    }

    static void free_device(T_vec ptr)
    {
        CUDA_SAFE_CALL(cudaFree(static_cast<void*>(ptr)));
    }

    static void free_host(T_vec ptr)
    {
        CUDA_SAFE_CALL(cudaFreeHost(static_cast<void*>(ptr)));
    }

    static void copy_device_to_host(T_vec host, const T_vec device, size_t size)
    {
        CUDA_SAFE_CALL(cudaMemcpy(host, device, sizeof(T)*size, cudaMemcpyDeviceToHost));
    }


    T reduction_sum(int N, const T_vec InputV, T_vec OutputV, T_vec Output, T_vec errV, T_vec err, bool use_abs_);
    
    T reduction_dot(int N, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec Output, T_vec errV, T_vec err);

    // void findBlockSize(int* whichSize, int num_el);
    // for any integer returns the closest larger power_of_two neighbour.
    unsigned int nextPow2(unsigned int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }
    bool isPow2(unsigned int x)
    {
        return ( (x&(x-1))==0 );
    }
    void get_blocks_threads_shmem(int n, int maxBlocks, int& blocks, int& threads, int& smemSize);
    
    void wrapper_reduce_sum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, T_vec errV, int N, bool first_run);
    
    void wrapper_reduce_asum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, T_vec errV, int N, bool first_run);
    
    void wrapper_reduce_dot(int blocks, int threads, int smemSize, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec errV, int N, bool first_run);


    T two_prod_(T &t, T a, T b)
    {
        T p = a*b;
        t = std::fma(a, b, -p);
        return p;
    }


    T two_sum_(T &t, T a, T b)
    {
        T s = a+b;
        T bs = s-a;
        T as = s-bs;
        t = (b-bs) + (a-as);
        return s;
    }



};



#endif

    
