#ifndef __gpu_vector_operations_H__
#define __gpu_vector_operations_H__

#include <utils/cuda_support.h>
#include <external_libraries/cublas_wrap.h>
#include <gpu_vector_operations_kernels.h>

namespace gpu_vector_operations_type{

    template<typename T>
    struct vec_ops_cuComplex_type_hlp
    {
    };


    template<>
    struct vec_ops_cuComplex_type_hlp<float>
    {
        typedef float norm_type;
    };

    template<>
    struct vec_ops_cuComplex_type_hlp<double>
    {
        typedef double norm_type;
    };

    template<>
    struct vec_ops_cuComplex_type_hlp<cuComplex>
    {
        typedef float norm_type;
    };
    template<>
    struct vec_ops_cuComplex_type_hlp<cuDoubleComplex>
    {
        typedef double norm_type;
    };    
    template<>
    struct vec_ops_cuComplex_type_hlp< thrust::complex<float> >
    {
        typedef float norm_type;
    };
    template<>
    struct vec_ops_cuComplex_type_hlp< thrust::complex<double> >
    {
        typedef double norm_type;
    };    
}


template <typename T, int BLOCK_SIZE = 256>
struct gpu_vector_operations
{
    typedef T  scalar_type;
    typedef T* vector_type;
    typedef typename gpu_vector_operations_type::vec_ops_cuComplex_type_hlp<T>::norm_type Tsc;


    gpu_vector_operations(int sz_, cublas_wrap *cuBLAS_) : sz(sz_), cuBLAS(cuBLAS_)
    {
        calculate_cuda_grid();
    }

    gpu_vector_operations(int sz_, dim3 dimBlock_, dim3 dimGrid_) : sz(sz_), dimBlock(dimBlock_), dimGrid(dimGrid_)
    {

    }

    void init_vector(vector_type& x)const 
    {
        x = NULL;
    }
    void free_vector(vector_type& x)const 
    {
        if (x != NULL) 
            cudaFree(x);
    }
    void start_use_vector(vector_type& x)const
    {
        if (x == NULL) 
            x = device_allocate<T>(sz);
    }
    void stop_use_vector(vector_type& x)const
    {
        
    }


    bool check_is_valid_number(const vector_type &x)const
    {
        
        bool result_h=true, *result_d;
        result_d=device_allocate<bool>(1);
        host_2_device_cpy<bool>(result_d, &result_h, 1);
        check_is_valid_number_wrap<scalar_type>(dimGrid, dimBlock, sz, x, result_d);
        device_2_host_cpy<bool>(&result_h, result_d, 1);
        CUDA_SAFE_CALL(cudaFree(result_d));
        return result_h;
    }
    //Observe, that for complex type we need template spetialization! vector type is compelx, but norm type is real!
    //for *PU pointer storage with call from *PU
    Tsc norm(const vector_type &x)const
    {
        Tsc result;
        cuBLAS->norm2<T>(sz, x, &result);
        return result;
    }
    //for GPU pointer storage with call from CPU
    void norm(const vector_type &x, Tsc *result)
    {
        cuBLAS->norm2<T>(sz, x, result);
    }    
    // dot product of two vectors
    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        scalar_type result;
        cuBLAS->dot<scalar_type>(sz, x, y, &result);
        return result;
    }
    void scalar_prod(const vector_type &x, const vector_type &y, scalar_type *result)
    {
        cuBLAS->dot<scalar_type>(sz, x, y, result);
    }
   //calc: x := <vector_type with all elements equal to given scalar value> 
    void assign_scalar(const scalar_type scalar, vector_type& x)const
    {
        assign_scalar_wrap<scalar_type>(dimGrid, dimBlock, sz, scalar, x);
    }
    //calc: ||x||_2=norm, x=x/norm, return norm.
    Tsc normalize(vector_type& x)
    {
        Tsc norm;
        cuBLAS->normalize<T>(sz, x, &norm);
        return norm;
    }
    void normalize(vector_type& x, Tsc *norm)
    {
        cuBLAS->normalize<T>(sz, x, norm);
    }

    //calc: x := mul_x*x + <vector_type of all scalar value> 
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)const
    {
        add_mul_scalar_wrap(dimGrid, dimBlock, sz, scalar, mul_x, x);
    }

    //copy: y := x
    void assign(const vector_type& x, vector_type& y)const
    {
        cuBLAS->copy<scalar_type>(sz, x, y);
    }
    //swaps vectors: y <-> x
    void swap(vector_type& x, vector_type& y)const
    {
        cuBLAS->swap<scalar_type>(sz, x, y);
    }    
    //calc: y := mul_x*x
    void assign_mul(const scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        assign_mul_wrap<scalar_type>(dimGrid, dimBlock, sz, mul_x, x, y);
    }
    //cublas axpy: y=y+mul_x*x;
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {   
        cuBLAS->axpy<scalar_type>(sz, mul_x, x, y);
    }
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                               vector_type& z)const
    {
        assign_mul_wrap<scalar_type>(dimGrid, dimBlock, sz, mul_x, x, mul_y, y, z);
    }
    //calc: y := mul_x*x + mul_y*y
    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y)const
    {
        add_mul_wrap<scalar_type>(dimGrid, dimBlock, sz, mul_x, x, mul_y, y);
    }
    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                            const scalar_type mul_z, vector_type& z)const
    {
        add_mul_wrap<scalar_type>(dimGrid, dimBlock, sz, mul_x, x, mul_y, y, mul_z, z);
    }
    //calc: z := (mul_x*x)*(mul_y*y)
    void mul_pointwise(const scalar_type mul_x, const vector_type x, const scalar_type mul_y, const vector_type y, 
                        vector_type z)
    {
        mul_pointwise_wrap<scalar_type>(dimGrid, dimBlock, sz, mul_x, x, mul_y, y, z);
    }

//*/
private:
    cublas_wrap *cuBLAS;
    int sz;
    dim3 dimBlock;
    dim3 dimGrid;
    void calculate_cuda_grid()
    {
        dim3 dimBlock_s(BLOCK_SIZE);
        unsigned int blocks_x=floor(sz/( BLOCK_SIZE ))+1;
        dim3 dimGrid_s(blocks_x);
        dimBlock=dimBlock_s;
        dimGrid=dimGrid_s;

    }

};




#endif