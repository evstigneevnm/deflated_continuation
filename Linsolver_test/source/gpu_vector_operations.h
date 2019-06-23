#ifndef __gpu_vector_operations_H__
#define __gpu_vector_operations_H__

#include <cuda_runtime.h>
#include <utils/cuda_support.h>
#include <external_libraries/cublas_wrap.h>
#include <utils/curand_safe_call.h>
#include <random>

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



template <typename T, int BLOCK_SIZE = 1024>
struct gpu_vector_operations
{
    typedef T  scalar_type;
    typedef T* vector_type;
    typedef typename gpu_vector_operations_type::vec_ops_cuComplex_type_hlp<T>::norm_type Tsc;
    bool location;

    gpu_vector_operations(size_t sz_, cublas_wrap *cuBLAS_):
    sz(sz_), 
    cuBLAS(cuBLAS_)
    {
        calculate_cuda_grid();
        location=true;
    }

    gpu_vector_operations(size_t sz_, dim3 dimBlock_, dim3 dimGrid_): 
    sz(sz_), 
    dimBlock(dimBlock_), 
    dimGrid(dimGrid_)
    {
        location=true;
    }

    void init_vector(vector_type& x)const 
    {
        //x = NULL;
        x = device_allocate<T>(sz);
    }
    void free_vector(vector_type& x)const 
    {
        if (x != NULL) 
            cudaFree(x);
    }
    void start_use_vector(vector_type& x)const
    {
        //if (x == NULL) 
        //    x = device_allocate<T>(sz);
    }
    void stop_use_vector(vector_type& x)const
    {
        
    }
    size_t get_vector_size()
    {
        return sz;
    }
    bool device_location()
    {
        return location;
    }

    bool check_is_valid_number(const vector_type &x)const;
    // dot product of two vectors
    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        scalar_type result;
        cuBLAS->dot<scalar_type>(sz, x, y, &result);
        
        return (result);
    }
    void scalar_prod(const vector_type &x, const vector_type &y, scalar_type *result)
    {
        cuBLAS->dot<scalar_type>(sz, x, y, result);
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
    void norm(const vector_type &x, Tsc* result)
    {
        cuBLAS->norm2<T>(sz, x, result);
    } 
    //calc: x := <vector_type with all elements equal to given scalar value> 
    void assign_scalar(const scalar_type scalar, vector_type& x)const;
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
    void scale(const T alpha, vector_type& x) const
    {
        cuBLAS->scale<T>(sz, alpha, x);
    }

    //calc: x := mul_x*x + <vector_type of all scalar value> 
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)const;

    //copy: y := x
    void assign(const vector_type& x, vector_type& y)const
    {
        //std::cout << sz << "\n";
        cuBLAS->copy<scalar_type>(sz, x, y);
    }
    //swaps vectors: y <-> x
    void swap(vector_type& x, vector_type& y)const
    {
        cuBLAS->swap<scalar_type>(sz, x, y);
    }    
    //calc: y := mul_x*x
    void assign_mul(const scalar_type mul_x, const vector_type& x, vector_type& y)const;
    //cublas axpy: y=y+mul_x*x;
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {   
        cuBLAS->axpy<scalar_type>(sz, mul_x, x, y);
    }
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                               vector_type& z)const;
    //calc: y := mul_x*x + mul_y*y
    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y)const;    //calc: z := mul_x*x + mul_y*y + mul_z*z
    //calc: z := mul_x*x + mul_y*y + mul_z*z;
    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                            const scalar_type mul_z, vector_type& z)const;
    //calc: z := (mul_x*x)*(mul_y*y)
    void mul_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                        vector_type& z)const;
    //calc: x := x*mul_y*y
    void mul_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const;
    //calc: z := (mul_x*x)/(mul_y*y)
    void div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                        vector_type& z)const;
    //calc: x := x/(mul_y*y)
    void div_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const;
    //calc: z := mul_x*x + mul_y*y + mul_w*w + mul_z*z;
    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                            const scalar_type mul_w, const vector_type& w, const scalar_type mul_z, vector_type& z)const;    
    //calc: z := mul_x*x + mul_y*y + mul_v*v + mul_w*w;
    void assign_mul(scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                                        scalar_type mul_v, const vector_type& v, const scalar_type mul_w, const vector_type& w, vector_type& z)const;
    //calc: x[at]=val_x
    void set_value_at_point(scalar_type val_x, size_t at, vector_type& x);

    //calc: x := <pseudo random vector with values in [0,1] > 
    void assign_random(vector_type& vec)
    {
        //vec is on the device!!!
        std::random_device r;
        curandGenerator_t gen;
        /* Create pseudo-random number generator */
        CURAND_SAFE_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
        /* Set seed */
        CURAND_SAFE_CALL( curandSetPseudoRandomGeneratorSeed(gen, r()) );
        /* Generate n doubles on device */
        curandGenerateUniformDistribution(gen, vec, sz);    
        CURAND_SAFE_CALL(curandDestroyGenerator(gen));

    }


//*/
private:
    cublas_wrap *cuBLAS;
    size_t sz;
    dim3 dimBlock;
    dim3 dimGrid;
    void calculate_cuda_grid();
    void curandGenerateUniformDistribution(curandGenerator_t gen, vector_type& vector, size_t size);
};




#endif