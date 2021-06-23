#ifndef __gpu_vector_operations_H__
#define __gpu_vector_operations_H__

#include <cuda_runtime.h>
#include <utils/cuda_support.h>
#include <external_libraries/cublas_wrap.h>
#include <utils/curand_safe_call.h>
#include <common/testing/gpu_reduction_ogita.h>
#include <random>
#include <stdexcept>

//debug for file output!
#include <iostream>
#include <fstream>
#include <iomanip>


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
    //typedef typename gpu_vector_operations_type::vec_ops_cuComplex_type_hlp<T>::norm_type Tsc;
    typedef typename gpu_vector_operations_type::vec_ops_cuComplex_type_hlp<T>::norm_type norm_type;
    typedef norm_type Tsc;
    bool location;
    using gpu_reduction_hp_t = gpu_reduction_ogita<T, vector_type>;

    //CONSTRUCTORS!
    gpu_vector_operations(size_t sz_, cublas_wrap *cuBLAS_):
    sz(sz_), 
    cuBLAS(cuBLAS_)
    {
        calculate_cuda_grid();
        location=true;
        x_host = host_allocate<scalar_type>(sz);
    }

    gpu_vector_operations(size_t sz_, dim3 dimBlock_, dim3 dimGrid_): 
    sz(sz_), 
    dimBlock(dimBlock_), 
    dimGrid(dimGrid_)
    {
        location=true;
        x_host = host_allocate<scalar_type>(sz);
    }
    //DISTRUCTOR!
    ~gpu_vector_operations()
    {
        host_deallocate<scalar_type>(x_host);
        if(gpu_reduction_hp != nullptr)
        {
            delete gpu_reduction_hp;
        }
        if(gpu_reduction_hp_rank1 != nullptr)
        {
            delete gpu_reduction_hp_rank1;
        }
    }   

    void use_high_precision()
    {
        if(gpu_reduction_hp == nullptr)
        {
            gpu_reduction_hp = new gpu_reduction_hp_t(sz);    
        }
        if(gpu_reduction_hp_rank1 == nullptr)
        {
            gpu_reduction_hp_rank1 = new gpu_reduction_hp_t(sz+1);    
        }
        
        use_high_precision_dot = true;
    }
    void use_standard_precision()
    {
        use_high_precision_dot = false;
    }

    void init_vector(vector_type& x)const 
    {
        x = NULL;
        //x = device_allocate<scalar_type>(sz);
    }
    void init_vector_rank1(vector_type& x)const 
    {
        x = NULL;
        // x = device_allocate<scalar_type>(sz+1);
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
    void start_use_vector_rank1(vector_type& x)const
    {
        if (x == NULL) 
           x = device_allocate<scalar_type>(sz+1);
    }    
    void stop_use_vector(vector_type& x)const
    {
        
    }
    size_t get_vector_size()
    {
        return sz;
    }
    size_t get_l2_size()
    {
        return std::sqrt(Tsc(sz));
    }    
    bool device_location()
    {
        return location;
    }

    // view returns pointer to a vector that is accessible at host.
    // for cpu vectors this is just a return of the original pointer
    // for gpu vectors a vector is copied to the buffer for local access 
    vector_type view(const vector_type &x) const
    {
        if(x!=nullptr)
        {
            device_2_host_cpy<scalar_type>(x_host, x, sz);
            return(x_host);
        }
        else
            return(nullptr);
    }
    //sets a vector from the buffer. Can be used to set a vector from the host 
    //if the vectors are on gpu. Does nothing for cpu vector.
    void set(vector_type& x) const
    {
        if(x!=nullptr)
        {
            host_2_device_cpy<scalar_type>(x, x_host, sz);
        }
    }
    //sets a vector from a host vector. 
    void set(vector_type& x_host_, vector_type& x_)
    {
        if(x_!=nullptr)
        {
            host_2_device_cpy<scalar_type>(x_, x_host_, sz);
        }
    }    

    //gets the data from a vector to a host vector
    void get(vector_type& x_, vector_type& x_host_)
    {
        if(x_!=nullptr)
        {
            device_2_host_cpy<scalar_type>(x_host_, x_, sz);
        }
    }
    //returns a pointer to the allocated host buffer vector. 
    vector_type get_buffer()
    {
        return(x_host);
    }



    // DEBUG! This plots the vector at request to the file on disk!
    void debug_view(const vector_type& vec, const std::string& f_name) const
    {
        vector_type x_host = view(vec);
        std::ofstream check_file(f_name);
        for(int j=0; j<sz-1; j++)
        {
            check_file << std::scientific << std::setprecision(16) << x_host[j] << std::endl;
        }
        check_file << std::scientific << std::setprecision(16) << x_host[sz-1];
        check_file.close();
    }
    //DEBUG ENDS!

    bool check_is_valid_number(const vector_type &x)const;
    // dot product of two vectors
    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        scalar_type result;
        if(use_high_precision_dot)
        {
            result = gpu_reduction_hp->dot(x, y);
        }
        else
        {        
            cuBLAS->dot<scalar_type>(sz, x, y, &result);
        }
        return (result);
    }
    void scalar_prod(const vector_type &x, const vector_type &y, scalar_type *result)
    {
        if(use_high_precision_dot)
        {
            result[0] = gpu_reduction_hp->dot(x, y);
        }
        else
        {
            cuBLAS->dot<scalar_type>(sz, x, y, result);
        }
    }

    
    typedef typename cublas_real_types::cublas_real_type_hlp<T>::type redef_type;
    redef_type absolute_sum(const vector_type &x)const
    {

        redef_type result;
        if(use_high_precision_dot)
        {
            result = gpu_reduction_hp->asum(x);
        }
        else
        {
            cuBLAS->asum<scalar_type>(sz, x, &result);    
        }
        
        return (result);
    }
    void absolute_sum(const vector_type &x, redef_type *result)
    {
        if(use_high_precision_dot)
        {
            result[0] = gpu_reduction_hp->asum(x);
        }
        else
        {        
            cuBLAS->asum<scalar_type>(sz, x, result);
        }
    }

    
    
    //Observe, that for complex type we need template spetialization! vector type is compelx, but norm type is real!
    //for *PU pointer storage with call from *PU
    Tsc norm(const vector_type &x)const
    {
        Tsc result;
        if(use_high_precision_dot)
        {
            result = gpu_reduction_hp->norm(x);
        }
        else
        {
            cuBLAS->norm2<T>(sz, x, &result);
        }
        return result;
    }
    Tsc norm_l2(const vector_type &x)const
    {
        Tsc result;
        if(use_high_precision_dot)
        {
            result = gpu_reduction_hp->norm(x);
        }
        else
        {     
            cuBLAS->norm2<T>(sz, x, &result);
        }
        return result/std::sqrt(Tsc(sz)); //implements l2 norm as sqrt(sum_j (x_j^2) * (1/x_size))
    }
    //norm for a rank 1 updated vector
    Tsc norm_rank1(const vector_type &x, const scalar_type val_x)
    {
        vector_type y;
        init_vector_rank1(y); start_use_vector_rank1(y); //this is not good, but it will do for now.

        assign(x, y);
        set_value_at_point(val_x, sz, y, sz+1);
        Tsc result;
        if(use_high_precision_dot)
        {
            result = gpu_reduction_hp_rank1->norm(y);
        }
        else        
        {
            cuBLAS->norm2<T>(sz+1, y, &result);
        }
        stop_use_vector(y); free_vector(y);
        return result;

    }
    // Tsc norm_rank1_l2(const vector_type &x, const scalar_type val_x)
    // {
    //     return( norm_rank1(x, val_x)/std::sqrt(Tsc(sz)) );
    // }
    //norm for a rank 1 updated vector with weight
    Tsc norm_rank1_(const vector_type &x, const scalar_type val_x)
    {
        vector_type y;
        init_vector_rank1(y); start_use_vector(y); //this is not good, but it will do for now.

        assign(x, y);
        set_value_at_point(scalar_type(sz)*val_x, sz, y, sz+1);
        Tsc result;
        if(use_high_precision_dot)
        {
            result = gpu_reduction_hp_rank1->norm(y);
        }
        else
        {
            cuBLAS->norm2<T>(sz+1, y, &result);
        }

        stop_use_vector(y); free_vector(y);
        return result;

    }
    Tsc norm_rank1_l2(const vector_type &x, const scalar_type val_x)
    {
        return( norm_rank1_(x, val_x)/std::sqrt(Tsc(sz)) );
    }
    //for GPU pointer storage with call from CPU
    void norm(const vector_type &x, Tsc* result)
    {
        if(use_high_precision_dot)
        {
            throw(std::runtime_error("high precision dot product with GPU-allocated result is not yet implemented."));
        }
        {
            cuBLAS->norm2<T>(sz, x, result);
        }
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
    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y)const;    
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
    //calc: x[at]=val_x for modified size
    void set_value_at_point(scalar_type val_x, size_t at, vector_type& x, size_t sz_l);

    //return value from the vector x[at]
    T get_value_at_point(size_t at, vector_type& x);

    //calc: x := <pseudo random vector with values in (0,1] > 
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
        // function is used to generate uniformly distributed floating point values 
        // between 0.0 and 1.0, where 0.0 is excluded and 1.0 is included. 
        curandGenerateUniformDistribution(gen, vec);    
        CURAND_SAFE_CALL(curandDestroyGenerator(gen));
        //Tsc v_norm = norm_l2(vec);
        //scale(T(v_norm), vec);
    }

    void assign_random(vector_type& vec, scalar_type a, scalar_type b)
    {
        // x = a + (b-a)x
        // z = a + (b-a)z, z - complex: this is wrong! Needed adapter
        // assign_random(vec);
        // add_mul_scalar(a, (b-a), vec);
        assign_random(vec);
        scale_adapter(a, b, vec); // this adapter scales complex numbers in a square aXb
                                          //another adapter is needed to scale in a circle
    }

//*/
private:
    vector_type x_host;
    cublas_wrap *cuBLAS;
    size_t sz;
    dim3 dimBlock;
    dim3 dimGrid;
    void calculate_cuda_grid();
    void curandGenerateUniformDistribution(curandGenerator_t gen, vector_type& vector);
    void scale_adapter(scalar_type a, scalar_type b, vector_type& vec);
    gpu_reduction_hp_t* gpu_reduction_hp = nullptr;
    gpu_reduction_hp_t* gpu_reduction_hp_rank1 = nullptr;
    bool use_high_precision_dot = false;
};




#endif