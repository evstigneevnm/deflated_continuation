#ifndef __gpu_vector_operations_H__
#define __gpu_vector_operations_H__

#include <utils/cuda_support.h>
#include <gpu_vector_operations_kernels.h>

template <typename T, int BLOCK_SIZE = 128>
struct gpu_vector_operations
{
    typedef T  scalar_type;
    typedef T* vector_type;


    gpu_vector_operations(int sz) : sz_(sz)
    {
        calculate_cuda_grid();
    }

    gpu_vector_operations(int sz, dim3 dimBlock_, dim3 dimGrid_) : sz_(sz), dimBlock(dimBlock_), dimGrid(dimGrid_)
    {

    }

    int     sz_;
    dim3 dimBlock;
    dim3 dimGrid;

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
            x = device_allocate<T>(sz_);
    }
    void stop_use_vector(vector_type& x)const
    {
        
    }


    bool check_is_valid_number(const vector_type &x)const
    {
        
        bool result_h=true, *result_d;
        result_d=device_allocate<bool>(1);
        host_2_device_cpy<bool>(result_d, &result_h, 1);
        check_is_valid_number_wrap<T>(dimGrid, dimBlock, sz_, x, result_d);
        device_2_host_cpy<bool>(&result_h, result_d, 1);
        return result_h;
    }

    scalar_type norm(const vector_type &x)const
    {
        T    res(0.f);
        for (int i = 0;i < sz_;++i) res += x[i]*x[i];
        return std::sqrt(res);
    }
    scalar_type     scalar_prod(const vector_type &x, const vector_type &y)const
    {
        T    res(0.f);
        for (int i = 0;i < sz_;++i) 
            res += x[i]*y[i];
        return res;
    }
/*    
    //calc: x := <vector_type with all elements equal to given scalar value> 
    void            assign_scalar(scalar_type scalar, vector_type& x)const
    {
        for (int i = 0;i < sz_;++i) x[i] = scalar;
    }
    //calc: x := mul_x*x + <vector_type of all scalar value> 
    void            add_mul_scalar(scalar_type scalar, scalar_type mul_x, vector_type& x)const
    {
        for (int i = 0;i < sz_;++i) x[i] = mul_x*x[i] + scalar;
    }
    //copy: y := x
    void            assign(const vector_type& x, vector_type& y)const
    {
        for (int i = 0;i < sz_;++i) y[i] = x[i];
    }
    //calc: y := mul_x*x
    void            assign_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        for (int i = 0;i < sz_;++i) y[i] = mul_x*x[i];
    }
    //calc: z := mul_x*x + mul_y*y
    void            assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                               vector_type& z)const
    {
        for (int i = 0;i < sz_;++i) z[i] = mul_x*x[i] + mul_y*y[i];
    }
    //calc: y := mul_x*x + mul_y*y
    void            add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
    {
        for (int i = 0;i < sz_;++i) y[i] = mul_x*x[i] + mul_y*y[i];
    }
    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void            add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                            scalar_type mul_z, vector_type& z)const
    {
        for (int i = 0;i < sz_;++i) z[i] = mul_x*x[i] + mul_y*y[i] + mul_z*z[i];
    }

//*/
private:
    void calculate_cuda_grid()
    {
        dim3 dimBlock_s(BLOCK_SIZE);
        unsigned int blocks_x=floor(sz_/( BLOCK_SIZE ))+1;
        dim3 dimGrid_s(blocks_x);
        dimBlock=dimBlock_s;
        dimGrid=dimGrid_s;

    }

};




#endif