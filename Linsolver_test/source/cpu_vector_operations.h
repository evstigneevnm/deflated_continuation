#ifndef __cpu_vector_operations_H__
#define __cpu_vector_operations_H__

#include <cmath>


template <typename T>
struct cpu_vector_operations
{
    typedef T  scalar_type;
    typedef T* vector_type;
    bool location;

    cpu_vector_operations(size_t sz):
    sz_(sz)
    {
        location=false;
    }
    
    size_t get_vector_size()
    {
        return sz_;
    }
    bool device_location()
    {
        return location;
    }

    size_t     sz_;

    void init_vector(vector_type& x)const 
    {
        x = NULL;
    }
    void free_vector(vector_type& x)const 
    {
        if (x != NULL) free(x);
    }
    void start_use_vector(vector_type& x)const
    {
        if (x == NULL) x = (T*)malloc(sz_*sizeof(T));
    }
    void stop_use_vector(vector_type& x)const
    {
    }

    bool check_is_valid_number(const vector_type &x)const
    {
        //TODO check isinf
        for (int i = 0;i < sz_;++i) if (x[i] != x[i]) return false;
        return true;
    }

    scalar_type norm(const vector_type &x)const
    {
        T    res(0.f);
        for (int i = 0;i < sz_;++i) res += x[i]*x[i];
        return std::sqrt(res);
    }
    scalar_type norm_sq(const vector_type &x)const
    {
        T    res(0.f);
        for (int i = 0;i < sz_;++i) res += x[i]*x[i];
    }    
    scalar_type     scalar_prod(const vector_type &x, const vector_type &y)const
    {
        T    res(0.f);
        for (int i = 0;i < sz_;++i) 
            res += x[i]*y[i];
        return res;
    }
    
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
};




#endif