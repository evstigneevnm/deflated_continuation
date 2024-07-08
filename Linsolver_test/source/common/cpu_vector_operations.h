#ifndef __cpu_vector_operations_H__
#define __cpu_vector_operations_H__

#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>
#include <common/dot_product.h>
#include <common/threaded_reduction.h>

// 230707 GLOBAL CHANGE IN THE CONCEPT!
// All vectors MUST contain size, hence vector_operations don't store size explicitly


template <typename T>
struct cpu_vector_operations
{
    // typedef T  scalar_type;
    // typedef T* vector_type;
    using scalar_type = T;
    using vector_type = std::vector<T>;//T*;
    using multivector_type = std::vector<vector_type>;

    bool location;
    dot_product<T, vector_type>* dot = nullptr;
    threaded_reduction<scalar_type, vector_type>* threaded_dot = nullptr;
    int use_threaded_dot = 0;
    size_t sz_default_;

    cpu_vector_operations(size_t sz_p, int use_high_precision_dot_product_ = 0, int use_threaded_dot_ = 0):
    sz_default_(sz_p),
    use_threaded_dot(use_threaded_dot_)
    {
        location=false;
        dot = new dot_product<T, vector_type>(use_high_precision_dot_product_);
        threaded_dot = new threaded_reduction<scalar_type, vector_type>(use_threaded_dot_, use_high_precision_dot_product_);
    }
    ~cpu_vector_operations()
    {
        if(dot!=nullptr)
        {
            delete dot;
        }
        if(threaded_dot!=nullptr)
        {
            delete threaded_dot;
        }
       
    }

    size_t get_default_size()const
    {
        return sz_default_;
    }
    size_t size()const
    {
        return get_default_size();
    }
    size_t get_size(const vector_type& x)const
    {
        return x.size();
    }
    bool device_location()const
    {
        return location;
    }


    void init_vector(vector_type& x, const size_t sz_p = 0)const 
    {
        // x = NULL;
        size_t sz_l = sz_p>0?sz_p:sz_default_;        
        x = std::vector<T>(sz_l);
    }
    template<class ...Args>
    void init_vectors(Args&&...args)const
    {
        std::initializer_list<int>{((void)init_vector(std::forward<Args>(args)), 0 )...};
    } 
    void free_vector(vector_type& x)const 
    {
        x.resize(0);
    }
    template<class ...Args>
    void free_vectors(Args&&...args) const
    {
        std::initializer_list<int>{((void)free_vector(std::forward<Args>(args)), 0 )...};
    }    
    void start_use_vector(vector_type& x, size_t sz_p = 0)const
    {
        // if (x == NULL) x = (T*)malloc( (sz_+1)*sizeof(T));
        size_t sz_l = sz_p>0?sz_p:sz_default_; 
        x.resize(sz_l);
    }
    template<class ...Args>
    void start_use_vectors(Args&&...args)const
    {
        std::initializer_list<int>{((void)start_use_vector(std::forward<Args>(args)), 0 )...};
    }   
    void stop_use_vector(vector_type& x)const
    {
    }
    template<class ...Args>
    void stop_use_vectors(Args&&...args)const
    {
        std::initializer_list<int>{((void)stop_use_vector(std::forward<Args>(args)), 0 )...};
    }
    bool check_is_valid_number(const vector_type &x)const
    {
        size_t sz_l = x.size();
        for (size_t i = 0;i < sz_l;++i)
        {
            if (std::isinf(x[i]))
            {
                return false;
            }
        }
        
        return true;
    }
    bool is_valid_number(const vector_type &x)const
    {
        return check_is_valid_number(x);  
    }

    //multivector operations:
    void init_multivector(multivector_type& x, std::size_t m) const
    {
        x = multivector_type();
        x.reserve(m);
        for(std::size_t j=0;j<m;j++)
        {
            vector_type x_l;
            init_vector(x_l);
            x.push_back(x_l);
        }
    }
    void free_multivector(multivector_type& x, std::size_t m) const
    {
        for(std::size_t j=0;j<m;j++)
        {
            free_vector(x[j]);
        }    
    }
    void start_use_multivector(multivector_type& x, std::size_t m) const
    {
    }
    void stop_use_multivector(multivector_type& x, std::size_t m) const
    {
    }
    [[nodiscard]] vector_type& at(multivector_type& x, std::size_t m, std::size_t k_) const
    {
        if (k_ < 0 || k_>=m  ) 
        {
            throw std::out_of_range("cpu_vector_operations: multivector.at");
        }
        return x[k_];
    }


    scalar_type scalar_prod(const vector_type &x, const vector_type &y, int use_high_prec_ = -1)const
    {
        // T res(0.f);
        // for (int i = 0;i < sz_;++i)
        // {
        //     res += x[i]*y[i];
        // }        
        // return res;
        scalar_type dot_res = T(0.0);

        if (use_threaded_dot == 0)
        {
            if(use_high_prec_ == 1)
            {
                dot->use_high_prec();
            }
            if(use_high_prec_ == 0)
            {
                dot->use_normal_prec();
            }
            dot_res = dot->dot(x, y);
        }
        else
        {
            if(use_high_prec_ == 1)
            {
                threaded_dot->use_high_prec();
            }
            if(use_high_prec_ == 0)
            {
                threaded_dot->use_normal_prec();
            }
            dot_res = threaded_dot->dot(x, y);            
        }
        return dot_res;
    }

    scalar_type norm(const vector_type &x)const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    scalar_type norm_sq(const vector_type &x)const
    {
        return scalar_prod(x, x);
    }    
    scalar_type norm_inf(const vector_type& x)const
    {
        size_t sz_l = x.size();
        scalar_type max_val = 0.0;
        for(size_t j=0;j<sz_l;j++)
        {
            max_val = (max_val<std::abs(x[j]))?std::abs(x[j]):max_val;
        }
        return max_val;
    }
    scalar_type norm_l2(const vector_type& x)const
    {
        return  std::sqrt( norm_sq(x)/size() );
    }
    scalar_type norm2_sq(const vector_type& x)const
    {
        return norm_sq(x);
    }
    scalar_type norm_rank1(const vector_type &x, const scalar_type val_x) const
    {    
        return std::sqrt(scalar_prod(x, x) + val_x*val_x);
    }
    scalar_type sum(const vector_type &x)
    {
        return 0;
    }
    scalar_type asum(const vector_type &x)
    {
        return 0;
    }

    scalar_type normalize(vector_type& x)const
    {
        auto norm_x = norm(x);
        if(norm_x>0.0)
        {
            scale(static_cast<scalar_type>(1.0)/norm_x, x);
        }
        return norm_x;
    }

    void set_value_at_point(scalar_type val_x, size_t at, vector_type& x) const
    {
        x[at] = val_x;
    }
    T get_value_at_point(size_t at, const vector_type& x) const
    {
        return x[at];
    }    
    //calc: x := <vector_type with all elements equal to given scalar value> 
    void assign_scalar(const scalar_type scalar, vector_type& x)const
    {
        size_t sz_l = x.size();
        for (size_t i = 0;i<sz_l;++i) 
            x[i] = scalar;
    }
    //calc: x := mul_x*x + <vector_type of all scalar value> 
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)const
    {
        size_t sz_l = x.size();
        for (size_t i = 0;i < sz_l;++i) 
            x[i] = mul_x*x[i] + scalar;
    }
    void scale(scalar_type scale, vector_type &x)const
    {
        add_mul_scalar(static_cast<scalar_type>(0.0), scale, x);
    }
    //copy: y := x
    void assign(const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::assign: incorrect vector sizes provided");
        }

        size_t sz_l = x.size();
        for (int i = 0;i < sz_l;++i) 
        {
            y[i] = x[i];
        }
    }
    //calc: y := mul_x*x
    void assign_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::assign_mul: incorrect vector sizes provided");
        }        
        for (int i = 0;i < x.size();++i) 
        {
            y[i] = mul_x*x[i];
        }
    }
    
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                               vector_type& z)const
    {
        if((x.size() != y.size() )&&(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations::assign_mul: incorrect vector sizes provided");
        }  
        for (int i = 0;i < x.size();++i) 
            z[i] = mul_x*x[i] + mul_y*y[i];
    }
    //calc: y := mul_x*x + y
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::add_mul: incorrect vector sizes provided");
        }         
        for (int i = 0;i < x.size();++i) 
            y[i] += mul_x*x[i];
    }
    //calc: y := mul_x*x + mul_y*y
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::add_mul: incorrect vector sizes provided");
        } 
        for (int i = 0;i < x.size();++i) 
            y[i] = mul_x*x[i] + mul_y*y[i];
    }
    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
    {
        add_mul(mul_x, x, mul_y, y);
    }

    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                            scalar_type mul_z, vector_type& z)const
    {
        if((x.size() != y.size() )&&(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations::add_mul: incorrect vector sizes provided");
        }         
        for (int i = 0;i < x.size();++i) 
            z[i] = mul_x*x[i] + mul_y*y[i] + mul_z*z[i];
    }
    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, const scalar_type mul_z, vector_type& z) const
    {
        add_mul(mul_x, x, mul_y, y, mul_z, z);
    }    
    void make_abs_copy(const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::make_abs_copy: incorrect vector sizes provided");
        }
        for(size_t j = 0;j<x.size();j++)
        {
            y[j] = std::abs(x[j]);
        }
    }
    void make_abs(vector_type& x)const
    {
        for(size_t j=0;j<x.size();j++)
        {
            auto xa = std::abs(x[j]);
            x[j] = xa;
        }
    }
    // y_j = max(x_j,y_j,sc)
    void max_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::max_pointwise: incorrect vector sizes provided");
        }        
        for(size_t j=0;j<x.size();j++)
        {
            y[j] = (x[j]>y[j])?( (x[j]>sc)?x[j]:sc):( (y[j]>sc)?y[j]:sc);
        }
    }
    void max_pointwise(const scalar_type sc, vector_type& y)const
    {
        for(size_t j=0;j<y.size();j++)
        {
            y[j] = (y[j]>sc)?y[j]:sc;
        }
    }    
    // y_j = min(x_j,y_j,sc)
    void min_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::min_pointwise: incorrect vector sizes provided");
        }         
        for(size_t j=0;j<x.size();j++)
        {
            y[j] = (x[j]<y[j])?( (x[j]<sc)?x[j]:sc):( (y[j]<sc)?y[j]:sc);
        }
    }  
    void min_pointwise(const scalar_type sc, vector_type& y)const
    {
        for(size_t j=0;j<y.size();j++)
        {
            y[j] = (y[j]<sc)?y[j]:sc;
        }
    }        
    //calc: x := x*mul_y*y
    void mul_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations::mul_pointwise: incorrect vector sizes provided");
        }           
        for(size_t j=0;j<x.size();j++)
        {
            x[j] *= mul_y*y[j];
        }        
    }   
    //calc: z := mul_x*x*mul_y*y
    void mul_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                        vector_type& z)const
    {
        if((x.size() != y.size() )&&(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations::mul_pointwise: incorrect vector sizes provided");
        }          
        for(size_t j=0;j<x.size();j++)
        {
            z[j] = (mul_x*x[j])*(mul_y*y[j]);
        }         
    }
    //calc: z := (mul_x*x)/(mul_y*y)
    void div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                        vector_type& z)const
    {
        if((x.size() != y.size() )&&(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations::div_pointwise: incorrect vector sizes provided");
        }     
        for(size_t j=0;j<x.size();j++)
        {
            z[j] = (mul_x*x[j])/(mul_y*y[j]);
        }
    }
    //calc: x := x/(mul_y*y)
    void div_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
    {
        if(x.size() != y.size())
        {
            throw std::logic_error("cpu_vector_operations::div_pointwise: incorrect vector sizes provided");
        }         
        for(size_t j=0;j<x.size();j++)
        {
            x[j] /= static_cast<scalar_type>(1.0)/(mul_y*y[j]);
        }
    }  

    //TODO:!
    std::pair<scalar_type, size_t> max_argmax_element(vector_type& y) const
    {
        auto max_iterator = std::max_element(y.begin(), y.end());
        size_t argmax = std::distance(y.begin(), max_iterator);

        return {*max_iterator, argmax};
    }

    scalar_type max_element(vector_type& x)const
    {
        auto ret = max_argmax_element(x);
        return ret.first;
    }

    size_t argmax_element(vector_type& x)const
    {
        auto ret = max_argmax_element(x);
        return ret.second;
    }
    
    // x.size()<= y.size()
    void assign_slices(const vector_type& x, const std::vector< std::pair<size_t,size_t> > slices, vector_type&y)const
    {
        size_t sz_l = x.size();
        if( sz_l<y.size() )
        {
            throw std::logic_error("cpu_vector_operations::assign_slice: can only be applied to vectors of sizes x.size<=y.size");
        }
        size_t index_y = 0;
        for(auto& slice: slices)
        {
            size_t begin = slice.first;
            size_t end = slice.second; 
            if(end>sz_l)
            {
                throw std::logic_error("cpu_vector_operations::assign_slice: provided slice size is greater than input vector size.");
            }
            for(size_t j = begin; j<end;j++)
            {
                y[index_y++] = x[j];
            }
        }      
    }

    // x.size()<= y.size()
    void assign_skip_slices(const vector_type& x, const std::vector< std::pair<size_t,size_t> > skip_slices, vector_type&y)const
    {
        size_t sz_l = x.size();
        if( sz_l<y.size() )
        {
            throw std::logic_error("cpu_vector_operations::assign_skip_slices: can only be applied to vectors of sizes x.size<=y.size");
        }        
        size_t index_y = 0;
        for(size_t j = 0; j<sz_l;j++)
        {
            for(auto& slice: skip_slices)
            {
                size_t begin = slice.first;
                size_t end = slice.second; 
                if((j<=begin)||(j>end))
                {
                    y[index_y++] = x[j];
                }
            } 
        }
    }

};




#endif