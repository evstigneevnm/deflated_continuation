#ifndef __cpu_vector_operations_HIGH_PREC_H__
#define __cpu_vector_operations_HIGH_PREC_H__

#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/random.hpp>
#include <chrono>

template <int SignificantBits>
struct cpu_vector_operations_var_prec
{
    using scalar_type = boost::multiprecision::number<boost::multiprecision::backends::cpp_bin_float<SignificantBits> >;
    using T = scalar_type;
    using vector_type = std::vector<T>;//T*;
    bool location;
    size_t sz_default_;


private:

    using gen_t = boost::random::independent_bits_engine<boost::random::mt19937, std::numeric_limits<T>::digits, boost::multiprecision::cpp_int>;
    
    mutable vector_type helper_vector_;
    size_t pow2_;
    gen_t* gen_;


    T sum_pow2_helper() const
    {
        size_t sz_reduction = 2 << (pow2_-2); // 2^(pow2_-1)
        while (sz_reduction>1)
        {
            
            // #pragma omp parallel for
            for(int j=0;j<sz_reduction;j++)
            {
                if(j+sz_reduction<sz_default_)
                {
                    helper_vector_[j] = helper_vector_[j] + helper_vector_[j+sz_reduction];
                }
                
            }

            sz_reduction /= 2;
        }
        return helper_vector_[0]+helper_vector_[1];
    }    

public:

    cpu_vector_operations_var_prec(size_t sz_p):
    sz_default_(sz_p)
    {
        location=false;
        pow2_ = static_cast<size_t>(std::ceil( std::log2(sz_default_) ));
        init_vector(helper_vector_);
        start_use_vector(helper_vector_);
        gen_ = new gen_t();
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        gen_->seed( seed ); //init seed with time. remove it to get default seed
    }
    ~cpu_vector_operations_var_prec()
    {
        delete gen_;
        stop_use_vector(helper_vector_);
        free_vector(helper_vector_);
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
            if (boost::multiprecision::isinf(x[i]))
            {
                return false;
            }
        }
        
        return true;
    }
    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        size_t sz_x = x.size();
        size_t sz_y = y.size();
        T res = 0;
        if(sz_x != sz_y)
        {
            throw std::logic_error("cpu_vector_operations_var_prec: scalar_prod can't be made with vetors of different size");
        }
        if( get_default_size() == sz_x)
        {
            res = scalar_prod(const_cast<T*>(x.data()), const_cast<T*>(y.data()) );
        }        
        else
        {
            for(size_t jj = 0;jj < sz_x; ++jj)
            {
                res = res + x[jj]*y[jj];
            }
        }
        
        return res;
    }
    scalar_type scalar_prod(const T* x, const T* y)const
    {

        // #pragma omp parallel for
        for(size_t j=0;j<get_default_size();j++)
        {
            helper_vector_[j] = x[j]*y[j];
        }
        return sum_pow2_helper();
    }

    scalar_type norm(const vector_type &x)const
    {
        return boost::multiprecision::sqrt(scalar_prod(x, x));
    }
    scalar_type norm_sq(const vector_type &x)const
    {
        return scalar_prod(x, x);
    }    
    scalar_type norm_inf(const vector_type& x)const
    {
        size_t sz_l = x.size();
        scalar_type max_val = 0;
        for(size_t j=0;j<sz_l;j++)
        {
            max_val = (max_val<boost::multiprecision::abs(x[j]))?boost::multiprecision::abs(x[j]):max_val;
        }
        return max_val;
    }
    scalar_type norm2_sq(const vector_type& x)const
    {
        return norm_sq(x);
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
        if(norm_x>static_cast<T>(0.0))
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
            throw std::logic_error("cpu_vector_operations_var_prec::assign: incorrect vector sizes provided");
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
            throw std::logic_error("cpu_vector_operations_var_prec::assign_mul: incorrect vector sizes provided");
        }        
        for (int i = 0;i < x.size();++i) 
        {
            y[i] = mul_x*x[i];
        }
    }
    
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, vector_type& z)const
    {
        if((x.size() != y.size() )&&(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations_var_prec::assign_mul: incorrect vector sizes provided");
        }  
        for (int i = 0;i < x.size();++i) 
            z[i] = mul_x*x[i] + mul_y*y[i];
    }
    //calc: y := mul_x*x + y
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations_var_prec::add_mul: incorrect vector sizes provided");
        }         
        for (int i = 0;i < x.size();++i) 
            y[i] += mul_x*x[i];
    }
    //calc: y := mul_x*x + mul_y*y
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations_var_prec::add_mul: incorrect vector sizes provided");
        } 
        for (int i = 0;i < x.size();++i) 
            y[i] = mul_x*x[i] + mul_y*y[i];
    }
    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                            scalar_type mul_z, vector_type& z)const
    {
        if((x.size() != y.size() )&&(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations_var_prec::add_mul: incorrect vector sizes provided");
        }         
        for (int i = 0;i < x.size();++i) 
            z[i] = mul_x*x[i] + mul_y*y[i] + mul_z*z[i];
    }
    void make_abs_copy(const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations_var_prec::make_abs_copy: incorrect vector sizes provided");
        }
        for(size_t j = 0;j<x.size();j++)
        {
            y[j] = boost::multiprecision::abs(x[j]);
        }
    }
    void make_abs(vector_type& x)const
    {
        for(size_t j=0;j<x.size();j++)
        {
            auto xa = boost::multiprecision::abs(x[j]);
            x[j] = xa;
        }
    }
    // y_j = max(x_j,y_j,sc)
    void max_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        if(x.size() != y.size() )
        {
            throw std::logic_error("cpu_vector_operations_var_prec::max_pointwise: incorrect vector sizes provided");
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
            throw std::logic_error("cpu_vector_operations_var_prec::min_pointwise: incorrect vector sizes provided");
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
            throw std::logic_error("cpu_vector_operations_var_prec::mul_pointwise: incorrect vector sizes provided");
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
            throw std::logic_error("cpu_vector_operations_var_prec::mul_pointwise: incorrect vector sizes provided");
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
            throw std::logic_error("cpu_vector_operations_var_prec::div_pointwise: incorrect vector sizes provided");
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
            throw std::logic_error("cpu_vector_operations_var_prec::div_pointwise: incorrect vector sizes provided");
        }         
        for(size_t j=0;j<x.size();j++)
        {
            x[j] /= static_cast<scalar_type>(1.0)/(mul_y*y[j]);
        }
    }  

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
    
    void assign_random(vector_type& vec) const
    {
        boost::random::uniform_real_distribution<T> ur;

        for(size_t j=0;j<vec.size();j++)
        {
            vec[j] = ur(*gen_);
        }
    }
    void assign_random(vector_type& vec, scalar_type a, scalar_type b) const
    {
        boost::random::uniform_real_distribution<T> ur(a, b);

        for(size_t j=0;j<vec.size();j++)
        {
            vec[j] = ur(*gen_);
        }
    }


    // x.size()<= y.size()
    void assign_slices(const vector_type& x, const std::vector< std::pair<size_t,size_t> > slices, vector_type&y)const
    {
        size_t sz_l = x.size();
        if( sz_l<y.size() )
        {
            throw std::logic_error("cpu_vector_operations_var_prec::assign_slice: can only be applied to vectors of sizes x.size<=y.size");
        }
        size_t index_y = 0;
        for(auto& slice: slices)
        {
            size_t begin = slice.first;
            size_t end = slice.second; 
            if(end>sz_l)
            {
                throw std::logic_error("cpu_vector_operations_var_prec::assign_slice: provided slice size is greater than input vector size.");
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
            throw std::logic_error("cpu_vector_operations_var_prec::assign_skip_slices: can only be applied to vectors of sizes x.size<=y.size");
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