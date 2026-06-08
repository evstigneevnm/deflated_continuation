#ifndef __cpu_vector_operations_HIGH_PREC_H__
#define __cpu_vector_operations_HIGH_PREC_H__

#include <cstddef>
#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/random.hpp>
#include <chrono>

#include <nmfd/operations/vector_space_base.h>
#include <common/scalar_math.h>

namespace nmfd_operations_detail
{

template <unsigned int SignificantBits>
using var_prec_scalar_t =
    boost::multiprecision::number<boost::multiprecision::backends::cpp_bin_float<SignificantBits>>;

} // namespace nmfd_operations_detail

template <unsigned int SignificantBitsP = 100>
struct cpu_vector_operations_var_prec :
    public nmfd::operations::vector_space_base
    <
        nmfd_operations_detail::var_prec_scalar_t<SignificantBitsP>,
        std::vector<nmfd_operations_detail::var_prec_scalar_t<SignificantBitsP>>,
        std::vector<std::vector<nmfd_operations_detail::var_prec_scalar_t<SignificantBitsP>>>,
        std::ptrdiff_t,
        nmfd_operations_detail::var_prec_scalar_t<SignificantBitsP>
    >
{
    static constexpr unsigned int SignificantBits = SignificantBitsP;
    using scalar_type = nmfd_operations_detail::var_prec_scalar_t<SignificantBits>;
    using norm_type = scalar_type;
    using Tsc = scalar_type;
    using T = scalar_type;
    using vector_type = std::vector<T>;//T*;
    using multivector_type = std::vector<vector_type>;
    using ordinal_type = std::ptrdiff_t;
    using big_ordinal_type = ordinal_type;
    using nmfd_parent_type = nmfd::operations::vector_space_base
    <
        scalar_type,
        vector_type,
        multivector_type,
        ordinal_type,
        norm_type
    >;
    bool location;
    size_t sz_default_;


private:

    
    // template <int N> boost::multiprecision::bmp::number< bmp::cpp_dec_float<N> > const my_const_pi = boost::multiprecision::default_ops::get_constant_pi<bmp::cpp_dec_float<N> >();



    using gen_t = boost::random::independent_bits_engine<boost::random::mt19937, std::numeric_limits<T>::digits, boost::multiprecision::cpp_int>;
    
    mutable vector_type helper_vector_;
    gen_t* gen_;


    static size_t checked_multivector_index(ordinal_type m, ordinal_type k)
    {
        if(k < ordinal_type(0) || k >= m)
        {
            throw std::out_of_range("cpu_vector_operations_var_prec: multivector index");
        }
        return static_cast<size_t>(k);
    }

    void ensure_helper_size(size_t count) const
    {
        if(helper_vector_.size() < count)
        {
            helper_vector_.resize(count);
        }
    }

    T sum_helper(size_t count) const
    {
        if(count == 0)
        {
            return static_cast<T>(0.0);
        }

        size_t active = count;
        while(active > 1)
        {
            size_t out = 0;
            size_t j = 0;
            for(; j + 1 < active; j += 2)
            {
                helper_vector_[out++] = helper_vector_[j] + helper_vector_[j + 1];
            }
            if(j < active)
            {
                helper_vector_[out++] = helper_vector_[j];
            }
            active = out;
        }
        return helper_vector_[0];
    }    

public:

    cpu_vector_operations_var_prec(size_t sz_p):
    nmfd_parent_type(false),
    sz_default_(sz_p)
    {
        location=false;
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
    size_t get_vector_size()const
    {
        return get_default_size();
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
    unsigned int get_fp_prec()const
    {
        return SignificantBits;
    }

    norm_type get_l2_size() const
    {
        return common::scalar_math::sqrt(static_cast<norm_type>(sz_default_));
    }


    void init_vector(vector_type& x)const override
    {
        init_vector(x, size_t(0));
    }

    void init_vector(vector_type& x, const size_t sz_p)const
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
    void free_vector(vector_type& x)const override
    {
        x.resize(0);
    }
    template<class ...Args>
    void free_vectors(Args&&...args) const
    {
        std::initializer_list<int>{((void)free_vector(std::forward<Args>(args)), 0 )...};
    }    
    void start_use_vector(vector_type& x)const override
    {
        start_use_vector(x, size_t(0));
    }

    void start_use_vector(vector_type& x, size_t sz_p)const
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
    void stop_use_vector(vector_type& x)const override
    {
    }
    template<class ...Args>
    void stop_use_vectors(Args&&...args)const
    {
        std::initializer_list<int>{((void)stop_use_vector(std::forward<Args>(args)), 0 )...};
    }

    void init_multivector(multivector_type& x, ordinal_type m)const override
    {
        x.clear();
        x.reserve(static_cast<size_t>(m));
        for(ordinal_type j = 0; j < m; ++j)
        {
            vector_type x_l;
            init_vector(x_l);
            x.push_back(std::move(x_l));
        }
    }

    void free_multivector(multivector_type& x, ordinal_type m)const override
    {
        for(ordinal_type j = 0; j < m; ++j)
        {
            free_vector(x[static_cast<size_t>(j)]);
        }
        x.clear();
    }

    void start_use_multivector(multivector_type& x, ordinal_type m)const override
    {
        for(ordinal_type j = 0; j < m; ++j)
        {
            start_use_vector(x[static_cast<size_t>(j)]);
        }
    }

    void stop_use_multivector(multivector_type& x, ordinal_type m)const override
    {
    }

    vector_type& at(multivector_type& x, size_t m, size_t k_) const
    {
        if(k_ >= m)
        {
            throw std::out_of_range("cpu_vector_operations_var_prec: multivector.at");
        }
        return x[k_];
    }

    void assign(const multivector_type& mx, ordinal_type m, ordinal_type k_, vector_type& x) const override
    {
        assign(mx[checked_multivector_index(m, k_)], x);
    }

    void assign(const vector_type& x, multivector_type& mx, ordinal_type m, ordinal_type k_) const override
    {
        assign(x, mx[checked_multivector_index(m, k_)]);
    }

    scalar_type scalar_prod(const multivector_type& mx, ordinal_type m, ordinal_type k_, const vector_type& y) const override
    {
        return scalar_prod(mx[checked_multivector_index(m, k_)], y);
    }

    scalar_type scalar_prod_l2(const multivector_type& mx, ordinal_type m, ordinal_type k_, const vector_type& y) const override
    {
        return scalar_prod_l2(mx[checked_multivector_index(m, k_)], y);
    }

    void add_lin_comb(const scalar_type mul_x, const multivector_type& mx, ordinal_type m, ordinal_type k_, const scalar_type mul_y, vector_type& y) const override
    {
        add_lin_comb(mul_x, mx[checked_multivector_index(m, k_)], mul_y, y);
    }

    bool check_is_valid_number(const vector_type &x)const
    {
        size_t sz_l = x.size();
        for (size_t i = 0;i < sz_l;++i)
        {
            if (!boost::multiprecision::isfinite(x[i]))
            {
                return false;
            }
        }
        
        return true;
    }

    bool is_valid_number(const vector_type& x) const override
    {
        return check_is_valid_number(x);
    }

    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const override
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

        ensure_helper_size(get_default_size());
        // #pragma omp parallel for
        for(size_t j=0;j<get_default_size();j++)
        {
            helper_vector_[j] = x[j]*y[j];
        }
        return sum_helper(get_default_size());
    }

    scalar_type scalar_prod_l2(const vector_type& x, const vector_type& y) const override
    {
        return scalar_prod(x, y);
    }

    scalar_type norm(const vector_type &x)const override
    {
        return boost::multiprecision::sqrt(scalar_prod(x, x));
    }
    scalar_type norm_sq(const vector_type &x)const override
    {
        return scalar_prod(x, x);
    }    

    scalar_type norm_l2(const vector_type& x)const override
    {
        auto result = norm(x);
        return result;//std::sqrt(sz);
    }
    scalar_type norm_inf(const vector_type& x)const override
    {
        size_t sz_l = x.size();
        scalar_type max_val = 0;
        for(size_t j=0;j<sz_l;j++)
        {
            max_val = (max_val<boost::multiprecision::abs(x[j]))?boost::multiprecision::abs(x[j]):max_val;
        }
        return max_val;
    }
    scalar_type norm_l_inf(const vector_type& x) const override
    {
        return norm_inf(x);
    }

    scalar_type norm2_sq(const vector_type& x)const override
    {
        return norm_sq(x);
    }

    scalar_type norm_l2_sq(const vector_type& x) const override
    {
        return norm2_sq(x);
    }

    scalar_type norm_rank1(const vector_type& x, const scalar_type val_x) const
    {
        return common::scalar_math::sqrt(norm_sq(x) + val_x*val_x);
    }

    scalar_type norm_rank1_l2(const vector_type& x, const scalar_type val_x) const
    {
        return norm_rank1(x, val_x)/get_l2_size();
    }

    scalar_type norm2(const vector_type& x) const override
    {
        return norm_l2(x);
    }

    scalar_type sum(const vector_type &x) const override
    {
        size_t sz_l = x.size();
        ensure_helper_size(sz_l);
        for(size_t j=0;j<sz_l;j++)
        {
            helper_vector_[j] = x[j];
        }
        return sum_helper(sz_l);
    }
    scalar_type asum(const vector_type &x) const override
    {
        size_t sz_l = x.size();
        ensure_helper_size(sz_l);
        for(size_t j=0;j<sz_l;j++)
        {
            helper_vector_[j] = boost::multiprecision::abs(x[j]);
        }
        return sum_helper(sz_l);
    }

    scalar_type norm1(const vector_type& x) const override
    {
        return asum(x);
    }

    scalar_type norm_l1(const vector_type& x) const override
    {
        return norm1(x);
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

    T* view(vector_type& x) const
    {
        return x.data();
    }

    const T* view(const vector_type& x) const
    {
        return x.data();
    }

    void set(vector_type&) const
    {
    }

    void set(const T* host, vector_type& x) const
    {
        set(host, x, x.size());
    }

    void set(const T* host, vector_type& x, size_t n) const
    {
        if(n > x.size())
        {
            throw std::logic_error("cpu_vector_operations_var_prec::set: input size is larger than destination vector");
        }
        std::copy(host, host + n, x.begin());
    }

    void get(const vector_type& x, T* host) const
    {
        get(x, host, x.size());
    }

    void get(const vector_type& x, T* host, size_t n) const
    {
        if(n > x.size())
        {
            throw std::logic_error("cpu_vector_operations_var_prec::get: input size is larger than source vector");
        }
        std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n), host);
    }

    //calc: x := <vector_type with all elements equal to given scalar value> 
    void assign_scalar(const scalar_type scalar, vector_type& x)const override
    {
        size_t sz_l = x.size();
        for (size_t i = 0;i<sz_l;++i) 
            x[i] = scalar;
    }
    //calc: x := mul_x*x + <vector_type of all scalar value> 
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)const override
    {
        size_t sz_l = x.size();
        for (size_t i = 0;i < sz_l;++i) 
            x[i] = mul_x*x[i] + scalar;
    }
    void scale(scalar_type scale, vector_type &x)const override
    {
        add_mul_scalar(static_cast<scalar_type>(0.0), scale, x);
    }
    //copy: y := x
    void assign(const vector_type& x, vector_type& y)const override
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

    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y)const override
    {
        assign_mul(mul_x, x, y);
    }
    
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, vector_type& z)const
    {
        if((x.size() != y.size() )||(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations_var_prec::assign_mul: incorrect vector sizes provided");
        }  
        for (int i = 0;i < x.size();++i) 
            z[i] = mul_x*x[i] + mul_y*y[i];
    }

    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z)const override
    {
        assign_mul(mul_x, x, mul_y, y, z);
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

    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y)const override
    {
        add_mul(mul_x, x, mul_y, y);
    }

    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                            scalar_type mul_z, vector_type& z)const
    {
        if((x.size() != y.size() )||(x.size() != z.size() ))
        {
            throw std::logic_error("cpu_vector_operations_var_prec::add_mul: incorrect vector sizes provided");
        }         
        for (int i = 0;i < x.size();++i) 
            z[i] = mul_x*x[i] + mul_y*y[i] + mul_z*z[i];
    }

    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, const scalar_type mul_z, vector_type& z)const
    {
        add_mul(mul_x, x, mul_y, y, mul_z, z);
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
        if((x.size() != y.size() )||(x.size() != z.size() ))
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
        if((x.size() != y.size() )||(x.size() != z.size() ))
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
            x[j] /= mul_y*y[j];
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
