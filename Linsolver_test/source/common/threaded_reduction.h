#ifndef __THREADED_REDUCTION_H__
#define __THREADED_REDUCTION_H__


#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <iostream>
#include <thrust/complex.h>
#include <common/complex_real_type_cast.hpp>
#include <type_traits>

template<class T, class T_vec>
class threaded_reduction
{
private:
    using accumulators = std::pair< T, T >;
    using TR = typename deduce_real_type_from_complex::recast_type<T>::real;

public:
    threaded_reduction(size_t vec_size_, int n_ = -1, int use_high_prec_ = 0, T initial_ = T(0.0)):
    use_high_prec(use_high_prec_),
    parts(n_),
    vec_size(vec_size_)
    {
        result_dot = T(0.0);
        result_sum = T(0.0);
        
        sigma_dot.first = T(0.0);
        sigma_dot.second = T(0.0);
        sigma_sum.first = T(0.0);
        sigma_sum.second = T(0.0);

        if(parts<1)
        {
            parts = std::thread::hardware_concurrency();
        }
        bounds();
        for(int j=0; j<parts; j++)
        {
            
            dot_naive.emplace_back(array_bounds[j], array_bounds[j+1]);
            dot_ogita.emplace_back(array_bounds[j], array_bounds[j+1]);
            
            sum_naive.emplace_back(array_bounds[j], array_bounds[j+1]);
            sum_ogita.emplace_back(array_bounds[j], array_bounds[j+1]);

        }
        g_lock_dot = std::make_shared<std::mutex>();
        g_lock_sum = std::make_shared<std::mutex>();

    }
    ~threaded_reduction()
    {

    }

    void use_high_precision()
    {
        use_high_prec = 1;
    }
    void use_standard_precision()
    {
        use_high_prec = 0;
    }

private:
    class sum_naive
    {
    public:
        sum_naive(int begin_, int end_): begin(begin_), end(end_){}
        ~sum_naive(){}
        
        void operator ()(const T_vec& x_, T& result_, std::shared_ptr<std::mutex> g_lock_, bool use_abs_)
        {

            T partial_sum = 0;
            if(use_abs_)
            {
                for(int i = begin; i < end; ++i)
                {
                    partial_sum += std::abs(x_[i]);
                }
            }
            else
            {
                for(int i = begin; i < end; ++i)
                {
                    partial_sum += x_[i];
                }
            }
            
            std::lock_guard<std::mutex> lock(*g_lock_);
            result_ = result_ + partial_sum;
        }    
    private:
        int begin;
        int end;

    };

    class sum_ogita
    {
    public:
        sum_ogita(int begin_, int end_): begin(begin_), end(end_){}
        ~sum_ogita(){}
        
        void operator ()(const T_vec& x_, T& result_, accumulators& sigma_, std::shared_ptr<std::mutex> g_lock_, bool use_abs_)
        {

            T s = T(0.0), c = T(0.0), p = T(0.0);
            t = T(0.0);
            T qq = T(0.0);
            if(use_abs_)
            {
                for (int j=begin; j<end; j++) 
                {
                    T x_l = std::abs(x_[j]);
                    s = two_sum(t, s, x_l);
                    c = c  + t;
                }
            }
            else
            {
                for (int j=begin; j<end; j++) 
                {
                    s = two_sum(t, s, x_[j]);
                    c = c  + t;
                }
            }

            std::lock_guard<std::mutex> lock(*g_lock_);
            
            result_ = two_sum(t, T(result_), s);
            sigma_.first = two_sum(qq, T(sigma_.first), c);
            sigma_.second = T(sigma_.second) + t + qq;

        }    
    private:
        int begin;
        int end;
        mutable T t = T(0.0);

        T two_sum(T &t, T a, T b) const
        {
            T s = a+b;
            T bs = s-a;
            T as = s-bs;
            t = (b-bs) + (a-as);
            return s;
        }
    };



    class dot_product_naive
    {
    public:
        dot_product_naive(int begin_, int end_): begin(begin_), end(end_){}
        ~dot_product_naive(){}
        
        void operator ()(const T_vec& x_, const T_vec& y_, T& result_, std::shared_ptr<std::mutex> g_lock_)
        {

            T partial_sum = 0;
            for(int i = begin; i < end; ++i)
            {
                partial_sum += two_prod(x_[i],y_[i]);
            }
            
            std::lock_guard<std::mutex> lock(*g_lock_);
            result_ = result_ + partial_sum;
        }    
    private:
        int begin;
        int end;
        //trick with D to specilize a templated function inside the class scope
        template<class T_l, bool D = true>
        T_l two_prod(T_l a, T_l b) const
        {
            
            return(a*b);
        }
        //trick with D to specilize a templated function inside the class scope
        template<bool D = true>
        thrust::complex<TR> two_prod(thrust::complex<TR> a, thrust::complex<TR> b) const
        {   
            return(conj(a)*b);
        }

    };


    class dot_product_ogita
    {
    public:
        dot_product_ogita(int begin_, int end_): begin(begin_), end(end_){}
        ~dot_product_ogita(){}
        
        void operator ()(const T_vec& x_, const T_vec& y_, T& result_, accumulators& sigma_, std::shared_ptr<std::mutex> g_lock_)
        {

            T s = T(0.0), c = T(0.0), p = T(0.0);
            pi = T(0.0);
            t = T(0.0);
            T qq = T(0.0);
            for (int j=begin; j<end; j++) 
            {
                p = two_prod(pi, x_[j], y_[j]);
                s = two_sum(t, s, p);
                c = c + pi + t;
            }
            
            std::lock_guard<std::mutex> lock(*g_lock_);
            
            result_ = two_sum(t, T(result_), s);
            sigma_.first = two_sum(qq, T(sigma_.first), c);
            sigma_.second = T(sigma_.second) + t + qq;

        }    
    private:
        int begin;
        int end;
        mutable T pi = T(0.0);
        mutable T t = T(0.0);

        //trick with D to specilize a templated function inside the class scope
        template<class T_l, bool D = true>
        T_l two_prod(T_l &t, T_l a, T_l b) const // [1], pdf: 71, 169, 198, 
        {
            T_l p = a*b;
            t = std::fma(a, b, -p);
            return p;
        }

        template<class T_l>
        T_l two_sum(T_l &t, T_l a, T_l b) const
        {
            T_l s = a+b;
            T_l bs = s-a;
            T_l as = s-bs;
            t = (b-bs) + (a-as);
            return s;
        }
        
        //trick with D to specilize a templated function inside the class scope
        template<bool D = true>
        thrust::complex<TR> two_prod(thrust::complex<TR> &t, thrust::complex<TR> a, thrust::complex<TR> b) const 
        {
            using T_real = TR;
            using TC = typename thrust::complex<T_real>;         
            T_real a_R = a.real();
            T_real a_I = a.imag();
            T_real b_R = b.real();
            T_real b_I = b.imag();

            T_real p_R1 = a_R*b_R;
            T_real t_R1 = std::fma<T_real>(a_R, b_R, -p_R1);
            T_real p_R2 = a_I*b_I;
            T_real t_R2 = std::fma<T_real>(a_I, b_I, -p_R2);
            T_real p_I1 = a_R*b_I;
            T_real t_I1 = std::fma<T_real>(a_R, b_I, -p_I1);
            T_real p_I2 = -a_I*b_R;
            T_real t_I2 = std::fma<T_real>(-a_I, b_R, -p_I2);

            T_real t1 = T_real(0.0);
            T_real t2 = T_real(0.0);
            T_real p_R = two_sum<T_real>(t1, p_R1, p_R2);
            T_real p_I = two_sum<T_real>(t2, p_I1, p_I2);
            
            TC p = TC(p_R, p_I);
            t = TC(t_R1 + t_R2 + t1, t_I1 + t_I2 + t2);

            return p;             
        }
    };



public:
    T dot(const T_vec& x_, const T_vec& y_) //const
    {   
        result_dot = T(0.0);
        sigma_dot.first = T(0.0);
        sigma_dot.second = T(0.0);

        std::vector<std::thread> threads;
        threads.reserve(parts);

        for (int j = 0; j < parts; j++) 
        {
            
            if(use_high_prec == 0)
            {
                threads.push_back( std::thread( std::ref(dot_naive[j]),  std::ref(x_),  std::ref(y_),  std::ref(result_dot), g_lock_dot) );
            }
            else if(use_high_prec == 1)
            {
                threads.push_back( std::thread( std::ref(dot_ogita[j]),  std::ref(x_),  std::ref(y_),  std::ref(result_dot), std::ref(sigma_dot), g_lock_dot) );
            }
            else
            {
                throw std::logic_error("Incorrect dot product scheme selected");
            }

            
        }

        for(auto &t : threads)
        {
            if(t.joinable())
                t.join();
        }

        if(use_high_prec == 0)
        {
            return T(result_dot);
        }
        else
        {
            return T(result_dot) + T(sigma_dot.first) + T(sigma_dot.second);
        }
    }


    T sum(const T_vec& x_, bool use_abs_ = false) //const
    {   
        result_sum = T(0.0);
        sigma_sum.first = T(0.0);
        sigma_sum.second = T(0.0);

        std::vector<std::thread> threads;
        threads.reserve(parts);

        for (int j = 0; j < parts; j++) 
        {
            
            if(use_high_prec == 0)
            {
                threads.push_back( std::thread( std::ref(sum_naive[j]),  std::ref(x_),   std::ref(result_sum), g_lock_sum, use_abs_) );
            }
            else if(use_high_prec == 1)
            {
                threads.push_back( std::thread( std::ref(sum_ogita[j]),  std::ref(x_),   std::ref(result_sum), std::ref(sigma_sum), g_lock_sum, use_abs_) );
            }
            else
            {
                throw std::logic_error("Incorrect sum scheme selected");
            }

            
        }

        for(auto &t : threads)
        {
            if(t.joinable())
                t.join();
        }

        if(use_high_prec == 0)
        {
            return T(result_sum);
        }
        else
        {
            return T(result_sum) + T(sigma_sum.first) + T(sigma_sum.second);
        }
    }

    T asum(const T_vec& x_)
    {
        return( sum(x_, true) );   
    }

private:
    T result_dot;
    T result_sum;
    int use_high_prec;
    int parts;
    size_t vec_size;
    std::vector<int> array_bounds;
    std::vector<dot_product_naive> dot_naive;
    std::vector<dot_product_ogita> dot_ogita;
    std::vector<sum_naive> sum_naive;
    std::vector<sum_ogita> sum_ogita;    
    accumulators sigma_dot;
    accumulators sigma_sum;
    
    std::shared_ptr<std::mutex> g_lock_dot;
    std::shared_ptr<std::mutex> g_lock_sum;

    void bounds()
    {
        array_bounds.reserve(parts+1);
        int delta = vec_size / parts;
        int reminder = vec_size % parts;
        int N1 = 0, N2 = 0;
        array_bounds.push_back(N1);
        for (int j = 0; j < parts; j++) 
        {
            N2 = N1 + delta;
            if (j == parts - 1)
            {
                N2 += reminder;
            }
            array_bounds.push_back(N2);
            N1 = N2;
        }
    }


};





#endif