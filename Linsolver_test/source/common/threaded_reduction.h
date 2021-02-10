#ifndef __THREADED_REDUCTION_H__
#define __THREADED_REDUCTION_H__


#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <iostream>

template<class T, class T_vec>
class threaded_reduction
{
private:
    using accumulators = std::pair< T, T >;
public:
    threaded_reduction(size_t vec_size_, int n_ = -1, int use_high_precision_ = 0, T initial_ = T(0.0)):
    use_high_precision(use_high_precision_),
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

    void use_high_prec()
    {
        use_high_precision = 1;
    }
    void use_normal_prec()
    {
        use_high_precision = 0;
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
                partial_sum += x_[i] * y_[i];
            }
            
            std::lock_guard<std::mutex> lock(*g_lock_);
            result_ = result_ + partial_sum;
        }    
    private:
        int begin;
        int end;

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

        T two_prod(T &t, T a, T b) const // [1], pdf: 71, 169, 198, 
        {
            T p = a*b;
            t = std::fma(a, b, -p);
            return p;
        }

        T two_sum(T &t, T a, T b) const
        {
            T s = a+b;
            T bs = s-a;
            T as = s-bs;
            t = (b-bs) + (a-as);
            return s;
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
            
            if(use_high_precision == 0)
            {
                threads.push_back( std::thread( std::ref(dot_naive[j]),  std::ref(x_),  std::ref(y_),  std::ref(result_dot), g_lock_dot) );
            }
            else if(use_high_precision == 1)
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

        if(use_high_precision == 0)
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
            
            if(use_high_precision == 0)
            {
                threads.push_back( std::thread( std::ref(sum_naive[j]),  std::ref(x_),   std::ref(result_sum), g_lock_sum, use_abs_) );
            }
            else if(use_high_precision == 1)
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

        if(use_high_precision == 0)
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
    int use_high_precision;
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