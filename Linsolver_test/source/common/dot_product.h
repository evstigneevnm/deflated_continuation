#ifndef __DOT_PRODUCT_H__
#define __DOT_PRODUCT_H__

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <thrust/complex.h>
#include <common/complex_real_type_cast.hpp>




template<class T, class T_vec>
class dot_product
{
private:
    size_t sz;
    int use_ogita = 0;


    mutable T pi = T(0.0);
    mutable T t = T(0.0);
    using TR = typename deduce_real_type_from_complex::recast_type<T>::real;


    template<class T_l>
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

    
public:
    dot_product(int use_high_precision_dot_product_ = 0):
    use_ogita(use_high_precision_dot_product_)
    {
    }
    
    ~dot_product()
    {
    }
  
    void use_high_prec()
    {
        use_ogita = 1;
    }
    void use_normal_prec()
    {
        use_ogita = 0;
    }

    
    template<class T_l>
    T_l prod_H(const T_l a, const T_l b) const
    {
        return(a*b);
    }

    T dot_naive(const size_t sz_l, const T_vec& X, const T_vec& Y) const
    {
        T s = T(0.0);
        for(size_t j=0; j<sz_l; j++)
        {
            s += prod_H<T>(X[j],Y[j]);
        }
        return s;
    }



    T dot_ogita(const size_t sz_l, const T_vec& X, const T_vec& Y) const
    {
        T s = T(0.0), c = T(0.0), p = T(0.0);
        pi = T(0.0);
        t = T(0.0);
        for (size_t j=0; j<sz_l; j++) 
        {
            p = two_prod<T>(pi, X[j], Y[j]);
            s = two_sum(t, s, p);
            c = c + pi + t;
        }
        return s+c;
    }

    T dot(const T_vec& X, const T_vec& Y) const
    {
        T res = T(0.0);
        auto sz = X.size();
        if(sz!=Y.size())
        {
            throw std::logic_error("dot product: vector sizes don't match.");
        }

        if(use_ogita == 1)
        {
            res = dot_ogita(sz, X,Y);
        }
        else if(use_ogita == 2)
        {
            //res = dot_ogita_parallel(X,Y);
        }
        else if(use_ogita == 0)
        {
            res = dot_naive(sz, X,Y);
        }
        else
        {
            throw std::logic_error("dot_product: selected incorrect dot product type.");
        }

        return(res);
    }

};



    template<>
    template<>
    thrust::complex<float>  dot_product<thrust::complex<float>, std::vector<thrust::complex<float>> >::two_prod<thrust::complex<float>>(thrust::complex<float> &t, thrust::complex<float> a, thrust::complex<float> b) const     
    {

        using T_real = float;
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

    template<>
    template<>
    thrust::complex<double>  dot_product<thrust::complex<double>, std::vector<thrust::complex<double>> >::two_prod(thrust::complex<double> &t, thrust::complex<double> a, thrust::complex<double> b) const     
    {

        using T_real = double;
        using TC = typename thrust::complex<T_real>;         
        T_real a_R = a.real();
        T_real a_I = a.imag();
        T_real b_R = b.real();
        T_real b_I = b.imag();

        T_real p_R1 = a_R*b_R;
        T_real t_R1 = std::fma(a_R, b_R, -p_R1);
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

    template<>
    template<>
    thrust::complex<float> dot_product<thrust::complex<float>, std::vector<thrust::complex<float>> >::prod_H(const thrust::complex<float> a, const thrust::complex<float> b) const
    {
        return(conj(a)*b);
    }

    template<>
    template<>
    thrust::complex<double> dot_product<thrust::complex<double>, std::vector<thrust::complex<double>> >::prod_H(const thrust::complex<double> a, const thrust::complex<double> b) const
    {
        return(conj(a)*b);
    }


    template<>
    template<>
    thrust::complex<float>  dot_product<thrust::complex<float>, thrust::complex<float>* >::two_prod<thrust::complex<float>>(thrust::complex<float> &t, thrust::complex<float> a, thrust::complex<float> b) const     
    {

        using T_real = float;
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

    template<>
    template<>
    thrust::complex<double>  dot_product<thrust::complex<double>, thrust::complex<double>* >::two_prod(thrust::complex<double> &t, thrust::complex<double> a, thrust::complex<double> b) const     
    {

        using T_real = double;
        using TC = typename thrust::complex<T_real>;         
        T_real a_R = a.real();
        T_real a_I = a.imag();
        T_real b_R = b.real();
        T_real b_I = b.imag();

        T_real p_R1 = a_R*b_R;
        T_real t_R1 = std::fma(a_R, b_R, -p_R1);
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

    template<>
    template<>
    thrust::complex<float> dot_product<thrust::complex<float>, thrust::complex<float>* >::prod_H(const thrust::complex<float> a, const thrust::complex<float> b) const
    {
        return(conj(a)*b);
    }

    template<>
    template<>
    thrust::complex<double> dot_product<thrust::complex<double>, thrust::complex<double>* >::prod_H(const thrust::complex<double> a, const thrust::complex<double> b) const
    {
        return(conj(a)*b);
    }



#endif