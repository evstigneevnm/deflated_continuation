#ifndef __DOT_PRODUCT_H__
#define __DOT_PRODUCT_H__

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>

template<class T>
class dot_product
{
private:
    size_t sz;
    int use_ogita = 0;

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

    
public:
    dot_product(size_t sz_, int use_high_precision_dot_product_ = 0):
    sz(sz_),
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

  
    T dot_naive(const T* X, const T* Y) const
    {
        T s = T(0.0);
        for(size_t j=0; j<sz; j++)
        {
            s += X[j]*Y[j];
        }
        return s;
    }



    T dot_ogita(const T* X, const T* Y) const
    {
        T s = T(0.0), c = T(0.0), p;
        pi = T(0.0);
        t = T(0.0);
        for (size_t j=0; j<sz; j++) 
        {
            p = two_prod(pi, X[j], Y[j]);
            s = two_sum(t, s, p);
            c = c + pi + t;
        }
        //std::cout << "s = " << s << "; c = " << c << std::endl;
        return s+c;
    }

    T dot(const T* X, const T* Y) const
    {
        T res = T(0.0);
        if(use_ogita == 1)
        {
            res = dot_ogita(X,Y);
        }
        else if(use_ogita == 2)
        {
            //res = dot_ogita_parallel(X,Y);
        }
        else if(use_ogita == 0)
        {
            res = dot_naive(X,Y);
        }
        else
        {
            throw std::logic_error("dot_product: selected incorrect dot product type.");
        }

        return(res);
    }

};

#endif