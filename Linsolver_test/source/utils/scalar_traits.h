// Copyright © 2016,2017 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCALAR_TRAITS_H__
#define __SCALAR_TRAITS_H__

#include <cmath>
#include <algorithm>

#ifndef __CUDACC__
#define __DEVICE_TAG__
#else
#define __DEVICE_TAG__ __device__ __host__
#endif

template<class T>
struct scalar_traits_err
{
    inline __DEVICE_TAG__ static T not_defined() { return T::this_type_is_missing_a_specialization(); };
};

template<class T>
struct scalar_traits
{
    static const T                  zero = T(0.f);
    //inline __DEVICE_TAG__ static T  zero() { return T(0.f); }
    //inline __DEVICE_TAG__ static T  one() { return T(1.f); }
    inline __DEVICE_TAG__ static T  pi() { return T(3.1415926535897932384626433832795f); }
    inline __DEVICE_TAG__ static T  sqrt(const T &x) { return std::sqrt(x); }
    inline __DEVICE_TAG__ static T  abs(const T &x) { return std::abs(x); }
    inline __DEVICE_TAG__ static T  sqr(const T &x) { return x*x; }
    inline __DEVICE_TAG__ static T  iconst(const int &i) { return T(i); }
    inline __DEVICE_TAG__ static T  fconst(const float &f) { return T(f); }
    inline __DEVICE_TAG__ static T  dconst(const double &d) { return T(d); }
    inline __DEVICE_TAG__ static T  max(const T &x,const T &y) { return std::max(x,y); }
    inline __DEVICE_TAG__ static T  min(const T &x,const T &y) { return std::min(x,y); }
    inline static std::string       name() { return scalar_traits_err<T>::not_defined(); }
};

template<>
struct scalar_traits<float>
{
    static const float                  zero = 0.f;
    //inline __DEVICE_TAG__ static float  zero() { return 0.f; }
    //inline __DEVICE_TAG__ static float  one() { return 1.f; }
    inline __DEVICE_TAG__ static float  pi() { return 3.1415926535897932384626433832795f; }
    inline __DEVICE_TAG__ static float  sqrt(const float &x) 
    { 
#ifndef __CUDA_ARCH__
        return std::sqrt(x);
#else
        return ::sqrtf(x);
#endif
    }
    inline __DEVICE_TAG__ static float  abs(const float &x) 
    { 
#ifndef __CUDA_ARCH__
        return std::abs(x); 
#else
        return ::fabsf(x);
#endif
    }
    inline __DEVICE_TAG__ static float  sqr(const float &x) { return x*x; }
    inline __DEVICE_TAG__ static float  iconst(const int &i) { return float(i); }
    inline __DEVICE_TAG__ static float  fconst(const float &f) { return f; }
    inline __DEVICE_TAG__ static float  dconst(const double &d) { return float(d); }
    inline __DEVICE_TAG__ static float  max(const float &x,const float &y)
    {
#ifndef __CUDA_ARCH__
        return std::max(x,y);
#else
        return ::fmaxf(x,y);      
#endif
    }
    inline __DEVICE_TAG__ static float  min(const float &x,const float &y)
    {
#ifndef __CUDA_ARCH__
        return std::min(x,y);
#else
        return ::fminf(x,y);      
#endif
    }
    //inline static std::string       name() { return scalar_traits_err<float>::not_defined(); }
};

template<>
struct scalar_traits<double>
{
    static const double                  zero = 0.;
    //inline __DEVICE_TAG__ static double  zero() { return 0.; }
    //inline __DEVICE_TAG__ static double  one() { return 1.; }
    inline __DEVICE_TAG__ static double  pi() { return 3.1415926535897932384626433832795; }
    inline __DEVICE_TAG__ static double  sqrt(const double &x) 
    { 
#ifndef __CUDA_ARCH__
        return std::sqrt(x);
#else
        return ::sqrt(x);
#endif
    }
    inline __DEVICE_TAG__ static double  abs(const double &x) 
    { 
#ifndef __CUDA_ARCH__
        return std::abs(x); 
#else
        return ::fabs(x);
#endif
    }
    inline __DEVICE_TAG__ static double  sqr(const double &x) { return x*x; }
    inline __DEVICE_TAG__ static double  iconst(const int &i) { return double(i); }
    inline __DEVICE_TAG__ static double  fconst(const double &f) { return f; }
    inline __DEVICE_TAG__ static double  dconst(const double &d) { return double(d); }
    inline __DEVICE_TAG__ static double  max(const double &x,const double &y)
    {
#ifndef __CUDA_ARCH__
        return std::max(x,y);
#else
        return ::fmax(x,y);      
#endif
    }
    inline __DEVICE_TAG__ static double  min(const double &x,const double &y)
    {
#ifndef __CUDA_ARCH__
        return std::min(x,y);
#else
        return ::fmin(x,y);      
#endif
    }
    //inline static std::string       name() { return scalar_traits_err<double>::not_defined(); }
};

#endif
