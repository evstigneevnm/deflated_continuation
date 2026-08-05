#ifndef __COMMON_SCFD_BACKEND_EXT_MATH_H__
#define __COMMON_SCFD_BACKEND_EXT_MATH_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

#include <scfd/utils/device_tag.h>

#include <common/scfd_backend_ext/complex.h>

namespace common
{
namespace scfd_backend_ext
{

template<class Backend, class T>
struct math
{
    __DEVICE_TAG__ static T zero()
    {
        return T(0);
    }

    __DEVICE_TAG__ static T abs(const T& x)
    {
        return x < T(0) ? -x : x;
    }

    __DEVICE_TAG__ static bool is_finite(const T& x)
    {
        return (x == x) && ((x - x) == T(0));
    }

    __DEVICE_TAG__ static T sqrt(const T& x)
    {
#if defined(__CUDA_ARCH__)
        return ::sqrt(x);
#else
        using std::sqrt;
        return sqrt(x);
#endif
    }

    __DEVICE_TAG__ static T sin(const T& x)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return ::sin(x);
#else
        using std::sin;
        return sin(x);
#endif
    }

    __DEVICE_TAG__ static T cos(const T& x)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return ::cos(x);
#else
        using std::cos;
        return cos(x);
#endif
    }
};

template<class Backend, class T>
struct scalar_traits
{
    using real_type = T;
    static constexpr bool is_complex = false;

    __DEVICE_TAG__ static T make(long double real, long double)
    {
        return static_cast<T>(real);
    }

    __DEVICE_TAG__ static T zero()
    {
        return T(0);
    }

    __DEVICE_TAG__ static T conj_mul(const T& x, const T& y)
    {
        return x*y;
    }

    __DEVICE_TAG__ static real_type norm_sq_term(const T& x)
    {
        return x*x;
    }

    __DEVICE_TAG__ static real_type asum_term(const T& x)
    {
        return math<Backend, T>::abs(x);
    }

    __DEVICE_TAG__ static bool is_finite(const T& x)
    {
        return math<Backend, T>::is_finite(x);
    }

    __DEVICE_TAG__ static real_type random_value(std::size_t i, std::size_t seed)
    {
        std::size_t v = (i + 1u)*(seed + 7919u);
        v ^= v >> 13u;
        v *= 1274126177u;
        return static_cast<real_type>(v % 100000u)/static_cast<real_type>(100000u);
    }

    __DEVICE_TAG__ static T random_scalar(std::size_t i, std::size_t seed)
    {
        return random_value(i, seed);
    }
};

template<class Backend, class T>
struct scalar_traits<Backend, std::complex<T>>
{
    using scalar_type = std::complex<T>;
    using real_type = T;
    static constexpr bool is_complex = true;

    static scalar_type make(long double real, long double imag)
    {
        return scalar_type(static_cast<T>(real), static_cast<T>(imag));
    }

    static scalar_type zero()
    {
        return scalar_type(T(0), T(0));
    }

    static scalar_type conj_mul(const scalar_type& x, const scalar_type& y)
    {
        return std::conj(x)*y;
    }

    static real_type norm_sq_term(const scalar_type& x)
    {
        return x.real()*x.real() + x.imag()*x.imag();
    }

    static real_type asum_term(const scalar_type& x)
    {
        return math<Backend, T>::abs(x.real()) + math<Backend, T>::abs(x.imag());
    }

    static bool is_finite(const scalar_type& x)
    {
        return math<Backend, T>::is_finite(x.real()) && math<Backend, T>::is_finite(x.imag());
    }

    static T random_value(std::size_t i, std::size_t seed)
    {
        return scalar_traits<Backend, T>::random_value(i, seed);
    }

    static scalar_type random_scalar(std::size_t i, std::size_t seed)
    {
        return scalar_type(random_value(i, seed), random_value(i + 17u, seed + 23u));
    }
};

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX
template<class Backend, class T>
struct scalar_traits<Backend, thrust::complex<T>>
{
    using scalar_type = thrust::complex<T>;
    using real_type = T;
    static constexpr bool is_complex = true;

    __DEVICE_TAG__ static scalar_type make(long double real, long double imag)
    {
        return scalar_type(static_cast<T>(real), static_cast<T>(imag));
    }

    __DEVICE_TAG__ static scalar_type zero()
    {
        return scalar_type(T(0), T(0));
    }

    __DEVICE_TAG__ static scalar_type conj_mul(const scalar_type& x, const scalar_type& y)
    {
        return thrust::conj(x)*y;
    }

    __DEVICE_TAG__ static real_type norm_sq_term(const scalar_type& x)
    {
        return x.real()*x.real() + x.imag()*x.imag();
    }

    __DEVICE_TAG__ static real_type asum_term(const scalar_type& x)
    {
        return math<Backend, T>::abs(x.real()) + math<Backend, T>::abs(x.imag());
    }

    __DEVICE_TAG__ static bool is_finite(const scalar_type& x)
    {
        return math<Backend, T>::is_finite(x.real()) && math<Backend, T>::is_finite(x.imag());
    }

    __DEVICE_TAG__ static T random_value(std::size_t i, std::size_t seed)
    {
        return scalar_traits<Backend, T>::random_value(i, seed);
    }

    __DEVICE_TAG__ static scalar_type random_scalar(std::size_t i, std::size_t seed)
    {
        return scalar_type(random_value(i, seed), random_value(i + 17u, seed + 23u));
    }
};
#endif

template<class T>
struct max_op
{
    __DEVICE_TAG__ T operator()(const T& a, const T& b) const
    {
        return a > b ? a : b;
    }
};

} // namespace scfd_backend_ext
} // namespace common

#endif
