#ifndef __COMMON_SCFD_BACKEND_EXT_COMPLEX_H__
#define __COMMON_SCFD_BACKEND_EXT_COMPLEX_H__

#include <complex>
#include <type_traits>

#include <scfd/utils/device_tag.h>

#if __has_include(<thrust/complex.h>)
#include <thrust/complex.h>
#define COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX 1
#endif

#if __has_include(<cufft.h>)
#include <cufft.h>
#define COMMON_SCFD_BACKEND_EXT_HAS_CUFFT_COMPLEX 1
#endif

namespace scfd
{
namespace backend
{
struct cuda;
struct hip;
struct omp;
struct serial_cpu;
}
}

namespace common
{
namespace scfd_backend_ext
{

template<class Backend, class Real>
struct complex
{
    using type = std::complex<Real>;
};

template<class>
struct dependent_false : std::false_type
{
};

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX
template<class Real>
struct complex<scfd::backend::cuda, Real>
{
    using type = thrust::complex<Real>;
};
#else
template<class Real>
struct complex<scfd::backend::cuda, Real>
{
    static_assert(dependent_false<Real>::value, "CUDA complex type requires thrust::complex.");
};
#endif

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX
template<class Real>
struct complex<scfd::backend::hip, Real>
{
    using type = thrust::complex<Real>;
};
#else
template<class Real>
struct complex<scfd::backend::hip, Real>
{
    static_assert(dependent_false<Real>::value, "HIP complex type requires thrust::complex.");
};
#endif

template<class Backend, class Real>
using complex_t = typename complex<Backend, Real>::type;

template<class T>
struct is_complex_type : std::false_type
{
};

template<class Complex>
struct complex_value_traits
{
    static_assert(dependent_false<Complex>::value, "Unsupported complex type.");
};

template<class Real>
struct is_complex_type<std::complex<Real>> : std::true_type
{
};

template<class Real>
struct complex_value_traits<std::complex<Real>>
{
    using complex_type = std::complex<Real>;
    using real_type = Real;

    __DEVICE_TAG__ static complex_type make(const real_type real, const real_type imag)
    {
        return complex_type(real, imag);
    }

    __DEVICE_TAG__ static complex_type from_real(const real_type real)
    {
        return complex_type(real, real_type(0));
    }

    __DEVICE_TAG__ static real_type real(const complex_type& value)
    {
        return value.real();
    }

    __DEVICE_TAG__ static real_type imag(const complex_type& value)
    {
        return value.imag();
    }

    __DEVICE_TAG__ static complex_type conj(const complex_type& value)
    {
        return complex_type(value.real(), -value.imag());
    }

    __DEVICE_TAG__ static complex_type add(const complex_type& left, const complex_type& right)
    {
        return left + right;
    }

    __DEVICE_TAG__ static complex_type mul(const complex_type& left, const complex_type& right)
    {
        return left * right;
    }

    __DEVICE_TAG__ static real_type abs_sq(const complex_type& value)
    {
        return value.real()*value.real() + value.imag()*value.imag();
    }
};

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX
template<class Real>
struct is_complex_type<thrust::complex<Real>> : std::true_type
{
};

template<class Real>
struct complex_value_traits<thrust::complex<Real>>
{
    using complex_type = thrust::complex<Real>;
    using real_type = Real;

    __DEVICE_TAG__ static complex_type make(const real_type real, const real_type imag)
    {
        return complex_type(real, imag);
    }

    __DEVICE_TAG__ static complex_type from_real(const real_type real)
    {
        return complex_type(real, real_type(0));
    }

    __DEVICE_TAG__ static real_type real(const complex_type& value)
    {
        return value.real();
    }

    __DEVICE_TAG__ static real_type imag(const complex_type& value)
    {
        return value.imag();
    }

    __DEVICE_TAG__ static complex_type conj(const complex_type& value)
    {
        return complex_type(value.real(), -value.imag());
    }

    __DEVICE_TAG__ static complex_type add(const complex_type& left, const complex_type& right)
    {
        return left + right;
    }

    __DEVICE_TAG__ static complex_type mul(const complex_type& left, const complex_type& right)
    {
        return left * right;
    }

    __DEVICE_TAG__ static real_type abs_sq(const complex_type& value)
    {
        return value.real()*value.real() + value.imag()*value.imag();
    }
};
#endif

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_CUFFT_COMPLEX
template<>
struct is_complex_type<cufftComplex> : std::true_type
{
};

template<>
struct complex_value_traits<cufftComplex>
{
    using complex_type = cufftComplex;
    using real_type = float;

    __DEVICE_TAG__ static complex_type make(const real_type real, const real_type imag)
    {
        complex_type value;
        value.x = real;
        value.y = imag;
        return value;
    }

    __DEVICE_TAG__ static complex_type from_real(const real_type real)
    {
        return make(real, real_type(0));
    }

    __DEVICE_TAG__ static real_type real(const complex_type& value)
    {
        return value.x;
    }

    __DEVICE_TAG__ static real_type imag(const complex_type& value)
    {
        return value.y;
    }

    __DEVICE_TAG__ static complex_type conj(const complex_type& value)
    {
        return make(value.x, -value.y);
    }

    __DEVICE_TAG__ static complex_type add(const complex_type& left, const complex_type& right)
    {
        return make(left.x + right.x, left.y + right.y);
    }

    __DEVICE_TAG__ static complex_type mul(const complex_type& left, const complex_type& right)
    {
        return make(left.x*right.x - left.y*right.y, left.x*right.y + left.y*right.x);
    }

    __DEVICE_TAG__ static real_type abs_sq(const complex_type& value)
    {
        return value.x*value.x + value.y*value.y;
    }
};

template<>
struct is_complex_type<cufftDoubleComplex> : std::true_type
{
};

template<>
struct complex_value_traits<cufftDoubleComplex>
{
    using complex_type = cufftDoubleComplex;
    using real_type = double;

    __DEVICE_TAG__ static complex_type make(const real_type real, const real_type imag)
    {
        complex_type value;
        value.x = real;
        value.y = imag;
        return value;
    }

    __DEVICE_TAG__ static complex_type from_real(const real_type real)
    {
        return make(real, real_type(0));
    }

    __DEVICE_TAG__ static real_type real(const complex_type& value)
    {
        return value.x;
    }

    __DEVICE_TAG__ static real_type imag(const complex_type& value)
    {
        return value.y;
    }

    __DEVICE_TAG__ static complex_type conj(const complex_type& value)
    {
        return make(value.x, -value.y);
    }

    __DEVICE_TAG__ static complex_type add(const complex_type& left, const complex_type& right)
    {
        return make(left.x + right.x, left.y + right.y);
    }

    __DEVICE_TAG__ static complex_type mul(const complex_type& left, const complex_type& right)
    {
        return make(left.x*right.x - left.y*right.y, left.x*right.y + left.y*right.x);
    }

    __DEVICE_TAG__ static real_type abs_sq(const complex_type& value)
    {
        return value.x*value.x + value.y*value.y;
    }
};
#endif

template<class Complex>
using complex_real_t = typename complex_value_traits<Complex>::real_type;

} // namespace scfd_backend_ext
} // namespace common

#endif
