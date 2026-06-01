#ifndef __COMMON_SCFD_BACKEND_EXT_COMPLEX_H__
#define __COMMON_SCFD_BACKEND_EXT_COMPLEX_H__

#include <complex>
#include <type_traits>

#if __has_include(<thrust/complex.h>)
#include <thrust/complex.h>
#define COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX 1
#endif

namespace scfd
{
namespace backend
{
struct cuda;
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

template<class Backend, class Real>
using complex_t = typename complex<Backend, Real>::type;

} // namespace scfd_backend_ext
} // namespace common

#endif
