#ifndef __NMFD_OPERATIONS_BLAS1_HIGH_PRECISION_COMPENSATED_REDUCTION_H__
#define __NMFD_OPERATIONS_BLAS1_HIGH_PRECISION_COMPENSATED_REDUCTION_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <common/scfd_backend_ext/complex.h>

#if defined(NMFD_HIGH_PRECISION_BLAS1_ENABLE_HIP_NVIDIA) || (defined(__HIPCC__) && (defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)))
#define NMFD_HIGH_PRECISION_BLAS1_HIP_USES_CUDA_OGITA 1
#endif

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <memory>
#include <nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita.h>
#elif defined(NMFD_HIGH_PRECISION_BLAS1_HIP_USES_CUDA_OGITA)
#include <memory>
#include <nmfd/operations/blas1/high_precision/hip/gpu_reduction_ogita.h>
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

namespace nmfd
{
namespace operations
{
namespace blas1
{
namespace high_precision
{
namespace detail
{

template<class T>
inline T abs_value(const T& x)
{
    return x < T(0) ? -x : x;
}

template<class T>
inline T two_sum(const T& a, const T& b, T& err)
{
    const T s = a + b;
    const T bs = s - a;
    const T as = s - bs;
    err = (b - bs) + (a - as);
    return s;
}

template<class T>
inline T two_prod(const T& a, const T& b, T& err)
{
    const T p = a*b;
    err = std::fma(a, b, -p);
    return p;
}

template<class T>
struct arithmetic
{
    using scalar_type = T;
    using real_type = T;

    static scalar_type zero()
    {
        return scalar_type(0);
    }

    static real_type real_part(const scalar_type& x)
    {
        return x;
    }

    static real_type asum_term(const scalar_type& x)
    {
        return abs_value(x);
    }

    static scalar_type conj_product(const scalar_type& x, const scalar_type& y, scalar_type& err)
    {
        return two_prod(x, y, err);
    }

    static real_type norm_sq_term(const scalar_type& x, real_type& err)
    {
        return two_prod(x, x, err);
    }
};

template<class T>
struct arithmetic<std::complex<T>>
{
    using scalar_type = std::complex<T>;
    using real_type = T;

    static scalar_type zero()
    {
        return scalar_type(T(0), T(0));
    }

    static real_type real_part(const scalar_type& x)
    {
        return x.real();
    }

    static real_type asum_term(const scalar_type& x)
    {
        return abs_value(x.real()) + abs_value(x.imag());
    }

    static scalar_type conj_product(const scalar_type& x, const scalar_type& y, scalar_type& err)
    {
        T e_rr;
        T e_ii;
        T e_ri;
        T e_ir;
        T e_rsum;
        T e_isum;

        const T rr = two_prod(x.real(), y.real(), e_rr);
        const T ii = two_prod(x.imag(), y.imag(), e_ii);
        const T ri = two_prod(x.real(), y.imag(), e_ri);
        const T ir = two_prod(-x.imag(), y.real(), e_ir);

        const T real_sum = two_sum(rr, ii, e_rsum);
        const T imag_sum = two_sum(ri, ir, e_isum);
        err = scalar_type(e_rr + e_ii + e_rsum, e_ri + e_ir + e_isum);
        return scalar_type(real_sum, imag_sum);
    }

    static real_type norm_sq_term(const scalar_type& x, real_type& err)
    {
        T e_real;
        T e_imag;
        T e_sum;
        const T real_sq = two_prod(x.real(), x.real(), e_real);
        const T imag_sq = two_prod(x.imag(), x.imag(), e_imag);
        const T sum = two_sum(real_sq, imag_sq, e_sum);
        err = e_real + e_imag + e_sum;
        return sum;
    }
};

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX
template<class T>
struct arithmetic<thrust::complex<T>>
{
    using scalar_type = thrust::complex<T>;
    using real_type = T;

    static scalar_type zero()
    {
        return scalar_type(T(0), T(0));
    }

    static real_type real_part(const scalar_type& x)
    {
        return x.real();
    }

    static real_type asum_term(const scalar_type& x)
    {
        return abs_value(x.real()) + abs_value(x.imag());
    }

    static scalar_type conj_product(const scalar_type& x, const scalar_type& y, scalar_type& err)
    {
        T e_rr;
        T e_ii;
        T e_ri;
        T e_ir;
        T e_rsum;
        T e_isum;

        const T rr = two_prod(x.real(), y.real(), e_rr);
        const T ii = two_prod(x.imag(), y.imag(), e_ii);
        const T ri = two_prod(x.real(), y.imag(), e_ri);
        const T ir = two_prod(-x.imag(), y.real(), e_ir);

        const T real_sum = two_sum(rr, ii, e_rsum);
        const T imag_sum = two_sum(ri, ir, e_isum);
        err = scalar_type(e_rr + e_ii + e_rsum, e_ri + e_ir + e_isum);
        return scalar_type(real_sum, imag_sum);
    }

    static real_type norm_sq_term(const scalar_type& x, real_type& err)
    {
        T e_real;
        T e_imag;
        T e_sum;
        const T real_sq = two_prod(x.real(), x.real(), e_real);
        const T imag_sq = two_prod(x.imag(), x.imag(), e_imag);
        const T sum = two_sum(real_sq, imag_sq, e_sum);
        err = e_real + e_imag + e_sum;
        return sum;
    }
};
#endif

template<class T>
class compensated_accumulator
{
public:
    using arithmetic_type = arithmetic<T>;

    void add(const T& value)
    {
        T err;
        sum_ = two_sum(sum_, value, err);
        corr_ += err;
    }

    void add_with_error(const T& value, const T& value_error)
    {
        add(value);
        add(value_error);
    }

    void merge(const compensated_accumulator& other)
    {
        add(other.sum_);
        add(other.corr_);
    }

    T result() const
    {
        return sum_ + corr_;
    }

private:
    T sum_ = arithmetic_type::zero();
    T corr_ = arithmetic_type::zero();
};

template<class T, class Ordinal>
class serial_compensated_reduction_impl
{
public:
    using arithmetic_type = arithmetic<T>;
    using real_type = typename arithmetic_type::real_type;

    explicit serial_compensated_reduction_impl(Ordinal)
    {
    }

    T sum(Ordinal n, const T* x) const
    {
        compensated_accumulator<T> acc;
        for(Ordinal i = 0; i < n; ++i)
        {
            acc.add(x[i]);
        }
        return acc.result();
    }

    real_type asum(Ordinal n, const T* x) const
    {
        compensated_accumulator<real_type> acc;
        for(Ordinal i = 0; i < n; ++i)
        {
            acc.add(arithmetic_type::asum_term(x[i]));
        }
        return acc.result();
    }

    T dot(Ordinal n, const T* x, const T* y) const
    {
        compensated_accumulator<T> acc;
        for(Ordinal i = 0; i < n; ++i)
        {
            T err;
            const T term = arithmetic_type::conj_product(x[i], y[i], err);
            acc.add_with_error(term, err);
        }
        return acc.result();
    }

    real_type norm_sq(Ordinal n, const T* x) const
    {
        compensated_accumulator<real_type> acc;
        for(Ordinal i = 0; i < n; ++i)
        {
            real_type err;
            const real_type term = arithmetic_type::norm_sq_term(x[i], err);
            acc.add_with_error(term, err);
        }
        return acc.result();
    }
};

template<class T, class Ordinal>
class omp_compensated_reduction_impl
{
public:
    using arithmetic_type = arithmetic<T>;
    using real_type = typename arithmetic_type::real_type;

    explicit omp_compensated_reduction_impl(Ordinal size):
        serial_fallback_(size)
    {
    }

    T sum(Ordinal n, const T* x) const
    {
#ifdef _OPENMP
        std::vector<compensated_accumulator<T>> partials(static_cast<std::size_t>(omp_get_max_threads()));
#pragma omp parallel
        {
            compensated_accumulator<T> local;
#pragma omp for schedule(static)
            for(Ordinal i = 0; i < n; ++i)
            {
                local.add(x[i]);
            }
            partials[static_cast<std::size_t>(omp_get_thread_num())] = local;
        }
        compensated_accumulator<T> total;
        for(const auto& partial : partials)
        {
            total.merge(partial);
        }
        return total.result();
#else
        return serial_fallback_.sum(n, x);
#endif
    }

    real_type asum(Ordinal n, const T* x) const
    {
#ifdef _OPENMP
        std::vector<compensated_accumulator<real_type>> partials(static_cast<std::size_t>(omp_get_max_threads()));
#pragma omp parallel
        {
            compensated_accumulator<real_type> local;
#pragma omp for schedule(static)
            for(Ordinal i = 0; i < n; ++i)
            {
                local.add(arithmetic_type::asum_term(x[i]));
            }
            partials[static_cast<std::size_t>(omp_get_thread_num())] = local;
        }
        compensated_accumulator<real_type> total;
        for(const auto& partial : partials)
        {
            total.merge(partial);
        }
        return total.result();
#else
        return serial_fallback_.asum(n, x);
#endif
    }

    T dot(Ordinal n, const T* x, const T* y) const
    {
#ifdef _OPENMP
        std::vector<compensated_accumulator<T>> partials(static_cast<std::size_t>(omp_get_max_threads()));
#pragma omp parallel
        {
            compensated_accumulator<T> local;
#pragma omp for schedule(static)
            for(Ordinal i = 0; i < n; ++i)
            {
                T err;
                const T term = arithmetic_type::conj_product(x[i], y[i], err);
                local.add_with_error(term, err);
            }
            partials[static_cast<std::size_t>(omp_get_thread_num())] = local;
        }
        compensated_accumulator<T> total;
        for(const auto& partial : partials)
        {
            total.merge(partial);
        }
        return total.result();
#else
        return serial_fallback_.dot(n, x, y);
#endif
    }

    real_type norm_sq(Ordinal n, const T* x) const
    {
#ifdef _OPENMP
        std::vector<compensated_accumulator<real_type>> partials(static_cast<std::size_t>(omp_get_max_threads()));
#pragma omp parallel
        {
            compensated_accumulator<real_type> local;
#pragma omp for schedule(static)
            for(Ordinal i = 0; i < n; ++i)
            {
                real_type err;
                const real_type term = arithmetic_type::norm_sq_term(x[i], err);
                local.add_with_error(term, err);
            }
            partials[static_cast<std::size_t>(omp_get_thread_num())] = local;
        }
        compensated_accumulator<real_type> total;
        for(const auto& partial : partials)
        {
            total.merge(partial);
        }
        return total.result();
#else
        return serial_fallback_.norm_sq(n, x);
#endif
    }

private:
    serial_compensated_reduction_impl<T, Ordinal> serial_fallback_;
};

} // namespace detail

template<class Backend, class T, class Ordinal = std::ptrdiff_t>
class compensated_reduction
{
public:
    using arithmetic_type = detail::arithmetic<T>;
    using real_type = typename arithmetic_type::real_type;
    static constexpr bool is_supported = false;

    explicit compensated_reduction(Ordinal)
    {
    }

    T sum(Ordinal, const T*) const
    {
        throw std::logic_error("high precision BLAS1 reduction is not implemented for this backend");
    }

    real_type asum(Ordinal, const T*) const
    {
        throw std::logic_error("high precision BLAS1 reduction is not implemented for this backend");
    }

    T dot(Ordinal, const T*, const T*) const
    {
        throw std::logic_error("high precision BLAS1 reduction is not implemented for this backend");
    }

    real_type norm_sq(Ordinal, const T*) const
    {
        throw std::logic_error("high precision BLAS1 reduction is not implemented for this backend");
    }
};

template<class T, class Ordinal>
class compensated_reduction<scfd::backend::serial_cpu, T, Ordinal> :
    public detail::serial_compensated_reduction_impl<T, Ordinal>
{
public:
    static constexpr bool is_supported = true;
    using detail::serial_compensated_reduction_impl<T, Ordinal>::serial_compensated_reduction_impl;
};

template<class T, class Ordinal>
class compensated_reduction<scfd::backend::omp, T, Ordinal> :
    public detail::omp_compensated_reduction_impl<T, Ordinal>
{
public:
    static constexpr bool is_supported = true;
    using detail::omp_compensated_reduction_impl<T, Ordinal>::omp_compensated_reduction_impl;
};

#if defined(__CUDACC__) && !defined(__HIPCC__)
template<class T, class Ordinal>
class compensated_reduction<scfd::backend::cuda, T, Ordinal>
{
public:
    using arithmetic_type = detail::arithmetic<T>;
    using real_type = typename arithmetic_type::real_type;
    static constexpr bool is_supported = true;

    explicit compensated_reduction(Ordinal size):
        size_(size)
    {
    }

    T sum(Ordinal n, const T* x) const
    {
        check_size(n);
        return reducer().sum(const_cast<T*>(x));
    }

    real_type asum(Ordinal n, const T* x) const
    {
        check_size(n);
        return reducer().asum(const_cast<T*>(x));
    }

    T dot(Ordinal n, const T* x, const T* y) const
    {
        check_size(n);
        return reducer().dot(const_cast<T*>(x), const_cast<T*>(y));
    }

    real_type norm_sq(Ordinal n, const T* x) const
    {
        check_size(n);
        return arithmetic_type::real_part(reducer().dot(const_cast<T*>(x), const_cast<T*>(x)));
    }

private:
    void check_size(Ordinal n) const
    {
        if(n != size_)
        {
            throw std::logic_error("CUDA high precision reduction currently requires the vector-operations default size");
        }
    }

    gpu_reduction_ogita<T, T*>& reducer() const
    {
        if(!reducer_)
        {
            reducer_.reset(new gpu_reduction_ogita<T, T*>(static_cast<std::size_t>(size_)));
        }
        return *reducer_;
    }

    Ordinal size_;
    mutable std::unique_ptr<gpu_reduction_ogita<T, T*>> reducer_;
};
#endif

#if defined(NMFD_HIGH_PRECISION_BLAS1_HIP_USES_CUDA_OGITA)
template<class T, class Ordinal>
class compensated_reduction<scfd::backend::hip, T, Ordinal>
{
public:
    using arithmetic_type = detail::arithmetic<T>;
    using real_type = typename arithmetic_type::real_type;
    static constexpr bool is_supported = true;

    explicit compensated_reduction(Ordinal size):
        size_(size)
    {
    }

    T sum(Ordinal n, const T* x) const
    {
        check_size(n);
        return reducer().sum(const_cast<T*>(x));
    }

    real_type asum(Ordinal n, const T* x) const
    {
        check_size(n);
        return reducer().asum(const_cast<T*>(x));
    }

    T dot(Ordinal n, const T* x, const T* y) const
    {
        check_size(n);
        return reducer().dot(const_cast<T*>(x), const_cast<T*>(y));
    }

    real_type norm_sq(Ordinal n, const T* x) const
    {
        check_size(n);
        return arithmetic_type::real_part(reducer().dot(const_cast<T*>(x), const_cast<T*>(x)));
    }

private:
    void check_size(Ordinal n) const
    {
        if(n != size_)
        {
            throw std::logic_error("HIP high precision reduction currently requires the vector-operations default size");
        }
    }

    gpu_reduction_ogita<T, T*>& reducer() const
    {
        if(!reducer_)
        {
            reducer_.reset(new gpu_reduction_ogita<T, T*>(static_cast<std::size_t>(size_)));
        }
        return *reducer_;
    }

    Ordinal size_;
    mutable std::unique_ptr<gpu_reduction_ogita<T, T*>> reducer_;
};
#endif

} // namespace high_precision
} // namespace blas1
} // namespace operations
} // namespace nmfd

#endif
