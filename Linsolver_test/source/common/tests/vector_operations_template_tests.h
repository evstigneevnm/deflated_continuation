#ifndef __VECTOR_OPERATIONS_TEMPLATE_TESTS_H__
#define __VECTOR_OPERATIONS_TEMPLATE_TESTS_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if __has_include(<thrust/complex.h>)
#include <thrust/complex.h>
#define VECTOR_OPERATIONS_TESTS_HAS_THRUST_COMPLEX 1
#endif

namespace vector_operations_tests
{

struct test_report
{
    std::size_t checks = 0;
    std::size_t failures = 0;

    void require(bool condition, const std::string& label)
    {
        ++checks;
        if(!condition)
        {
            ++failures;
            std::cout << "FAIL " << label << std::endl;
        }
    }
};

template<class T>
struct scalar_traits
{
    using real_type = T;
    static constexpr bool is_complex = false;

    static T make(long double real, long double)
    {
        return static_cast<T>(real);
    }

    static T conj_mul(const T& x, const T& y)
    {
        return x*y;
    }

    static long double norm_sq_term(const T& x)
    {
        const long double xr = static_cast<long double>(x);
        return xr*xr;
    }

    static long double asum_term(const T& x)
    {
        return std::fabs(static_cast<long double>(x));
    }

    static long double distance(const T& x, const T& y)
    {
        return std::fabs(static_cast<long double>(x-y));
    }

    static long double magnitude(const T& x)
    {
        return std::fabs(static_cast<long double>(x));
    }
};

template<class T>
struct scalar_traits<std::complex<T>>
{
    using real_type = T;
    static constexpr bool is_complex = true;

    static std::complex<T> make(long double real, long double imag)
    {
        return std::complex<T>(static_cast<T>(real), static_cast<T>(imag));
    }

    static std::complex<T> conj_mul(const std::complex<T>& x, const std::complex<T>& y)
    {
        return std::conj(x)*y;
    }

    static long double norm_sq_term(const std::complex<T>& x)
    {
        const long double xr = static_cast<long double>(x.real());
        const long double xi = static_cast<long double>(x.imag());
        return xr*xr + xi*xi;
    }

    static long double asum_term(const std::complex<T>& x)
    {
        return std::fabs(static_cast<long double>(x.real())) + std::fabs(static_cast<long double>(x.imag()));
    }

    static long double distance(const std::complex<T>& x, const std::complex<T>& y)
    {
        const long double dr = std::fabs(static_cast<long double>(x.real() - y.real()));
        const long double di = std::fabs(static_cast<long double>(x.imag() - y.imag()));
        return std::max(dr, di);
    }

    static long double magnitude(const std::complex<T>& x)
    {
        const long double xr = static_cast<long double>(x.real());
        const long double xi = static_cast<long double>(x.imag());
        return std::sqrt(xr*xr + xi*xi);
    }
};

#ifdef VECTOR_OPERATIONS_TESTS_HAS_THRUST_COMPLEX
template<class T>
struct scalar_traits<thrust::complex<T>>
{
    using real_type = T;
    static constexpr bool is_complex = true;

    static thrust::complex<T> make(long double real, long double imag)
    {
        return thrust::complex<T>(static_cast<T>(real), static_cast<T>(imag));
    }

    static thrust::complex<T> conj_mul(const thrust::complex<T>& x, const thrust::complex<T>& y)
    {
        return thrust::conj(x)*y;
    }

    static long double norm_sq_term(const thrust::complex<T>& x)
    {
        const long double xr = static_cast<long double>(x.real());
        const long double xi = static_cast<long double>(x.imag());
        return xr*xr + xi*xi;
    }

    static long double asum_term(const thrust::complex<T>& x)
    {
        return std::fabs(static_cast<long double>(x.real())) + std::fabs(static_cast<long double>(x.imag()));
    }

    static long double distance(const thrust::complex<T>& x, const thrust::complex<T>& y)
    {
        const long double dr = std::fabs(static_cast<long double>(x.real() - y.real()));
        const long double di = std::fabs(static_cast<long double>(x.imag() - y.imag()));
        return std::max(dr, di);
    }

    static long double magnitude(const thrust::complex<T>& x)
    {
        const long double xr = static_cast<long double>(x.real());
        const long double xi = static_cast<long double>(x.imag());
        return std::sqrt(xr*xr + xi*xi);
    }
};
#endif

template<class T>
std::string scalar_to_string(const T& value)
{
    std::ostringstream out;
    out << std::setprecision(18) << value;
    return out.str();
}

template<class T>
std::string scalar_to_string(const std::complex<T>& value)
{
    std::ostringstream out;
    out << std::setprecision(18) << "(" << value.real() << "," << value.imag() << ")";
    return out.str();
}

#ifdef VECTOR_OPERATIONS_TESTS_HAS_THRUST_COMPLEX
template<class T>
std::string scalar_to_string(const thrust::complex<T>& value)
{
    std::ostringstream out;
    out << std::setprecision(18) << "(" << value.real() << "," << value.imag() << ")";
    return out.str();
}
#endif

template<class T>
long double scalar_real_part(const T& value)
{
    return static_cast<long double>(value);
}

template<class T>
long double scalar_imag_part(const T&)
{
    return 0.0L;
}

template<class T>
long double scalar_real_part(const std::complex<T>& value)
{
    return static_cast<long double>(value.real());
}

template<class T>
long double scalar_imag_part(const std::complex<T>& value)
{
    return static_cast<long double>(value.imag());
}

#ifdef VECTOR_OPERATIONS_TESTS_HAS_THRUST_COMPLEX
template<class T>
long double scalar_real_part(const thrust::complex<T>& value)
{
    return static_cast<long double>(value.real());
}

template<class T>
long double scalar_imag_part(const thrust::complex<T>& value)
{
    return static_cast<long double>(value.imag());
}
#endif

template<class T>
bool scalar_is_finite(const T& value)
{
    return std::isfinite(scalar_real_part(value)) && std::isfinite(scalar_imag_part(value));
}

template<class T>
bool scalar_in_unit_box(const T& value)
{
    const long double real = scalar_real_part(value);
    const long double imag = scalar_imag_part(value);
    return real >= 0.0L && real <= 1.0L && imag >= 0.0L && imag <= 1.0L;
}

template<class VecOps, class = void>
struct has_assign_random : std::false_type
{
};

template<class VecOps>
struct has_assign_random<
    VecOps,
    std::void_t<decltype(std::declval<VecOps&>().assign_random(std::declval<typename VecOps::vector_type&>()))>>
    : std::true_type
{
};

template<class VecOps, class = void>
struct has_assign_random_range : std::false_type
{
};

template<class VecOps>
struct has_assign_random_range<
    VecOps,
    std::void_t<decltype(std::declval<VecOps&>().assign_random(
        std::declval<typename VecOps::vector_type&>(),
        std::declval<typename VecOps::scalar_type>(),
        std::declval<typename VecOps::scalar_type>()))>>
    : std::true_type
{
};

template<class T>
T make_scalar(long double real, long double imag = 0.0L)
{
    return scalar_traits<T>::make(real, imag);
}

template<class T>
T invalid_scalar()
{
    using real_type = typename scalar_traits<T>::real_type;
    return scalar_traits<T>::make(std::numeric_limits<real_type>::quiet_NaN(), 0.0L);
}

template<class T>
std::vector<T> make_pattern(std::size_t n, long double shift)
{
    std::vector<T> values(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        const long double real = (static_cast<long double>(static_cast<int>(i%9) - 4))*0.375L + shift;
        const long double imag = (static_cast<long double>(static_cast<int>((i*3)%7) - 3))*0.25L - 0.125L*shift;
        values[i] = make_scalar<T>(real, imag);
    }
    return values;
}

template<class T>
std::vector<T> make_nonzero_pattern(std::size_t n, long double shift)
{
    std::vector<T> values(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        const long double real = 1.0L + 0.2L*static_cast<long double>(i%11) + shift;
        const long double imag = 0.15L*static_cast<long double>((i+2)%5);
        values[i] = make_scalar<T>(real, imag);
    }
    return values;
}

template<class T>
T reference_sum(const std::vector<T>& x)
{
    T result = make_scalar<T>(0.0L);
    for(const auto& value : x)
    {
        result += value;
    }
    return result;
}

template<class T>
T reference_dot(const std::vector<T>& x, const std::vector<T>& y)
{
    T result = make_scalar<T>(0.0L);
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        result += scalar_traits<T>::conj_mul(x[i], y[i]);
    }
    return result;
}

template<class T>
long double reference_asum(const std::vector<T>& x)
{
    long double result = 0.0L;
    for(const auto& value : x)
    {
        result += scalar_traits<T>::asum_term(value);
    }
    return result;
}

template<class T>
long double reference_norm_sq(const std::vector<T>& x)
{
    long double result = 0.0L;
    for(const auto& value : x)
    {
        result += scalar_traits<T>::norm_sq_term(value);
    }
    return result;
}

template<class Real>
long double tolerance_for(long double reference, long double multiplier)
{
    return multiplier*static_cast<long double>(std::numeric_limits<Real>::epsilon())*std::max(1.0L, std::fabs(reference));
}

template<class T>
void check_close(test_report& report, const std::string& label, const T& got, const T& expected, long double multiplier = 4096.0L)
{
    const long double err = scalar_traits<T>::distance(got, expected);
    const long double ref = scalar_traits<T>::magnitude(expected);
    const long double tol = tolerance_for<typename scalar_traits<T>::real_type>(ref, multiplier);
    ++report.checks;
    if(!(err <= tol))
    {
        ++report.failures;
        std::cout << "FAIL " << label << " got=" << scalar_to_string(got)
                  << " expected=" << scalar_to_string(expected)
                  << " err=" << std::setprecision(18) << err
                  << " tol=" << tol << std::endl;
    }
}

template<class Real>
void check_close_real(test_report& report, const std::string& label, Real got, long double expected, long double multiplier = 4096.0L)
{
    const long double got_ld = static_cast<long double>(got);
    const long double err = std::fabs(got_ld - expected);
    const long double tol = tolerance_for<Real>(expected, multiplier);
    ++report.checks;
    if(!(err <= tol))
    {
        ++report.failures;
        std::cout << "FAIL " << label << " got=" << std::setprecision(18) << got_ld
                  << " expected=" << expected
                  << " err=" << err
                  << " tol=" << tol << std::endl;
    }
}

template<class T>
void check_vector_close(
    test_report& report,
    const std::string& label,
    const std::vector<T>& got,
    const std::vector<T>& expected,
    long double multiplier = 4096.0L)
{
    report.require(got.size() == expected.size(), label + " size");
    const std::size_t n = std::min(got.size(), expected.size());
    std::size_t local_failures = 0;
    for(std::size_t i = 0; i < n; ++i)
    {
        const long double err = scalar_traits<T>::distance(got[i], expected[i]);
        const long double ref = scalar_traits<T>::magnitude(expected[i]);
        const long double tol = tolerance_for<typename scalar_traits<T>::real_type>(ref, multiplier);
        ++report.checks;
        if(!(err <= tol))
        {
            ++report.failures;
            if(local_failures < 8)
            {
                std::cout << "FAIL " << label << "[" << i << "] got=" << scalar_to_string(got[i])
                          << " expected=" << scalar_to_string(expected[i])
                          << " err=" << std::setprecision(18) << err
                          << " tol=" << tol << std::endl;
            }
            ++local_failures;
        }
    }
    if(local_failures > 8)
    {
        std::cout << "FAIL " << label << " suppressed " << (local_failures - 8)
                  << " additional element failures" << std::endl;
    }
}

template<class T>
void scale_ref(std::vector<T>& x, const T& alpha)
{
    for(auto& value : x)
    {
        value *= alpha;
    }
}

template<class T>
void add_mul_scalar_ref(std::vector<T>& x, const T& scalar, const T& mul_x)
{
    for(auto& value : x)
    {
        value = mul_x*value + scalar;
    }
}

template<class T>
std::vector<T> assign_mul_ref(const T& mul_x, const std::vector<T>& x)
{
    std::vector<T> result(x.size());
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        result[i] = mul_x*x[i];
    }
    return result;
}

template<class T>
std::vector<T> assign_mul_ref(const T& mul_x, const std::vector<T>& x, const T& mul_y, const std::vector<T>& y)
{
    std::vector<T> result(x.size());
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        result[i] = mul_x*x[i] + mul_y*y[i];
    }
    return result;
}

template<class T>
void add_mul_ref(const T& mul_x, const std::vector<T>& x, std::vector<T>& y)
{
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        y[i] += mul_x*x[i];
    }
}

template<class T>
void add_mul_ref(const T& mul_x, const std::vector<T>& x, const T& mul_y, std::vector<T>& y)
{
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        y[i] = mul_x*x[i] + mul_y*y[i];
    }
}

template<class T>
void add_mul_ref(const T& mul_x, const std::vector<T>& x, const T& mul_y, const std::vector<T>& y, const T& mul_z, std::vector<T>& z)
{
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        z[i] = mul_x*x[i] + mul_y*y[i] + mul_z*z[i];
    }
}

template<class T>
std::vector<T> mul_pointwise_ref(const T& mul_x, const std::vector<T>& x, const T& mul_y, const std::vector<T>& y)
{
    std::vector<T> result(x.size());
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        result[i] = (mul_x*x[i])*(mul_y*y[i]);
    }
    return result;
}

template<class T>
std::vector<T> div_pointwise_ref(const T& mul_x, const std::vector<T>& x, const T& mul_y, const std::vector<T>& y)
{
    std::vector<T> result(x.size());
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        result[i] = (mul_x*x[i])/(mul_y*y[i]);
    }
    return result;
}

template<class VecOps, class Access>
void run_core_vector_operations_suite(VecOps& vec_ops, Access access, std::size_t n, const std::string& label, test_report& report)
{
    using scalar_type = typename VecOps::scalar_type;
    using norm_type = typename VecOps::norm_type;
    using vector_type = typename VecOps::vector_type;
    using traits = scalar_traits<scalar_type>;

    vector_type x;
    vector_type y;
    vector_type z;
    vector_type w;
    vec_ops.init_vectors(x, y, z, w);
    vec_ops.start_use_vectors(x, y, z, w);

    report.require(vec_ops.get_vector_size() == n, label + " get_vector_size");
    check_close_real(report, label + " get_l2_size", static_cast<norm_type>(vec_ops.get_l2_size()), std::sqrt(static_cast<long double>(n)), 1024.0L);

    const auto hx0 = make_pattern<scalar_type>(n, 0.15L);
    const auto hy0 = make_pattern<scalar_type>(n, -0.35L);
    const auto hz0 = make_pattern<scalar_type>(n, 0.70L);

    vec_ops.assign_scalar(make_scalar<scalar_type>(2.5L, -0.5L), x);
    check_vector_close(report, label + " assign_scalar", access.read(vec_ops, x, n), std::vector<scalar_type>(n, make_scalar<scalar_type>(2.5L, -0.5L)));

    access.write(vec_ops, x, hx0);
    vec_ops.assign(x, y);
    check_vector_close(report, label + " assign", access.read(vec_ops, y, n), hx0);

    auto expected = hx0;
    vec_ops.add_mul_scalar(make_scalar<scalar_type>(-1.25L, 0.375L), make_scalar<scalar_type>(0.5L, -0.25L), x);
    add_mul_scalar_ref(expected, make_scalar<scalar_type>(-1.25L, 0.375L), make_scalar<scalar_type>(0.5L, -0.25L));
    check_vector_close(report, label + " add_mul_scalar", access.read(vec_ops, x, n), expected);

    access.write(vec_ops, x, hx0);
    expected = hx0;
    vec_ops.scale(make_scalar<scalar_type>(-2.0L, 0.25L), x);
    scale_ref(expected, make_scalar<scalar_type>(-2.0L, 0.25L));
    check_vector_close(report, label + " scale", access.read(vec_ops, x, n), expected);

    access.write(vec_ops, x, hx0);
    vec_ops.assign_mul(make_scalar<scalar_type>(1.75L, -0.25L), x, y);
    check_vector_close(report, label + " assign_mul two-vector", access.read(vec_ops, y, n), assign_mul_ref(make_scalar<scalar_type>(1.75L, -0.25L), hx0));

    access.write(vec_ops, x, hx0);
    access.write(vec_ops, y, hy0);
    vec_ops.assign_mul(make_scalar<scalar_type>(1.25L, 0.125L), x, make_scalar<scalar_type>(-0.5L, 0.25L), y, z);
    check_vector_close(
        report,
        label + " assign_mul three-vector",
        access.read(vec_ops, z, n),
        assign_mul_ref(make_scalar<scalar_type>(1.25L, 0.125L), hx0, make_scalar<scalar_type>(-0.5L, 0.25L), hy0));

    access.write(vec_ops, x, hx0);
    auto y_expected = hy0;
    access.write(vec_ops, y, hy0);
    vec_ops.add_mul(make_scalar<scalar_type>(-1.5L, 0.25L), x, y);
    add_mul_ref(make_scalar<scalar_type>(-1.5L, 0.25L), hx0, y_expected);
    check_vector_close(report, label + " add_mul axpy", access.read(vec_ops, y, n), y_expected);

    access.write(vec_ops, x, hx0);
    y_expected = hy0;
    access.write(vec_ops, y, hy0);
    vec_ops.add_mul(make_scalar<scalar_type>(0.75L, -0.125L), x, make_scalar<scalar_type>(-1.25L, 0.5L), y);
    add_mul_ref(make_scalar<scalar_type>(0.75L, -0.125L), hx0, make_scalar<scalar_type>(-1.25L, 0.5L), y_expected);
    check_vector_close(report, label + " add_mul two-term", access.read(vec_ops, y, n), y_expected);

    access.write(vec_ops, x, hx0);
    access.write(vec_ops, y, hy0);
    auto z_expected = hz0;
    access.write(vec_ops, z, hz0);
    vec_ops.add_lin_comb(make_scalar<scalar_type>(-0.5L, 0.25L), x, make_scalar<scalar_type>(1.5L, -0.125L), y, make_scalar<scalar_type>(0.25L, 0.125L), z);
    add_mul_ref(
        make_scalar<scalar_type>(-0.5L, 0.25L),
        hx0,
        make_scalar<scalar_type>(1.5L, -0.125L),
        hy0,
        make_scalar<scalar_type>(0.25L, 0.125L),
        z_expected);
    check_vector_close(report, label + " add_lin_comb three-term", access.read(vec_ops, z, n), z_expected);

    access.write(vec_ops, x, hx0);
    access.write(vec_ops, y, make_nonzero_pattern<scalar_type>(n, 0.4L));
    vec_ops.mul_pointwise(make_scalar<scalar_type>(1.5L, 0.125L), x, make_scalar<scalar_type>(-0.75L, 0.25L), y, z);
    check_vector_close(
        report,
        label + " mul_pointwise out",
        access.read(vec_ops, z, n),
        mul_pointwise_ref(make_scalar<scalar_type>(1.5L, 0.125L), hx0, make_scalar<scalar_type>(-0.75L, 0.25L), make_nonzero_pattern<scalar_type>(n, 0.4L)));

    auto x_expected = hx0;
    const auto hden = make_nonzero_pattern<scalar_type>(n, 0.8L);
    access.write(vec_ops, x, hx0);
    access.write(vec_ops, y, hden);
    vec_ops.div_pointwise(x, make_scalar<scalar_type>(2.0L, 0.25L), y);
    for(std::size_t i = 0; i < n; ++i)
    {
        x_expected[i] = x_expected[i]/(make_scalar<scalar_type>(2.0L, 0.25L)*hden[i]);
    }
    check_vector_close(report, label + " div_pointwise in-place", access.read(vec_ops, x, n), x_expected);

    access.write(vec_ops, x, hx0);
    access.write(vec_ops, y, hden);
    vec_ops.div_pointwise(make_scalar<scalar_type>(1.25L, -0.125L), x, make_scalar<scalar_type>(2.0L, 0.25L), y, z);
    check_vector_close(
        report,
        label + " div_pointwise out",
        access.read(vec_ops, z, n),
        div_pointwise_ref(make_scalar<scalar_type>(1.25L, -0.125L), hx0, make_scalar<scalar_type>(2.0L, 0.25L), hden));

    access.write(vec_ops, x, hx0);
    access.write(vec_ops, y, hy0);
    check_close(report, label + " scalar_prod", vec_ops.scalar_prod(x, y), reference_dot(hx0, hy0), 16384.0L);
    check_close(report, label + " sum", vec_ops.sum(x), reference_sum(hx0), 16384.0L);
    check_close_real(report, label + " asum", vec_ops.asum(x), reference_asum(hx0), 16384.0L);

    const long double norm_sq = reference_norm_sq(hx0);
    check_close_real(report, label + " norm", vec_ops.norm(x), std::sqrt(norm_sq), 16384.0L);
    check_close_real(report, label + " norm_sq", vec_ops.norm_sq(x), norm_sq, 32768.0L);
    check_close_real(report, label + " norm_l2", vec_ops.norm_l2(x), std::sqrt(norm_sq/static_cast<long double>(n)), 16384.0L);
    check_close_real(report, label + " norm2_sq", vec_ops.norm2_sq(x), norm_sq/static_cast<long double>(n), 32768.0L);
    const auto rank1_value = make_scalar<scalar_type>(-0.75L, 0.5L);
    check_close_real(report, label + " norm_rank1", vec_ops.norm_rank1(x, rank1_value), std::sqrt(norm_sq + scalar_traits<scalar_type>::norm_sq_term(rank1_value)), 16384.0L);

    auto invalid = hx0;
    invalid.back() = invalid_scalar<scalar_type>();
    access.write(vec_ops, x, invalid);
    report.require(!vec_ops.check_is_valid_number(x), label + " check_is_valid_number rejects NaN");

    if constexpr(has_assign_random<VecOps>::value)
    {
        vec_ops.assign_random(x);
        const auto random_values = access.read(vec_ops, x, n);
        bool random_values_are_finite = true;
        bool random_values_are_in_unit_box = true;
        for(const auto& value : random_values)
        {
            random_values_are_finite = random_values_are_finite && scalar_is_finite(value);
            random_values_are_in_unit_box = random_values_are_in_unit_box && scalar_in_unit_box(value);
        }
        report.require(random_values_are_finite, label + " assign_random finite");
        report.require(random_values_are_in_unit_box, label + " assign_random unit range");
    }

    if constexpr(!traits::is_complex)
    {
        if constexpr(has_assign_random_range<VecOps>::value)
        {
            vec_ops.assign_random(x, make_scalar<scalar_type>(-2.0L), make_scalar<scalar_type>(3.0L));
            const auto ranged_random_values = access.read(vec_ops, x, n);
            bool ranged_random_values_are_in_range = true;
            for(const auto& value : ranged_random_values)
            {
                const long double real = scalar_real_part(value);
                ranged_random_values_are_in_range = ranged_random_values_are_in_range && real >= -2.0L && real <= 3.0L;
            }
            report.require(ranged_random_values_are_in_range, label + " assign_random ranged");
        }

        const auto real_pattern = make_pattern<scalar_type>(n, -0.2L);
        access.write(vec_ops, x, real_pattern);
        vec_ops.make_abs_copy(x, y);
        auto abs_expected = real_pattern;
        for(auto& value : abs_expected)
        {
            value = std::abs(value);
        }
        check_vector_close(report, label + " make_abs_copy", access.read(vec_ops, y, n), abs_expected);

        access.write(vec_ops, x, real_pattern);
        vec_ops.make_abs(x);
        check_vector_close(report, label + " make_abs", access.read(vec_ops, x, n), abs_expected);

        access.write(vec_ops, x, real_pattern);
        const auto norm_before = vec_ops.normalize(x);
        check_close_real(report, label + " normalize returned norm", norm_before, std::sqrt(reference_norm_sq(real_pattern)), 16384.0L);
        check_close_real(report, label + " normalize output norm", vec_ops.norm(x), 1.0L, 16384.0L);

        long double expected_inf = 0.0L;
        for(const auto& value : real_pattern)
        {
            expected_inf = std::max(expected_inf, std::fabs(static_cast<long double>(value)));
        }
        access.write(vec_ops, x, real_pattern);
        check_close_real(report, label + " norm_inf expected", vec_ops.norm_inf(x), expected_inf, 16384.0L);

        auto expected_max = real_pattern.front();
        std::size_t expected_argmax = 0;
        for(std::size_t i = 1; i < real_pattern.size(); ++i)
        {
            if(real_pattern[i] > expected_max)
            {
                expected_max = real_pattern[i];
                expected_argmax = i;
            }
        }
        check_close(report, label + " max_element", vec_ops.max_element(x), expected_max, 1024.0L);
        report.require(vec_ops.argmax_element(x) == expected_argmax, label + " argmax_element");
        const auto max_argmax = vec_ops.max_argmax_element(x);
        check_close(report, label + " max_argmax_element value", max_argmax.first, expected_max, 1024.0L);
        report.require(max_argmax.second == expected_argmax, label + " max_argmax_element index");

        const auto maxmin_x = make_pattern<scalar_type>(n, -0.2L);
        const auto maxmin_y = make_pattern<scalar_type>(n, 0.45L);
        const auto max_sc = make_scalar<scalar_type>(0.35L);
        const auto min_sc = make_scalar<scalar_type>(-0.25L);

        access.write(vec_ops, x, maxmin_x);
        access.write(vec_ops, y, maxmin_y);
        auto max_expected = maxmin_y;
        for(std::size_t i = 0; i < n; ++i)
        {
            max_expected[i] = std::max(std::max(maxmin_x[i], maxmin_y[i]), max_sc);
        }
        vec_ops.max_pointwise(max_sc, x, y);
        check_vector_close(report, label + " max_pointwise two-vector", access.read(vec_ops, y, n), max_expected);

        access.write(vec_ops, y, maxmin_y);
        max_expected = maxmin_y;
        for(auto& value : max_expected)
        {
            value = std::max(value, max_sc);
        }
        vec_ops.max_pointwise(max_sc, y);
        check_vector_close(report, label + " max_pointwise in-place", access.read(vec_ops, y, n), max_expected);

        access.write(vec_ops, x, maxmin_x);
        access.write(vec_ops, y, maxmin_y);
        auto min_expected = maxmin_y;
        for(std::size_t i = 0; i < n; ++i)
        {
            min_expected[i] = std::min(std::min(maxmin_x[i], maxmin_y[i]), min_sc);
        }
        vec_ops.min_pointwise(min_sc, x, y);
        check_vector_close(report, label + " min_pointwise two-vector", access.read(vec_ops, y, n), min_expected);

        access.write(vec_ops, y, maxmin_y);
        min_expected = maxmin_y;
        for(auto& value : min_expected)
        {
            value = std::min(value, min_sc);
        }
        vec_ops.min_pointwise(min_sc, y);
        check_vector_close(report, label + " min_pointwise in-place", access.read(vec_ops, y, n), min_expected);

        vec_ops.set_value_at_point(make_scalar<scalar_type>(7.25L), n/2, x);
        check_close(report, label + " get_value_at_point", vec_ops.get_value_at_point(n/2, x), make_scalar<scalar_type>(7.25L), 1024.0L);
    }

    vec_ops.stop_use_vectors(x, y, z, w);
    vec_ops.free_vectors(x, y, z, w);
}

template<class VecOps, class Access>
void run_multivector_suite(VecOps& vec_ops, Access access, std::size_t n, std::size_t m, const std::string& label, test_report& report)
{
    using scalar_type = typename VecOps::scalar_type;
    using multivector_type = typename VecOps::multivector_type;

    multivector_type values;
    vec_ops.init_multivector(values, m);
    vec_ops.start_use_multivector(values, m);

    for(std::size_t k = 0; k < m; ++k)
    {
        auto& vector = vec_ops.at(values, m, k);
        vec_ops.assign_scalar(make_scalar<scalar_type>(static_cast<long double>(k + 1), 0.25L*static_cast<long double>(k)), vector);
    }

    for(std::size_t k = 0; k < m; ++k)
    {
        auto& vector = vec_ops.at(values, m, k);
        const auto expected = std::vector<scalar_type>(n, make_scalar<scalar_type>(static_cast<long double>(k + 1), 0.25L*static_cast<long double>(k)));
        check_vector_close(report, label + " multivector at " + std::to_string(k), access.read(vec_ops, vector, n), expected);
    }

    vec_ops.stop_use_multivector(values, m);
    vec_ops.free_multivector(values, m);
}

template<class VecOps, class Access>
void run_vector_operations_template_tests(VecOps& vec_ops, Access access, std::size_t n, const std::string& label, test_report& report)
{
    run_core_vector_operations_suite(vec_ops, access, n, label, report);
    run_multivector_suite(vec_ops, access, n, 3, label, report);
}

} // namespace vector_operations_tests

#endif
