#ifndef __SCFD_VECTOR_OPERATIONS_HIGH_PRECISION_TESTS_H__
#define __SCFD_VECTOR_OPERATIONS_HIGH_PRECISION_TESTS_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <common/tests/vector_operations_template_tests.h>

namespace vector_operations_tests
{

template<class VecOps, class = void>
struct has_last_reduction_used_high_precision : std::false_type
{
};

template<class VecOps>
struct has_last_reduction_used_high_precision<
    VecOps,
    std::void_t<decltype(std::declval<VecOps&>().last_reduction_used_high_precision())>>
    : std::true_type
{
};

template<class VecOps>
void check_last_reduction_path(
    test_report& report,
    const std::string& label,
    VecOps& vec_ops,
    bool expected)
{
    if constexpr(has_last_reduction_used_high_precision<VecOps>::value)
    {
        report.require(vec_ops.last_reduction_used_high_precision() == expected, label);
    }
}

template<class T>
std::vector<T> make_high_precision_pattern(std::size_t n)
{
    using real_type = typename scalar_traits<T>::real_type;
    const long double large = (std::numeric_limits<real_type>::digits <= 24) ? 1.0e8L : 1.0e16L;
    std::vector<T> values(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        switch(i%3)
        {
        case 0:
            values[i] = make_scalar<T>(large, -0.25L*large);
            break;
        case 1:
            values[i] = make_scalar<T>(1.0L + static_cast<long double>(i%7), -0.5L - 0.25L*static_cast<long double>(i%5));
            break;
        default:
            values[i] = make_scalar<T>(-large, 0.25L*large);
            break;
        }
    }
    return values;
}

template<class T>
std::vector<T> make_unit_pattern(std::size_t n)
{
    return std::vector<T>(n, make_scalar<T>(1.0L, 0.0L));
}

template<class T>
T reference_sum_high_precision(const std::vector<T>& x)
{
    long double real = 0.0L;
    long double imag = 0.0L;
    for(const auto& value : x)
    {
        real += scalar_real_part(value);
        imag += scalar_imag_part(value);
    }
    return make_scalar<T>(real, imag);
}

template<class T>
T reference_dot_high_precision(const std::vector<T>& x, const std::vector<T>& y)
{
    long double real = 0.0L;
    long double imag = 0.0L;
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        const long double xr = scalar_real_part(x[i]);
        const long double xi = scalar_imag_part(x[i]);
        const long double yr = scalar_real_part(y[i]);
        const long double yi = scalar_imag_part(y[i]);
        real += xr*yr + xi*yi;
        imag += xr*yi - xi*yr;
    }
    return make_scalar<T>(real, imag);
}

template<class T>
long double reference_asum_high_precision(const std::vector<T>& x)
{
    long double result = 0.0L;
    for(const auto& value : x)
    {
        result += std::fabs(scalar_real_part(value)) + std::fabs(scalar_imag_part(value));
    }
    return result;
}

template<class T>
long double reference_norm_sq_high_precision(const std::vector<T>& x)
{
    long double result = 0.0L;
    for(const auto& value : x)
    {
        const long double real = scalar_real_part(value);
        const long double imag = scalar_imag_part(value);
        result += real*real + imag*imag;
    }
    return result;
}

template<class VecOps, class Access>
void run_scfd_vector_operations_high_precision_tests(
    VecOps& vec_ops,
    Access access,
    std::size_t n,
    const std::string& label,
    test_report& report)
{
    if constexpr(!has_high_precision_state<VecOps>::value)
    {
        return;
    }
    else
    {
        using scalar_type = typename VecOps::scalar_type;
        using vector_type = typename VecOps::vector_type;

        const auto hx = make_high_precision_pattern<scalar_type>(n);
        const auto hy = make_unit_pattern<scalar_type>(n);

        vector_type x;
        vector_type y;
        vec_ops.init_vector(x);
        vec_ops.init_vector(y);
        vec_ops.start_use_vector(x);
        vec_ops.start_use_vector(y);
        access.write(vec_ops, x, hx);
        access.write(vec_ops, y, hy);

        vec_ops.set_regular_precision();
        (void)vec_ops.sum(x);
        check_last_reduction_path(report, label + " regular sum dispatch", vec_ops, false);

        vec_ops.set_high_precision();

        const auto sum_value = vec_ops.sum(x);
        check_last_reduction_path(report, label + " high precision sum dispatch", vec_ops, true);
        check_close(report, label + " high precision sum", sum_value, reference_sum_high_precision(hx), 16.0L);

        const auto asum_value = vec_ops.asum(x);
        check_last_reduction_path(report, label + " high precision asum dispatch", vec_ops, true);
        check_close_real(report, label + " high precision asum", asum_value, reference_asum_high_precision(hx), 128.0L);

        const auto dot_value = vec_ops.scalar_prod(x, y);
        check_last_reduction_path(report, label + " high precision dot dispatch", vec_ops, true);
        check_close(report, label + " high precision dot", dot_value, reference_dot_high_precision(hx, hy), 16.0L);

        const auto norm_sq_value = vec_ops.norm_sq(x);
        check_last_reduction_path(report, label + " high precision norm_sq dispatch", vec_ops, true);
        const long double norm_sq_ref = reference_norm_sq_high_precision(hx);
        check_close_real(report, label + " high precision norm_sq", norm_sq_value, norm_sq_ref, 512.0L);

        const auto norm_value = vec_ops.norm(x);
        check_last_reduction_path(report, label + " high precision norm dispatch", vec_ops, true);
        check_close_real(report, label + " high precision norm", norm_value, std::sqrt(norm_sq_ref), 512.0L);

        vec_ops.set_regular_precision();
        vec_ops.stop_use_vector(x);
        vec_ops.stop_use_vector(y);
        vec_ops.free_vector(x);
        vec_ops.free_vector(y);
    }
}

} // namespace vector_operations_tests

#endif
