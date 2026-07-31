#ifndef __STABILITY_TESTS_COMMON_COMPLEX_AFFINE_TRANSFORMATIONS_TEST_SUITE_H__
#define __STABILITY_TESTS_COMMON_COMPLEX_AFFINE_TRANSFORMATIONS_TEST_SUITE_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/product_vector_space.h>
#include <stability/eigensolvers/transformations/complex_affine_block_operator.h>
#include <stability/eigensolvers/transformations/complex_shift_block_operator.h>
#include <stability/eigensolvers/transformations/stability_polynomial_factorization.h>

#include "analytical_dense_operator.h"
#include "analytical_eigenproblem.h"

namespace stability
{
namespace tests
{
namespace complex_affine_transformations_test
{

inline std::size_t checks = 0;
inline std::size_t failures = 0;

inline void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

template<class Function>
void require_throws(Function&& function, const std::string& message)
{
    bool threw = false;
    try
    {
        function();
    }
    catch(const std::exception&)
    {
        threw = true;
    }
    require(threw, message);
}

template<class Real>
bool close(
    const std::complex<Real>& actual,
    const std::complex<Real>& expected,
    Real tolerance)
{
    return std::abs(actual - expected) <=
        tolerance *
        std::max({Real(1), std::abs(actual), std::abs(expected)});
}

template<class Real>
Real operator_tolerance()
{
    return std::is_same<Real, float>::value
        ? Real(3.0e-5)
        : Real(3.0e-12);
}

template<class Real>
Real factorization_tolerance()
{
    return std::is_same<Real, float>::value
        ? Real(2.0e-3)
        : Real(2.0e-10);
}

template<class Scalar>
Scalar integer_power(Scalar value, std::size_t exponent)
{
    Scalar result(1);
    while(exponent > 0)
    {
        if(exponent % 2 != 0)
            result *= value;
        exponent /= 2;
        if(exponent > 0)
            value *= value;
    }
    return result;
}

template<class Real>
void test_factorization(const std::string& label)
{
    using complex_type = std::complex<Real>;
    namespace transformations =
        stability::eigensolvers::transformations;

    const Real step = Real(0.075);
    const std::size_t repetitions = 3;
    const complex_type shift =
        std::polar(Real(1.02), Real(0.43));
    const std::vector<complex_type> probes{
        complex_type(Real(-3.2), Real(0.7)),
        complex_type(Real(-0.5), Real(-1.1)),
        complex_type(Real(0), Real(2.0)),
        complex_type(Real(1.3), Real(0.2))};
    const Real tolerance = factorization_tolerance<Real>();

    const auto euler =
        transformations::euler_denominator_factors(
            step,
            repetitions,
            shift);
    require(
        euler.size() == repetitions,
        label + " Euler factor count");
    for(const auto& probe : probes)
    {
        const auto actual =
            transformations::evaluate_affine_factors(euler, probe);
        const auto expected =
            integer_power(
                complex_type(Real(1), Real{}) + step * probe,
                repetitions) -
            shift;
        require(
            close(actual, expected, tolerance),
            label + " Euler factor identity");
    }

    const auto rk4 =
        transformations::rk4_denominator_factors(
            step,
            repetitions,
            shift);
    const auto rk4_again =
        transformations::rk4_denominator_factors(
            step,
            repetitions,
            shift);
    require(
        rk4.size() == 4 * repetitions,
        label + " RK4 factor count");
    require(
        rk4_again.size() == rk4.size(),
        label + " RK4 deterministic factor count");
    for(std::size_t index = 0; index < rk4.size(); ++index)
    {
        require(
            transformations::is_finite(rk4[index]),
            label + " RK4 finite factor");
        require(
            close(
                rk4[index].operator_scale,
                rk4_again[index].operator_scale,
                tolerance) &&
            close(
                rk4[index].diagonal_shift,
                rk4_again[index].diagonal_shift,
                tolerance),
            label + " RK4 deterministic factors");
    }

    const auto rk4_coefficients =
        transformations::classical_rk4_stability_polynomial<Real>();
    for(const auto& probe : probes)
    {
        const auto one_step =
            transformations::evaluate_stability_polynomial(
                rk4_coefficients,
                step * probe);
        const auto expected =
            integer_power(one_step, repetitions) - shift;
        const auto actual =
            transformations::evaluate_affine_factors(rk4, probe);
        require(
            close(actual, expected, tolerance * Real(8)),
            label + " RK4 factor identity");
    }

    const std::vector<Real> quadratic{
        Real(2),
        Real(-3),
        Real(0.5)};
    const auto quadratic_factors =
        transformations::polynomial_denominator_factors<Real>(
            quadratic,
            step,
            2,
            shift);
    require(
        quadratic_factors.size() == 4,
        label + " generic polynomial factor count");
    for(const auto& probe : probes)
    {
        const auto one_step =
            transformations::evaluate_stability_polynomial(
                quadratic,
                step * probe);
        const auto expected = one_step * one_step - shift;
        const auto actual =
            transformations::evaluate_affine_factors(
                quadratic_factors,
                probe);
        require(
            close(actual, expected, tolerance * Real(8)),
            label + " generic polynomial factor identity");
    }

    require_throws(
        [&]
        {
            transformations::polynomial_denominator_factors<Real>(
                std::vector<Real>{Real(1)},
                step,
                repetitions,
                shift);
        },
        label + " rejects a constant polynomial");
    require_throws(
        [&]
        {
            transformations::polynomial_denominator_factors<Real>(
                quadratic,
                Real{},
                repetitions,
                shift);
        },
        label + " rejects zero step");
    require_throws(
        [&]
        {
            transformations::polynomial_denominator_factors<Real>(
                quadratic,
                step,
                0,
                shift);
        },
        label + " rejects zero repetitions");
    require_throws(
        [&]
        {
            transformations::polynomial_denominator_factors<Real>(
                std::vector<Real>{Real(1), Real{}},
                step,
                repetitions,
                shift);
        },
        label + " rejects zero leading coefficient");
}

template<class ProductSpace, class Real>
void set_complex_vector(
    const ProductSpace& vector_space,
    const std::vector<std::complex<Real>>& host,
    typename ProductSpace::vector_type& destination)
{
    std::vector<Real> packed(2 * host.size());
    for(std::size_t index = 0; index < host.size(); ++index)
    {
        packed[index] = host[index].real();
        packed[host.size() + index] = host[index].imag();
    }
    vector_space.set(packed.data(), destination, packed.size());
}

template<class ProductSpace, class Real>
std::vector<std::complex<Real>> get_complex_vector(
    const ProductSpace& vector_space,
    const typename ProductSpace::vector_type& source)
{
    const std::size_t size = vector_space.first_size();
    std::vector<Real> packed(2 * size);
    vector_space.get(source, packed.data(), packed.size());
    std::vector<std::complex<Real>> result(size);
    for(std::size_t index = 0; index < size; ++index)
    {
        result[index] =
            std::complex<Real>(packed[index], packed[size + index]);
    }
    return result;
}

template<class Backend, class Real>
void test_affine_operator(const std::string& label)
{
    using component_space_type =
        scfd_vector_operations<Backend, Real>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<
            component_space_type>;
    using real_operator_type =
        stability::tests::analytical_dense_operator<
            component_space_type,
            Real>;
    using affine_operator_type =
        stability::eigensolvers::transformations::
            complex_affine_block_operator<
                product_space_type,
                real_operator_type>;
    using shift_operator_type =
        stability::eigensolvers::transformations::
            complex_shift_block_operator<
                product_space_type,
                real_operator_type>;
    using vector_type = typename product_space_type::vector_type;
    using complex_type = std::complex<Real>;

    const auto problem =
        stability::tests::nonnormal_eigenproblem<Real>();
    component_space_type component_space(problem.dimension());
    product_space_type product_space(
        component_space,
        component_space);
    real_operator_type real_operator(component_space, problem);

    const std::vector<complex_type> input{
        complex_type(Real(1), Real(0.5)),
        complex_type(Real(-2), Real(0.25)),
        complex_type(Real(0.75), Real(-1.5))};
    const complex_type alpha(Real(0.35), Real(-0.6));
    const complex_type beta(Real(-1.2), Real(0.45));
    const Real tolerance = operator_tolerance<Real>();

    vector_type source;
    vector_type destination;
    product_space.init_vector(source);
    product_space.init_vector(destination);
    product_space.start_use_vector(source);
    product_space.start_use_vector(destination);
    set_complex_vector(product_space, input, source);

    affine_operator_type affine(
        product_space,
        real_operator,
        alpha,
        beta);
    require(affine.apply(source, destination), label + " affine apply");
    const auto actual =
        get_complex_vector<product_space_type, Real>(
            product_space,
            destination);
    const auto applied = problem.apply(input);
    for(std::size_t index = 0; index < input.size(); ++index)
    {
        require(
            close(
                actual[index],
                alpha * applied[index] + beta * input[index],
                tolerance),
            label + " affine value");
    }
    require(
        affine.operator_scale() == alpha &&
            affine.diagonal_shift() == beta,
        label + " affine coefficients");
    require(
        affine.operator_calls() == 1 &&
            affine.component_operator_calls() == 2 &&
            affine.component_operator_failures() == 0,
        label + " affine call accounting");

    const std::size_t calls_before_beta_only =
        real_operator.operator_calls();
    affine_operator_type beta_only(
        product_space,
        real_operator,
        complex_type{},
        beta);
    require(
        beta_only.apply(source, destination),
        label + " beta-only apply");
    const auto beta_actual =
        get_complex_vector<product_space_type, Real>(
            product_space,
            destination);
    for(std::size_t index = 0; index < input.size(); ++index)
    {
        require(
            close(
                beta_actual[index],
                beta * input[index],
                tolerance),
            label + " beta-only value");
    }
    require(
        beta_only.component_operator_calls() == 0 &&
            real_operator.operator_calls() == calls_before_beta_only,
        label + " beta-only skips real operator");

    const complex_type shift(Real(0.8), Real(-0.3));
    shift_operator_type shifted(
        product_space,
        real_operator,
        shift.real(),
        shift.imag());
    require(
        shifted.apply(source, destination),
        label + " shift specialization apply");
    const auto shifted_actual =
        get_complex_vector<product_space_type, Real>(
            product_space,
            destination);
    for(std::size_t index = 0; index < input.size(); ++index)
    {
        require(
            close(
                shifted_actual[index],
                applied[index] - shift * input[index],
                tolerance),
            label + " shift specialization value");
    }
    require(
        shifted.shift_real() == shift.real() &&
            shifted.shift_imaginary() == shift.imag(),
        label + " shift specialization coefficients");

    real_operator_type failing_operator(component_space, problem);
    failing_operator.fail_after(1);
    affine_operator_type failing(
        product_space,
        failing_operator,
        alpha,
        beta);
    const std::vector<complex_type> sentinel(
        input.size(),
        complex_type(Real(7), Real(-9)));
    set_complex_vector(product_space, sentinel, destination);
    require(
        !failing.apply(source, destination),
        label + " component failure propagation");
    require(
        failing.operator_calls() == 1 &&
            failing.component_operator_calls() == 2 &&
            failing.component_operator_failures() == 1,
        label + " component failure accounting");
    const auto after_failure =
        get_complex_vector<product_space_type, Real>(
            product_space,
            destination);
    for(std::size_t index = 0; index < sentinel.size(); ++index)
    {
        require(
            close(after_failure[index], sentinel[index], tolerance),
            label + " failure leaves destination unchanged");
    }

    require_throws(
        [&]
        {
            affine_operator_type invalid(
                product_space,
                real_operator,
                complex_type(
                    std::numeric_limits<Real>::quiet_NaN(),
                    Real{}),
                beta);
        },
        label + " rejects nonfinite coefficient");

    product_space.stop_use_vector(destination);
    product_space.stop_use_vector(source);
    product_space.free_vector(destination);
    product_space.free_vector(source);
}

template<class Backend>
void run_backend(const std::string& label)
{
    test_affine_operator<Backend, float>(label + " float");
    test_affine_operator<Backend, double>(label + " double");
}

inline void run_factorization_tests(const std::string& label)
{
    test_factorization<float>(label + " float");
    test_factorization<double>(label + " double");
}

inline int finish()
{
    std::cout << "Checks: " << checks
              << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        std::cout << "FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASSED\n";
    return EXIT_SUCCESS;
}

} // namespace complex_affine_transformations_test
} // namespace tests
} // namespace stability

#endif
