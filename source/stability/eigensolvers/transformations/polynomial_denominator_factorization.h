#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_POLYNOMIAL_DENOMINATOR_FACTORIZATION_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_POLYNOMIAL_DENOMINATOR_FACTORIZATION_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "complex_affine_factor.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class Real>
std::vector<std::complex<Real>> complex_roots(
    const std::complex<Real>& value,
    std::size_t count)
{
    if(count == 0)
        throw std::invalid_argument(
            "complex root count must be positive");
    if(!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument(
            "complex root value must be finite");

    const Real pi = std::acos(Real(-1));
    const Real inverse_count = Real(1) / static_cast<Real>(count);
    const Real root_magnitude =
        std::pow(std::abs(value), inverse_count);
    const Real principal_angle = std::arg(value) * inverse_count;

    std::vector<std::complex<Real>> result;
    result.reserve(count);
    for(std::size_t index = 0; index < count; ++index)
    {
        const Real angle =
            principal_angle +
            Real(2) * pi * static_cast<Real>(index) * inverse_count;
        result.emplace_back(std::polar(root_magnitude, angle));
    }
    return result;
}

namespace detail
{

template<class Real>
bool finite(const std::complex<Real>& value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

template<class Real>
std::pair<std::complex<Real>, std::complex<Real>>
evaluate_polynomial_and_derivative(
    const std::vector<std::complex<Real>>& coefficients,
    const std::complex<Real>& argument)
{
    std::complex<Real> value = coefficients.back();
    std::complex<Real> derivative{};
    for(std::size_t index = coefficients.size() - 1;
        index > 0;
        --index)
    {
        derivative = derivative * argument + value;
        value = value * argument + coefficients[index - 1];
    }
    return {value, derivative};
}

template<class Real>
Real polynomial_scale(
    const std::vector<std::complex<Real>>& coefficients,
    const std::complex<Real>& argument)
{
    const Real magnitude = std::abs(argument);
    Real result = std::abs(coefficients.back());
    for(std::size_t index = coefficients.size() - 1;
        index > 0;
        --index)
    {
        result =
            result * magnitude +
            std::abs(coefficients[index - 1]);
    }
    return result;
}

template<class Real>
std::vector<std::complex<Real>> polynomial_roots(
    std::vector<std::complex<Real>> coefficients)
{
    if(coefficients.size() < 2)
        throw std::invalid_argument(
            "polynomial root extraction requires positive degree");
    for(const auto& coefficient : coefficients)
    {
        if(!finite(coefficient))
            throw std::invalid_argument(
                "polynomial coefficients must be finite");
    }
    if(coefficients.back() == std::complex<Real>{})
        throw std::invalid_argument(
            "polynomial leading coefficient must be nonzero");

    const std::size_t degree = coefficients.size() - 1;
    if(degree == 1)
        return {-coefficients[0] / coefficients[1]};

    const std::complex<Real> leading = coefficients.back();
    for(auto& coefficient : coefficients)
        coefficient /= leading;

    Real radius = Real(1);
    for(std::size_t index = 0; index < degree; ++index)
        radius = std::max(radius, Real(1) + std::abs(coefficients[index]));

    const Real pi = std::acos(Real(-1));
    std::vector<std::complex<Real>> roots(degree);
    std::vector<std::complex<Real>> updated(degree);
    for(std::size_t index = 0; index < degree; ++index)
    {
        const Real angle =
            Real(2) * pi *
            (static_cast<Real>(index) + Real(0.5)) /
            static_cast<Real>(degree);
        roots[index] = std::polar(radius, angle);
    }

    const Real epsilon = std::numeric_limits<Real>::epsilon();
    const Real correction_tolerance = Real(256) * epsilon;
    const Real separation_tolerance = Real(64) * epsilon;
    constexpr std::size_t maximum_iterations = 2000;
    bool converged = false;
    for(std::size_t iteration = 0;
        iteration < maximum_iterations;
        ++iteration)
    {
        bool all_small = true;
        for(std::size_t index = 0; index < degree; ++index)
        {
            const auto value_and_derivative =
                evaluate_polynomial_and_derivative(
                    coefficients,
                    roots[index]);
            const auto value = value_and_derivative.first;
            const auto derivative = value_and_derivative.second;

            std::complex<Real> correction{};
            if(std::abs(value) >
               correction_tolerance *
                   std::max(
                       Real(1),
                       polynomial_scale(coefficients, roots[index])))
            {
                if(std::abs(derivative) <= separation_tolerance)
                {
                    const Real angle =
                        Real(2) * pi *
                        (static_cast<Real>(index) + Real(0.25)) /
                        static_cast<Real>(degree);
                    correction =
                        std::polar(
                            separation_tolerance *
                                (Real(1) + std::abs(roots[index])),
                            angle);
                }
                else
                {
                    const auto newton = value / derivative;
                    std::complex<Real> interaction{};
                    for(std::size_t other = 0;
                        other < degree;
                        ++other)
                    {
                        if(other == index)
                            continue;
                        auto difference = roots[index] - roots[other];
                        if(std::abs(difference) <= separation_tolerance)
                        {
                            difference += std::complex<Real>(
                                separation_tolerance,
                                separation_tolerance);
                        }
                        interaction += Real(1) / difference;
                    }
                    const auto denominator =
                        std::complex<Real>(Real(1), Real{}) -
                        newton * interaction;
                    correction =
                        std::abs(denominator) > separation_tolerance
                        ? newton / denominator
                        : newton;
                }
            }

            const Real maximum_correction =
                Real(0.5) * (Real(1) + std::abs(roots[index]));
            if(std::abs(correction) > maximum_correction)
                correction *= maximum_correction / std::abs(correction);
            updated[index] = roots[index] - correction;
            if(
                std::abs(correction) >
                correction_tolerance *
                    (Real(1) + std::abs(updated[index])))
            {
                all_small = false;
            }
        }
        roots.swap(updated);
        if(all_small)
        {
            converged = true;
            break;
        }
    }

    if(!converged)
        throw std::runtime_error(
            "polynomial root extraction did not converge");

    for(auto& root : roots)
    {
        for(std::size_t iteration = 0; iteration < 8; ++iteration)
        {
            const auto value_and_derivative =
                evaluate_polynomial_and_derivative(coefficients, root);
            if(std::abs(value_and_derivative.second) <=
               separation_tolerance)
            {
                break;
            }
            const auto correction =
                value_and_derivative.first /
                value_and_derivative.second;
            root -= correction;
            if(
                std::abs(correction) <=
                correction_tolerance * (Real(1) + std::abs(root)))
            {
                break;
            }
        }

        const auto residual =
            evaluate_polynomial_and_derivative(coefficients, root).first;
        const Real scale =
            std::max(Real(1), polynomial_scale(coefficients, root));
        if(
            !finite(root) ||
            std::abs(residual) >
                Real(4096) * std::sqrt(epsilon) * scale)
        {
            throw std::runtime_error(
                "polynomial root extraction failed residual validation");
        }
    }
    return roots;
}

} // namespace detail

template<class Real, class Coefficient>
std::vector<complex_affine_factor<Real>>
polynomial_denominator_factors(
    const std::vector<Coefficient>& coefficients,
    Real step,
    std::size_t repetitions,
    const std::complex<Real>& shift)
{
    static_assert(
        std::is_convertible<Coefficient, std::complex<Real>>::value,
        "polynomial coefficients must convert to std::complex<Real>");

    if(coefficients.size() < 2)
        throw std::invalid_argument(
            "denominator factorization requires positive polynomial degree");
    if(!std::isfinite(step) || step == Real{})
        throw std::invalid_argument(
            "denominator factorization requires a finite nonzero step");
    if(repetitions == 0)
        throw std::invalid_argument(
            "denominator factorization repetitions must be positive");
    if(!detail::finite(shift))
        throw std::invalid_argument(
            "denominator factorization shift must be finite");

    std::vector<std::complex<Real>> complex_coefficients;
    complex_coefficients.reserve(coefficients.size());
    for(const auto& coefficient : coefficients)
    {
        const std::complex<Real> converted =
            static_cast<std::complex<Real>>(coefficient);
        if(!detail::finite(converted))
            throw std::invalid_argument(
                "denominator factorization coefficients must be finite");
        complex_coefficients.push_back(converted);
    }
    if(complex_coefficients.back() == std::complex<Real>{})
        throw std::invalid_argument(
            "denominator factorization leading coefficient must be nonzero");

    const auto outer_roots = complex_roots(shift, repetitions);
    const std::size_t degree = complex_coefficients.size() - 1;
    std::vector<complex_affine_factor<Real>> result;
    result.reserve(degree * repetitions);
    for(const auto& outer_root : outer_roots)
    {
        auto shifted_coefficients = complex_coefficients;
        shifted_coefficients.front() -= outer_root;
        const auto roots =
            detail::polynomial_roots(std::move(shifted_coefficients));

        for(std::size_t index = 0; index < roots.size(); ++index)
        {
            complex_affine_factor<Real> factor{
                std::complex<Real>(step, Real{}),
                -roots[index]};
            if(index == 0)
            {
                factor.operator_scale *= complex_coefficients.back();
                factor.diagonal_shift *= complex_coefficients.back();
            }
            result.push_back(factor);
        }
    }
    return result;
}

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
