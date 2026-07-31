#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_AFFINE_FACTOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_AFFINE_FACTOR_H__

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class Real>
struct complex_affine_factor
{
    std::complex<Real> operator_scale;
    std::complex<Real> diagonal_shift;
};

template<class Real>
bool is_finite(const complex_affine_factor<Real>& factor)
{
    return
        std::isfinite(factor.operator_scale.real()) &&
        std::isfinite(factor.operator_scale.imag()) &&
        std::isfinite(factor.diagonal_shift.real()) &&
        std::isfinite(factor.diagonal_shift.imag());
}

template<class Real>
void validate(const complex_affine_factor<Real>& factor)
{
    if(!is_finite(factor))
        throw std::invalid_argument(
            "complex affine factor coefficients must be finite");
}

template<class Real>
std::complex<Real> evaluate_affine_factor(
    const complex_affine_factor<Real>& factor,
    const std::complex<Real>& value)
{
    return
        factor.operator_scale * value +
        factor.diagonal_shift;
}

template<class Real>
std::complex<Real> evaluate_affine_factors(
    const std::vector<complex_affine_factor<Real>>& factors,
    const std::complex<Real>& value)
{
    std::complex<Real> result(Real(1), Real{});
    for(const auto& factor : factors)
        result *= evaluate_affine_factor(factor, value);
    return result;
}

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
