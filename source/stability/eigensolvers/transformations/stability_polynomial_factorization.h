#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_STABILITY_POLYNOMIAL_FACTORIZATION_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_STABILITY_POLYNOMIAL_FACTORIZATION_H__

#include <complex>
#include <cstddef>
#include <vector>

#include "polynomial_denominator_factorization.h"
#include "stability_polynomial_operator.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class Real>
std::vector<complex_affine_factor<Real>> euler_denominator_factors(
    Real step,
    std::size_t repetitions,
    const std::complex<Real>& shift)
{
    return polynomial_denominator_factors<Real>(
        explicit_euler_stability_polynomial<Real>(),
        step,
        repetitions,
        shift);
}

template<class Real>
std::vector<complex_affine_factor<Real>> rk4_denominator_factors(
    Real step,
    std::size_t repetitions,
    const std::complex<Real>& shift)
{
    return polynomial_denominator_factors<Real>(
        classical_rk4_stability_polynomial<Real>(),
        step,
        repetitions,
        shift);
}

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
