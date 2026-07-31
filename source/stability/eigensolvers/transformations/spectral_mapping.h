#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SPECTRAL_MAPPING_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SPECTRAL_MAPPING_H__

#include <complex>
#include <limits>
#include <stdexcept>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class Real>
std::complex<Real> cayley_map(
    const std::complex<Real>& eigenvalue,
    Real sigma,
    Real sigma_zero)
{
    return
        (eigenvalue + std::complex<Real>(sigma_zero)) /
        (eigenvalue - std::complex<Real>(sigma));
}

template<class Real>
std::complex<Real> cayley_inverse_map(
    const std::complex<Real>& mapped,
    Real sigma,
    Real sigma_zero)
{
    const std::complex<Real> denominator =
        mapped - std::complex<Real>(Real(1));
    if(
        std::abs(denominator) <=
        Real(64) * std::numeric_limits<Real>::epsilon())
    {
        throw std::domain_error(
            "Cayley inverse map is singular at mapped eigenvalue one");
    }
    return
        (mapped * std::complex<Real>(sigma) +
         std::complex<Real>(sigma_zero)) /
        denominator;
}

template<class Real>
std::complex<Real> shift_inverse_map(
    const std::complex<Real>& eigenvalue,
    Real sigma)
{
    return
        std::complex<Real>(Real(1)) /
        (eigenvalue - std::complex<Real>(sigma));
}

template<class Real>
std::complex<Real> complex_shift_inverse_map(
    const std::complex<Real>& eigenvalue,
    const std::complex<Real>& shift)
{
    return
        std::complex<Real>(Real(1)) /
        (eigenvalue - shift);
}

template<class Real>
std::complex<Real> complex_shift_inverse_inverse_map(
    const std::complex<Real>& mapped,
    const std::complex<Real>& shift)
{
    if(
        std::abs(mapped) <=
        Real(64) * std::numeric_limits<Real>::epsilon())
    {
        throw std::domain_error(
            "complex shift-inverse recovery is singular at mapped "
            "eigenvalue zero");
    }
    return shift + std::complex<Real>(Real(1)) / mapped;
}

template<class Real>
std::complex<Real> shift_inverse_inverse_map(
    const std::complex<Real>& mapped,
    Real sigma)
{
    if(
        std::abs(mapped) <=
        Real(64) * std::numeric_limits<Real>::epsilon())
    {
        throw std::domain_error(
            "shift-inverse recovery is singular at mapped eigenvalue zero");
    }
    return
        std::complex<Real>(sigma) +
        std::complex<Real>(Real(1)) / mapped;
}

template<class Real>
std::complex<Real> inexact_exponential_map(
    const std::complex<Real>& eigenvalue,
    Real step,
    std::size_t power,
    const std::complex<Real>& shift)
{
    if(power == 0)
        throw std::invalid_argument(
            "inexact exponential map requires positive power");
    return
        std::complex<Real>(Real(1)) /
        (std::pow(
             std::complex<Real>(Real(1)) + step * eigenvalue,
             power) -
         shift);
}

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
