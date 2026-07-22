#ifndef __SYMMETRY_FOURIER_FOURIER_ISOTROPY_1D_H__
#define __SYMMETRY_FOURIER_FOURIER_ISOTROPY_1D_H__

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace symmetry
{
namespace fourier
{

template<class Complex>
std::size_t approximate_fourier_isotropy_order_1d(
    const std::vector<Complex>& values,
    const typename Complex::value_type relative_tolerance)
{
    using scalar_type = typename Complex::value_type;
    scalar_type norm_sq = scalar_type(0);
    for(std::size_t mode = 1; mode < values.size(); ++mode)
    {
        norm_sq += std::norm(values[mode]);
    }
    if(!(norm_sq > scalar_type(0)))
    {
        return 1;
    }

    const scalar_type threshold_sq =
        relative_tolerance*relative_tolerance*norm_sq;
    std::size_t order = 0;
    for(std::size_t mode = 1; mode < values.size(); ++mode)
    {
        if(std::norm(values[mode]) > threshold_sq)
        {
            order = order == 0 ? mode : std::gcd(order, mode);
            if(order == 1)
            {
                return 1;
            }
        }
    }
    return order == 0 ? std::size_t(1) : order;
}

template<class Complex>
typename Complex::value_type relative_fourier_transverse_norm_1d(
    const std::vector<Complex>& values,
    const std::size_t isotropy_order)
{
    using scalar_type = typename Complex::value_type;
    if(isotropy_order <= 1)
    {
        return scalar_type(0);
    }

    scalar_type norm_sq = scalar_type(0);
    scalar_type transverse_sq = scalar_type(0);
    for(std::size_t mode = 1; mode < values.size(); ++mode)
    {
        const scalar_type mode_norm_sq = std::norm(values[mode]);
        norm_sq += mode_norm_sq;
        if(mode%isotropy_order != 0)
        {
            transverse_sq += mode_norm_sq;
        }
    }
    return norm_sq > scalar_type(0)
        ? std::sqrt(transverse_sq/norm_sq)
        : scalar_type(0);
}

} // namespace fourier
} // namespace symmetry

#endif
