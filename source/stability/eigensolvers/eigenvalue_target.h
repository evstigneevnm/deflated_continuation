#ifndef __STABILITY_EIGENSOLVERS_EIGENVALUE_TARGET_H__
#define __STABILITY_EIGENSOLVERS_EIGENVALUE_TARGET_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace stability
{
namespace eigensolvers
{

enum class spectrum_target
{
    largest_magnitude,
    smallest_magnitude,
    largest_real,
    smallest_real,
    closest_to_shift
};

template<class Real>
struct eigenvalue_target
{
    spectrum_target kind = spectrum_target::largest_magnitude;
    std::complex<Real> shift{};

    bool precedes(
        const std::complex<Real>& left,
        const std::complex<Real>& right) const
    {
        const Real left_primary = primary_score(left);
        const Real right_primary = primary_score(right);
        if(left_primary != right_primary)
            return left_primary < right_primary;

        if(left.real() != right.real())
            return left.real() > right.real();
        if(std::abs(left.imag()) != std::abs(right.imag()))
            return std::abs(left.imag()) > std::abs(right.imag());
        return left.imag() > right.imag();
    }

    template<class EigenvalueContainer>
    std::vector<std::size_t> ordered_indices(
        const EigenvalueContainer& eigenvalues) const
    {
        std::vector<std::size_t> result(eigenvalues.size());
        std::iota(result.begin(), result.end(), std::size_t(0));
        std::stable_sort(
            result.begin(),
            result.end(),
            [this, &eigenvalues](std::size_t left, std::size_t right)
            {
                return precedes(eigenvalues[left], eigenvalues[right]);
            });
        return result;
    }

private:
    Real primary_score(const std::complex<Real>& value) const
    {
        switch(kind)
        {
        case spectrum_target::largest_magnitude:
            return -std::abs(value);
        case spectrum_target::smallest_magnitude:
            return std::abs(value);
        case spectrum_target::largest_real:
            return -value.real();
        case spectrum_target::smallest_real:
            return value.real();
        case spectrum_target::closest_to_shift:
            return std::abs(value - shift);
        }
        throw std::logic_error("unsupported spectrum target");
    }
};

} // namespace eigensolvers
} // namespace stability

#endif
