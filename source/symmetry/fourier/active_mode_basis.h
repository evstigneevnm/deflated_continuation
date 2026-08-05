#ifndef __SYMMETRY_FOURIER_ACTIVE_MODE_BASIS_H__
#define __SYMMETRY_FOURIER_ACTIVE_MODE_BASIS_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <symmetry/fourier/mode_descriptor.h>

namespace symmetry
{
namespace fourier
{

template<class Real, std::size_t Dimension>
struct active_mode_observation
{
    mode_index<Dimension> mode;
    Real amplitude = Real(0);
    std::size_t source_index = 0;
};

template<class Real, std::size_t Dimension>
struct active_mode_basis_result
{
    std::size_t group_dimension = Dimension;
    std::size_t active_rank = 0;
    Real maximum_amplitude = Real(0);
    std::array<active_mode_observation<Real, Dimension>, Dimension> selected{};
};

namespace detail
{

template<class Real, std::size_t Dimension>
bool lexicographic_mode_less(
    const active_mode_observation<Real, Dimension>& left,
    const active_mode_observation<Real, Dimension>& right
)
{
    for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
    {
        if(left.mode[dimension] != right.mode[dimension])
        {
            return left.mode[dimension] < right.mode[dimension];
        }
    }
    return left.source_index < right.source_index;
}

template<class Real, std::size_t Dimension>
Real vector_norm(const std::array<Real, Dimension>& value)
{
    Real norm_squared = Real(0);
    for(const Real component: value)
    {
        norm_squared += component*component;
    }
    using std::sqrt;
    return sqrt(norm_squared);
}

} // namespace detail

template<class Real, std::size_t Dimension>
active_mode_basis_result<Real, Dimension> select_active_mode_basis(
    std::vector<active_mode_observation<Real, Dimension>> observations,
    const Real absolute_amplitude_tolerance,
    const Real relative_amplitude_tolerance,
    const Real rank_tolerance
)
{
    static_assert(Dimension > 0, "active mode basis requires a positive dimension");
    if(absolute_amplitude_tolerance < Real(0) || relative_amplitude_tolerance < Real(0) ||
       rank_tolerance < Real(0))
    {
        throw std::invalid_argument("active mode basis tolerances must be nonnegative");
    }

    active_mode_basis_result<Real, Dimension> result;
    for(auto& observation: observations)
    {
        if(observation.amplitude < Real(0))
        {
            observation.amplitude = -observation.amplitude;
        }
        result.maximum_amplitude = std::max(result.maximum_amplitude, observation.amplitude);
    }
    const Real amplitude_threshold = std::max(
        absolute_amplitude_tolerance,
        relative_amplitude_tolerance*result.maximum_amplitude
    );
    std::stable_sort(
        observations.begin(),
        observations.end(),
        [](const auto& left, const auto& right)
        {
            if(left.amplitude != right.amplitude)
            {
                return left.amplitude > right.amplitude;
            }
            return detail::lexicographic_mode_less(left, right);
        }
    );

    std::array<std::array<Real, Dimension>, Dimension> orthonormal{};
    for(const auto& observation: observations)
    {
        if(result.active_rank == Dimension || observation.amplitude < amplitude_threshold)
        {
            break;
        }
        std::array<Real, Dimension> residual{};
        for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
        {
            residual[dimension] = static_cast<Real>(observation.mode[dimension]);
        }
        const Real original_norm = detail::vector_norm(residual);
        if(original_norm == Real(0))
        {
            continue;
        }
        for(std::size_t basis_index = 0; basis_index < result.active_rank; ++basis_index)
        {
            Real projection = Real(0);
            for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
            {
                projection += residual[dimension]*orthonormal[basis_index][dimension];
            }
            for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
            {
                residual[dimension] -= projection*orthonormal[basis_index][dimension];
            }
        }
        const Real residual_norm = detail::vector_norm(residual);
        if(residual_norm <= rank_tolerance*std::max(original_norm, Real(1)))
        {
            continue;
        }
        for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
        {
            orthonormal[result.active_rank][dimension] = residual[dimension]/residual_norm;
        }
        result.selected[result.active_rank] = observation;
        ++result.active_rank;
    }
    return result;
}

} // namespace fourier
} // namespace symmetry

#endif
