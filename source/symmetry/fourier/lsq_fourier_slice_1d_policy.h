#ifndef __SYMMETRY_FOURIER_LSQ_FOURIER_SLICE_1D_POLICY_H__
#define __SYMMETRY_FOURIER_LSQ_FOURIER_SLICE_1D_POLICY_H__

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace symmetry
{
namespace fourier
{

template<class T>
struct lsq_fourier_slice_1d_policy
{
    std::size_t mode_min = 1;
    std::size_t mode_max = 0;
    std::size_t max_active_modes = 8;
    std::size_t grid_points = 64;
    std::size_t newton_iterations = 8;
    bool prefer_trivial_residual_group = true;
    T minimum_coprime_relative_score = T(0.05);

    void validate() const
    {
        if(mode_min == 0)
        {
            throw std::invalid_argument("LSQ mode_min must be positive");
        }
        if(mode_max != 0 && mode_max < mode_min)
        {
            throw std::invalid_argument("LSQ mode_max must be zero or not smaller than mode_min");
        }
        if(max_active_modes == 0)
        {
            throw std::invalid_argument("LSQ max_active_modes must be positive");
        }
        if(grid_points < 8)
        {
            throw std::invalid_argument("LSQ grid_points must be at least 8");
        }
        if(!finite(minimum_coprime_relative_score) ||
           minimum_coprime_relative_score < T(0) ||
           minimum_coprime_relative_score > T(1))
        {
            throw std::invalid_argument(
                "LSQ minimum coprime relative score must be finite and in [0,1]");
        }
    }

    bool operator==(const lsq_fourier_slice_1d_policy& other) const
    {
        return mode_min == other.mode_min &&
               mode_max == other.mode_max &&
               max_active_modes == other.max_active_modes &&
               grid_points == other.grid_points &&
               newton_iterations == other.newton_iterations &&
               prefer_trivial_residual_group == other.prefer_trivial_residual_group &&
               minimum_coprime_relative_score == other.minimum_coprime_relative_score;
    }

    bool operator!=(const lsq_fourier_slice_1d_policy& other) const
    {
        return !(*this == other);
    }

private:
    static bool finite(const T value)
    {
        using std::isfinite;
        return isfinite(value);
    }
};

} // namespace fourier
} // namespace symmetry

#endif
