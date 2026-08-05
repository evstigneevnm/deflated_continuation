#ifndef __DISCRETIZATION_FOURIER_PERIODIC_GRID_H__
#define __DISCRETIZATION_FOURIER_PERIODIC_GRID_H__

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <discretization/common/structured_extent.h>

namespace discretization
{
namespace fourier
{

template<class T, std::size_t Dimension>
class periodic_grid
{
public:
    using scalar_type = T;
    using extent_type = discretization::common::structured_extent<Dimension>;
    using lengths_type = std::array<scalar_type, Dimension>;

    periodic_grid(const extent_type& extent, const lengths_type& lengths):
        extent_(extent),
        lengths_(lengths)
    {
        for(const scalar_type length: lengths_)
        {
            if(!(length > scalar_type(0)))
            {
                throw std::invalid_argument("periodic_grid lengths must be positive");
            }
        }
    }

    const extent_type& extent() const
    {
        return extent_;
    }

    scalar_type length(const std::size_t dimension) const
    {
        return lengths_.at(dimension);
    }

    scalar_type spacing(const std::size_t dimension) const
    {
        return length(dimension)/static_cast<scalar_type>(extent_[dimension]);
    }

    scalar_type coordinate(const std::size_t dimension, const std::size_t index) const
    {
        return spacing(dimension)*static_cast<scalar_type>(index);
    }

    scalar_type wave_number(const std::size_t dimension, const int signed_mode) const
    {
        const scalar_type pi = std::acos(scalar_type(-1));
        return scalar_type(2)*pi*static_cast<scalar_type>(signed_mode)/length(dimension);
    }

private:
    extent_type extent_;
    lengths_type lengths_;
};

} // namespace fourier
} // namespace discretization

#endif
