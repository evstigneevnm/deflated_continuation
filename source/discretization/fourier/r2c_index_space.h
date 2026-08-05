#ifndef __DISCRETIZATION_FOURIER_R2C_INDEX_SPACE_H__
#define __DISCRETIZATION_FOURIER_R2C_INDEX_SPACE_H__

#include <cstddef>
#include <stdexcept>

#include <discretization/common/structured_extent.h>

namespace discretization
{
namespace fourier
{

class r2c_index_space_2d
{
public:
    using extent_type = discretization::common::structured_extent<2>;

    explicit r2c_index_space_2d(const extent_type& physical_extent):
        nx_(physical_extent[0]),
        ny_(physical_extent[1]),
        my_(ny_/2 + 1)
    {
        if(nx_ < 4 || ny_ < 4 || nx_%2 != 0 || ny_%2 != 0)
        {
            throw std::invalid_argument("r2c_index_space_2d expects even dimensions of at least four");
        }
    }

    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }
    std::size_t my() const { return my_; }
    std::size_t physical_size() const { return nx_*ny_; }
    std::size_t complex_size() const { return nx_*my_; }

    extent_type physical_extent() const { return extent_type(nx_, ny_); }
    extent_type spectral_extent() const { return extent_type(nx_, my_); }

    std::size_t flat_index(const std::size_t ix, const std::size_t iy) const
    {
        if(ix >= nx_ || iy >= my_)
        {
            throw std::out_of_range("r2c_index_space_2d index is outside the half spectrum");
        }
        return ix*my_ + iy;
    }

    int signed_x_mode(const std::size_t ix) const
    {
        if(ix >= nx_)
        {
            throw std::out_of_range("r2c_index_space_2d x index is outside the spectrum");
        }
        return ix <= nx_/2 ? static_cast<int>(ix) : static_cast<int>(ix) - static_cast<int>(nx_);
    }

    int y_mode(const std::size_t iy) const
    {
        if(iy >= my_)
        {
            throw std::out_of_range("r2c_index_space_2d y index is outside the spectrum");
        }
        return static_cast<int>(iy);
    }

    bool is_y_boundary(const std::size_t iy) const
    {
        return iy == 0 || iy == ny_/2;
    }

    bool is_self_conjugate_x(const std::size_t ix) const
    {
        return ix == 0 || ix == nx_/2;
    }

    std::size_t conjugate_x(const std::size_t ix) const
    {
        return ix == 0 ? 0 : nx_ - ix;
    }

    std::size_t full_mean_zero_state_size() const
    {
        return physical_size() - 1;
    }

    std::size_t translation_equivariant_mean_zero_state_size() const
    {
        return (nx() - 1)*(ny() - 1) - 1;
    }

    std::size_t inversion_odd_state_size() const
    {
        return physical_size()/2 - 2;
    }

private:
    std::size_t nx_;
    std::size_t ny_;
    std::size_t my_;
};

} // namespace fourier
} // namespace discretization

#endif
