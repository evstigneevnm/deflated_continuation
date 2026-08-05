#ifndef __DISCRETIZATION_FOURIER_WAVEVECTOR_TABLE_H__
#define __DISCRETIZATION_FOURIER_WAVEVECTOR_TABLE_H__

#include <cstddef>
#include <vector>

#include <discretization/fourier/periodic_grid.h>
#include <discretization/fourier/r2c_index_space.h>
#include <discretization/fourier/spectral_field.h>

namespace discretization
{
namespace fourier
{

template<class Backend, class T>
class wavevector_table_2d
{
public:
    static constexpr std::size_t dimension = 2;
    using backend_type = Backend;
    using scalar_type = T;
    using grid_type = periodic_grid<scalar_type, 2>;
    using field_type = spectral_field<backend_type, scalar_type, 2>;
    using copy_type = typename backend_type::copy_type;

    wavevector_table_2d(const grid_type& grid, const r2c_index_space_2d& index_space):
        index_space_(index_space),
        kx_(index_space.spectral_extent()),
        ky_(index_space.spectral_extent()),
        k_squared_(index_space.spectral_extent())
    {
        std::vector<scalar_type> host_kx(index_space_.complex_size());
        std::vector<scalar_type> host_ky(index_space_.complex_size());
        std::vector<scalar_type> host_k_squared(index_space_.complex_size());
        for(std::size_t ix = 0; ix < index_space_.nx(); ++ix)
        {
            const scalar_type kx = grid.wave_number(0, index_space_.signed_x_mode(ix));
            for(std::size_t iy = 0; iy < index_space_.my(); ++iy)
            {
                const scalar_type ky = grid.wave_number(1, index_space_.y_mode(iy));
                const std::size_t index = index_space_.flat_index(ix, iy);
                host_kx[index] = kx;
                host_ky[index] = ky;
                host_k_squared[index] = kx*kx + ky*ky;
            }
        }
        copy_type()(static_cast<std::ptrdiff_t>(host_kx.size()), host_kx.data(), kx_.data());
        copy_type()(static_cast<std::ptrdiff_t>(host_ky.size()), host_ky.data(), ky_.data());
        copy_type()(static_cast<std::ptrdiff_t>(host_k_squared.size()), host_k_squared.data(), k_squared_.data());
    }

    const r2c_index_space_2d& index_space() const { return index_space_; }
    const field_type& kx() const { return kx_; }
    const field_type& ky() const { return ky_; }
    const field_type& k_squared() const { return k_squared_; }

    const field_type& component(const std::size_t dimension_index) const
    {
        if(dimension_index == 0)
        {
            return kx_;
        }
        if(dimension_index == 1)
        {
            return ky_;
        }
        throw std::out_of_range("wavevector_table_2d component index");
    }

private:
    r2c_index_space_2d index_space_;
    field_type kx_;
    field_type ky_;
    field_type k_squared_;
};

} // namespace fourier
} // namespace discretization

#endif
