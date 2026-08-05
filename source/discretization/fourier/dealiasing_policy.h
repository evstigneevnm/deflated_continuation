#ifndef __DISCRETIZATION_FOURIER_DEALIASING_POLICY_H__
#define __DISCRETIZATION_FOURIER_DEALIASING_POLICY_H__

#include <cstddef>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <discretization/fourier/r2c_index_space.h>
#include <scfd/utils/device_tag.h>

namespace discretization
{
namespace fourier
{

class no_dealiasing_2d
{
public:
    template<class Backend, class SpectralField>
    void apply(SpectralField&) const
    {
    }
};

class two_thirds_dealiasing_2d
{
public:
    explicit two_thirds_dealiasing_2d(const r2c_index_space_2d& index_space):
        nx_(static_cast<std::ptrdiff_t>(index_space.nx())),
        ny_(static_cast<std::ptrdiff_t>(index_space.ny())),
        my_(static_cast<std::ptrdiff_t>(index_space.my()))
    {
    }

    template<class Backend, class SpectralField>
    void apply(SpectralField& spectrum) const
    {
        using complex_type = typename SpectralField::value_type;
        using traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
        using scalar_type = typename traits::real_type;
        using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;

        if(static_cast<std::ptrdiff_t>(spectrum.size()) != nx_*my_)
        {
            throw std::invalid_argument("two_thirds_dealiasing_2d spectrum size mismatch");
        }

        complex_type* values = spectrum.data();
        const std::ptrdiff_t nx = nx_;
        const std::ptrdiff_t ny = ny_;
        const std::ptrdiff_t my = my_;
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
            {
                const std::ptrdiff_t ix = index/my;
                const std::ptrdiff_t iy = index - ix*my;
                const std::ptrdiff_t kx = ix <= nx/2 ? ix : ix - nx;
                const std::ptrdiff_t abs_kx = kx < 0 ? -kx : kx;
                if(3*abs_kx >= nx || 3*iy >= ny)
                {
                    values[index] = traits::make(scalar_type(0), scalar_type(0));
                }
            },
            nx*my
        );
        for_each.wait();
    }

private:
    std::ptrdiff_t nx_;
    std::ptrdiff_t ny_;
    std::ptrdiff_t my_;
};

} // namespace fourier
} // namespace discretization

#endif
