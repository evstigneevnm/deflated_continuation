#ifndef __DISCRETIZATION_FOURIER_NORMALIZED_FFT_H__
#define __DISCRETIZATION_FOURIER_NORMALIZED_FFT_H__

#include <cstddef>
#include <stdexcept>
#include <string>

#include <common/scfd_backend_ext/complex.h>
#include <discretization/common/structured_extent.h>
#include <discretization/fourier/spectral_field.h>
#include <external_libraries/fft_facade.h>
#include <scfd/utils/device_tag.h>

namespace discretization
{
namespace fourier
{

namespace detail
{

template<std::size_t Dimension>
external_libraries::fft::dimensions make_fft_dimensions(
    const discretization::common::structured_extent<Dimension>& extent
);

template<>
inline external_libraries::fft::dimensions make_fft_dimensions<1>(
    const discretization::common::structured_extent<1>& extent
)
{
    return external_libraries::fft::dimensions(extent[0]);
}

template<>
inline external_libraries::fft::dimensions make_fft_dimensions<2>(
    const discretization::common::structured_extent<2>& extent
)
{
    return external_libraries::fft::dimensions(extent[0], extent[1]);
}

template<>
inline external_libraries::fft::dimensions make_fft_dimensions<3>(
    const discretization::common::structured_extent<3>& extent
)
{
    return external_libraries::fft::dimensions(extent[0], extent[1], extent[2]);
}

template<std::size_t Dimension>
discretization::common::structured_extent<Dimension> make_spectral_extent(
    const discretization::common::structured_extent<Dimension>& physical_extent
)
{
    typename discretization::common::structured_extent<Dimension>::dimensions_type dimensions =
        physical_extent.dimensions();
    dimensions[Dimension - 1] = dimensions[Dimension - 1]/2 + 1;
    return discretization::common::structured_extent<Dimension>(dimensions);
}

} // namespace detail

template<class Backend, class FFTBackend, class T, std::size_t Dimension>
class normalized_r2c_transform
{
public:
    using backend_type = Backend;
    using fft_backend_type = FFTBackend;
    using scalar_type = T;
    using fft_type = external_libraries::fft::r2c<fft_backend_type, scalar_type>;
    using complex_type = typename fft_type::complex_type;
    using extent_type = discretization::common::structured_extent<Dimension>;
    using physical_field_type = physical_field<backend_type, scalar_type, Dimension>;
    using spectral_field_type = spectral_field<backend_type, complex_type, Dimension>;
    using copy_type = typename backend_type::copy_type;
    using for_each_type = typename backend_type::template for_each_type<std::ptrdiff_t>;

    explicit normalized_r2c_transform(const extent_type& physical_extent):
        physical_extent_(physical_extent),
        spectral_extent_(detail::make_spectral_extent(physical_extent)),
        fft_(detail::make_fft_dimensions(physical_extent)),
        physical_stage_(physical_extent_),
        spectral_stage_(spectral_extent_)
    {
    }

    normalized_r2c_transform(const normalized_r2c_transform&) = delete;
    normalized_r2c_transform& operator=(const normalized_r2c_transform&) = delete;

    const extent_type& physical_extent() const { return physical_extent_; }
    const extent_type& spectral_extent() const { return spectral_extent_; }
    std::size_t physical_size() const { return physical_extent_.size(); }
    std::size_t complex_size() const { return spectral_extent_.size(); }

    void forward(const physical_field_type& source, spectral_field_type& destination)
    {
        require_extent(source.extent(), physical_extent_, "forward source");
        require_extent(destination.extent(), spectral_extent_, "forward destination");
        copy_type()(static_cast<std::ptrdiff_t>(physical_size()), source.data(), physical_stage_.data());
        fft_.forward(physical_stage_.data(), spectral_stage_.data());
        copy_type()(static_cast<std::ptrdiff_t>(complex_size()), spectral_stage_.data(), destination.data());
    }

    void inverse(const spectral_field_type& source, physical_field_type& destination)
    {
        require_extent(source.extent(), spectral_extent_, "inverse source");
        require_extent(destination.extent(), physical_extent_, "inverse destination");
        copy_type()(static_cast<std::ptrdiff_t>(complex_size()), source.data(), spectral_stage_.data());
        fft_.inverse(spectral_stage_.data(), physical_stage_.data());

        scalar_type* values = physical_stage_.data();
        const scalar_type inverse_size = scalar_type(1)/static_cast<scalar_type>(physical_size());
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
            {
                values[index] *= inverse_size;
            },
            static_cast<std::ptrdiff_t>(physical_size())
        );
        for_each.wait();
        copy_type()(static_cast<std::ptrdiff_t>(physical_size()), physical_stage_.data(), destination.data());
    }

    // Adjoint with respect to Euclidean physical coordinates and the real
    // inner product of the stored Hermitian half spectrum.
    void forward_adjoint(const spectral_field_type& source, physical_field_type& destination)
    {
        require_extent(source.extent(), spectral_extent_, "forward_adjoint source");
        require_extent(destination.extent(), physical_extent_, "forward_adjoint destination");
        copy_type()(static_cast<std::ptrdiff_t>(complex_size()), source.data(), spectral_stage_.data());

        using complex_traits =
            ::common::scfd_backend_ext::complex_value_traits<complex_type>;
        complex_type* values = spectral_stage_.data();
        const std::ptrdiff_t last_spectral_size =
            static_cast<std::ptrdiff_t>(spectral_extent_[Dimension - 1]);
        const std::ptrdiff_t last_physical_size =
            static_cast<std::ptrdiff_t>(physical_extent_[Dimension - 1]);
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
            {
                const std::ptrdiff_t last_index = index%last_spectral_size;
                if(last_index != 0 && 2*last_index != last_physical_size)
                {
                    values[index] = complex_traits::make(
                        scalar_type(0.5)*complex_traits::real(values[index]),
                        scalar_type(0.5)*complex_traits::imag(values[index]));
                }
            },
            static_cast<std::ptrdiff_t>(complex_size()));
        for_each.wait();
        fft_.inverse(spectral_stage_.data(), physical_stage_.data());
        copy_type()(static_cast<std::ptrdiff_t>(physical_size()), physical_stage_.data(), destination.data());
    }

    void inverse_adjoint(const physical_field_type& source, spectral_field_type& destination)
    {
        require_extent(source.extent(), physical_extent_, "inverse_adjoint source");
        require_extent(destination.extent(), spectral_extent_, "inverse_adjoint destination");
        copy_type()(static_cast<std::ptrdiff_t>(physical_size()), source.data(), physical_stage_.data());
        fft_.forward(physical_stage_.data(), spectral_stage_.data());

        using complex_traits =
            ::common::scfd_backend_ext::complex_value_traits<complex_type>;
        complex_type* values = spectral_stage_.data();
        const std::ptrdiff_t last_spectral_size =
            static_cast<std::ptrdiff_t>(spectral_extent_[Dimension - 1]);
        const std::ptrdiff_t last_physical_size =
            static_cast<std::ptrdiff_t>(physical_extent_[Dimension - 1]);
        const scalar_type inverse_size =
            scalar_type(1)/static_cast<scalar_type>(physical_size());
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
            {
                const std::ptrdiff_t last_index = index%last_spectral_size;
                const scalar_type weight =
                    (last_index == 0 || 2*last_index == last_physical_size)
                        ? inverse_size
                        : scalar_type(2)*inverse_size;
                values[index] = complex_traits::make(
                    weight*complex_traits::real(values[index]),
                    weight*complex_traits::imag(values[index]));
            },
            static_cast<std::ptrdiff_t>(complex_size()));
        for_each.wait();
        copy_type()(static_cast<std::ptrdiff_t>(complex_size()), spectral_stage_.data(), destination.data());
    }

private:
    static void require_extent(const extent_type& actual, const extent_type& expected, const char* label)
    {
        if(actual != expected)
        {
            throw std::invalid_argument(std::string("normalized_r2c_transform ") + label + " extent mismatch");
        }
    }

    extent_type physical_extent_;
    extent_type spectral_extent_;
    fft_type fft_;
    physical_field_type physical_stage_;
    spectral_field_type spectral_stage_;
};

} // namespace fourier
} // namespace discretization

#endif
