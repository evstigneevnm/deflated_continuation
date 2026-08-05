#ifndef __DISCRETIZATION_FOURIER_TESTS_FOURIER_FOUNDATION_TEST_SUITE_H__
#define __DISCRETIZATION_FOURIER_TESTS_FOURIER_FOUNDATION_TEST_SUITE_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <common/NMFD-operations/nmfd/operations/scfd_vector_operations.h>
#include <common/scfd_backend_ext/complex.h>
#include <discretization/fourier/codecs/full_real_field.h>
#include <discretization/fourier/codecs/inversion_odd_field.h>
#include <discretization/fourier/codecs/translation_equivariant_real_field.h>
#include <discretization/fourier/dealiasing_policy.h>
#include <discretization/fourier/normalized_fft.h>
#include <discretization/fourier/operations/derivative.h>
#include <discretization/fourier/operations/inverse_laplacian.h>
#include <discretization/fourier/operations/laplacian.h>
#include <discretization/fourier/operations/pseudospectral_product.h>
#include <discretization/fourier/periodic_grid.h>
#include <discretization/fourier/r2c_index_space.h>
#include <discretization/fourier/tests/legacy_ks2d_codec_adapter.h>
#include <discretization/fourier/wavevector_table.h>

namespace discretization
{
namespace fourier
{
namespace tests
{

class test_report
{
public:
    void check(const bool condition, const std::string& message)
    {
        ++checks_;
        if(!condition)
        {
            ++failures_;
            std::cerr << "FAIL: " << message << std::endl;
        }
    }

    template<class T>
    void near(const T actual, const T expected, const T tolerance, const std::string& message)
    {
        check(std::abs(actual - expected) <= tolerance, message);
    }

    int finish(const std::string& backend_name) const
    {
        std::cout << backend_name << ": " << checks_ << " checks, " << failures_ << " failures" << std::endl;
        return failures_ == 0 ? 0 : 1;
    }

private:
    std::size_t checks_ = 0;
    std::size_t failures_ = 0;
};

template<class Backend, class FFTBackend>
int run_fourier_foundation_tests(const std::string& backend_name)
{
    using scalar_type = double;
    using extent_type = discretization::common::structured_extent<2>;
    using transform_type = normalized_r2c_transform<Backend, FFTBackend, scalar_type, 2>;
    using complex_type = typename transform_type::complex_type;
    using complex_traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using physical_field_type = typename transform_type::physical_field_type;
    using spectral_field_type = typename transform_type::spectral_field_type;
    using vector_operations_type = scfd_vector_operations<Backend, scalar_type>;
    using copy_type = typename Backend::copy_type;

    constexpr std::size_t nx = 12;
    constexpr std::size_t ny = 16;
    const extent_type physical_extent(nx, ny);
    const r2c_index_space_2d index_space(physical_extent);
    const scalar_type tolerance = 2.0e-11;
    test_report report;

    report.check(index_space.my() == ny/2 + 1, "R2C reduced y extent");
    report.check(index_space.complex_size() == nx*(ny/2 + 1), "R2C complex size");
    report.check(index_space.full_mean_zero_state_size() == nx*ny - 1, "full mean-zero state size");
    report.check(
        index_space.translation_equivariant_mean_zero_state_size() == (nx - 1)*(ny - 1) - 1,
        "translation-equivariant mean-zero state size"
    );
    report.check(index_space.inversion_odd_state_size() == nx*ny/2 - 2, "inversion-odd state size");
    report.check(index_space.signed_x_mode(nx - 1) == -1, "signed x wave number");

    periodic_grid<scalar_type, 2> grid(physical_extent, {scalar_type(2)*std::acos(-1.0), scalar_type(4)*std::acos(-1.0)});
    report.near(grid.wave_number(0, 3), scalar_type(3), tolerance, "x wave number scaling");
    report.near(grid.wave_number(1, 3), scalar_type(1.5), tolerance, "y wave number scaling");

    transform_type transform(physical_extent);
    physical_field_type physical(physical_extent);
    physical_field_type restored(physical_extent);
    spectral_field_type spectrum(index_space.spectral_extent());
    std::vector<scalar_type> physical_host(nx*ny);
    const scalar_type pi = std::acos(-1.0);
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            physical_host[ix*ny + iy] =
                std::sin(scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx)) +
                scalar_type(0.3)*std::cos(scalar_type(6)*pi*static_cast<scalar_type>(iy)/static_cast<scalar_type>(ny));
        }
    }
    copy_type()(static_cast<std::ptrdiff_t>(physical_host.size()), physical_host.data(), physical.data());
    transform.forward(physical, spectrum);
    transform.inverse(spectrum, restored);
    std::vector<scalar_type> restored_host(physical_host.size());
    copy_type()(static_cast<std::ptrdiff_t>(restored_host.size()), restored.data(), restored_host.data());
    for(std::size_t index = 0; index < physical_host.size(); ++index)
    {
        report.near(restored_host[index], physical_host[index], tolerance, "normalized FFT round trip");
    }

    wavevector_table_2d<Backend, scalar_type> wavevectors(grid, index_space);
    spectral_field_type derivative_spectrum(index_space.spectral_extent());
    spectral_field_type laplacian_spectrum(index_space.spectral_extent());
    spectral_field_type recovered_spectrum(index_space.spectral_extent());
    physical_field_type derivative_physical(physical_extent);
    physical_field_type laplacian_physical(physical_extent);
    physical_field_type recovered_physical(physical_extent);

    operations::derivative<Backend>(spectrum, wavevectors, 0, derivative_spectrum);
    transform.inverse(derivative_spectrum, derivative_physical);
    operations::laplacian<Backend>(spectrum, wavevectors, laplacian_spectrum);
    transform.inverse(laplacian_spectrum, laplacian_physical);
    operations::inverse_laplacian<Backend>(laplacian_spectrum, wavevectors, recovered_spectrum);
    transform.inverse(recovered_spectrum, recovered_physical);
    std::vector<scalar_type> derivative_host(physical_host.size());
    std::vector<scalar_type> laplacian_host(physical_host.size());
    std::vector<scalar_type> recovered_operator_host(physical_host.size());
    copy_type()(static_cast<std::ptrdiff_t>(derivative_host.size()), derivative_physical.data(), derivative_host.data());
    copy_type()(static_cast<std::ptrdiff_t>(laplacian_host.size()), laplacian_physical.data(), laplacian_host.data());
    copy_type()(static_cast<std::ptrdiff_t>(recovered_operator_host.size()), recovered_physical.data(), recovered_operator_host.data());
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            const std::size_t index = ix*ny + iy;
            const scalar_type x_phase = scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx);
            const scalar_type y_phase = scalar_type(6)*pi*static_cast<scalar_type>(iy)/static_cast<scalar_type>(ny);
            report.near(derivative_host[index], std::cos(x_phase), tolerance, "spectral x derivative");
            report.near(
                laplacian_host[index],
                -std::sin(x_phase) - scalar_type(0.675)*std::cos(y_phase),
                tolerance,
                "spectral laplacian"
            );
            report.near(recovered_operator_host[index], physical_host[index], tolerance, "inverse laplacian recovery");
        }
    }

    spectral_field_type filtered_spectrum(index_space.spectral_extent());
    std::vector<complex_type> filter_host(
        index_space.complex_size(),
        complex_traits::make(scalar_type(1), scalar_type(-0.5))
    );
    copy_type()(static_cast<std::ptrdiff_t>(filter_host.size()), filter_host.data(), filtered_spectrum.data());
    two_thirds_dealiasing_2d dealiasing(index_space);
    dealiasing.template apply<Backend>(filtered_spectrum);
    copy_type()(static_cast<std::ptrdiff_t>(filter_host.size()), filtered_spectrum.data(), filter_host.data());
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        const int kx = index_space.signed_x_mode(ix);
        const int abs_kx = kx < 0 ? -kx : kx;
        for(std::size_t iy = 0; iy < index_space.my(); ++iy)
        {
            const bool retained =
                3*abs_kx < static_cast<int>(nx) && 3*static_cast<int>(iy) < static_cast<int>(ny);
            const complex_type value = filter_host[index_space.flat_index(ix, iy)];
            report.near(complex_traits::real(value), retained ? scalar_type(1) : scalar_type(0), tolerance, "two-thirds filter real part");
            report.near(complex_traits::imag(value), retained ? scalar_type(-0.5) : scalar_type(0), tolerance, "two-thirds filter imaginary part");
        }
    }

    physical_field_type product_left_physical(physical_extent);
    physical_field_type product_right_physical(physical_extent);
    physical_field_type product_physical(physical_extent);
    spectral_field_type product_left_spectrum(index_space.spectral_extent());
    spectral_field_type product_right_spectrum(index_space.spectral_extent());
    spectral_field_type product_spectrum(index_space.spectral_extent());
    std::vector<scalar_type> product_left_host(physical_host.size());
    std::vector<scalar_type> product_right_host(physical_host.size());
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            product_left_host[ix*ny + iy] =
                std::sin(scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx));
            product_right_host[ix*ny + iy] =
                std::cos(scalar_type(4)*pi*static_cast<scalar_type>(iy)/static_cast<scalar_type>(ny));
        }
    }
    copy_type()(static_cast<std::ptrdiff_t>(product_left_host.size()), product_left_host.data(), product_left_physical.data());
    copy_type()(static_cast<std::ptrdiff_t>(product_right_host.size()), product_right_host.data(), product_right_physical.data());
    transform.forward(product_left_physical, product_left_spectrum);
    transform.forward(product_right_physical, product_right_spectrum);
    operations::pseudospectral_product_2d<Backend, FFTBackend, scalar_type, two_thirds_dealiasing_2d>
        product_operation(&transform, two_thirds_dealiasing_2d(index_space));
    product_operation.apply(product_left_spectrum, product_right_spectrum, product_spectrum);
    transform.inverse(product_spectrum, product_physical);
    std::vector<scalar_type> product_host(physical_host.size());
    copy_type()(static_cast<std::ptrdiff_t>(product_host.size()), product_physical.data(), product_host.data());
    for(std::size_t index = 0; index < product_host.size(); ++index)
    {
        report.near(
            product_host[index],
            product_left_host[index]*product_right_host[index],
            tolerance,
            "pseudospectral product"
        );
    }

    vector_operations_type full_operations(index_space.full_mean_zero_state_size());
    typename vector_operations_type::vector_type full_state;
    typename vector_operations_type::vector_type full_roundtrip;
    full_operations.init_vectors(full_state, full_roundtrip);
    full_operations.start_use_vectors(full_state, full_roundtrip);
    codecs::full_real_field_2d<vector_operations_type, complex_type> full_codec(&full_operations, index_space);
    std::vector<scalar_type> full_host(full_codec.state_size());
    for(std::size_t index = 0; index < full_host.size(); ++index)
    {
        full_host[index] = scalar_type(0.01)*static_cast<scalar_type>(index + 1);
    }
    full_operations.set(full_host.data(), full_state);
    full_codec.unpack(full_state, spectrum);
    full_codec.pack(spectrum, full_roundtrip);
    std::vector<scalar_type> full_roundtrip_host(full_host.size());
    full_operations.get(full_roundtrip, full_roundtrip_host.data());
    for(std::size_t index = 0; index < full_host.size(); ++index)
    {
        report.near(full_roundtrip_host[index], full_host[index], tolerance, "full codec round trip");
    }

    std::vector<complex_type> spectrum_host(index_space.complex_size());
    copy_type()(static_cast<std::ptrdiff_t>(spectrum_host.size()), spectrum.data(), spectrum_host.data());
    report.near(complex_traits::real(spectrum_host[0]), scalar_type(0), tolerance, "full codec zero mean real part");
    report.near(complex_traits::imag(spectrum_host[0]), scalar_type(0), tolerance, "full codec zero mean imaginary part");
    for(const std::size_t iy: {std::size_t(0), ny/2})
    {
        for(std::size_t ix = 1; ix < nx/2; ++ix)
        {
            const complex_type left = spectrum_host[index_space.flat_index(ix, iy)];
            const complex_type right = spectrum_host[index_space.flat_index(nx - ix, iy)];
            report.near(complex_traits::real(right), complex_traits::real(left), tolerance, "full boundary conjugate real part");
            report.near(complex_traits::imag(right), -complex_traits::imag(left), tolerance, "full boundary conjugate imaginary part");
        }
        report.near(complex_traits::imag(spectrum_host[index_space.flat_index(0, iy)]), scalar_type(0), tolerance, "full self mode is real");
        report.near(complex_traits::imag(spectrum_host[index_space.flat_index(nx/2, iy)]), scalar_type(0), tolerance, "full Nyquist x mode is real");
    }

    vector_operations_type translation_operations(
        index_space.translation_equivariant_mean_zero_state_size()
    );
    typename vector_operations_type::vector_type translation_state;
    typename vector_operations_type::vector_type translation_roundtrip;
    translation_operations.init_vectors(translation_state, translation_roundtrip);
    translation_operations.start_use_vectors(translation_state, translation_roundtrip);
    codecs::translation_equivariant_real_field_2d<vector_operations_type, complex_type>
        translation_codec(&translation_operations, index_space);
    std::vector<scalar_type> translation_host(translation_codec.state_size());
    for(std::size_t index = 0; index < translation_host.size(); ++index)
    {
        translation_host[index] = scalar_type(0.015)*std::cos(scalar_type(index + 1));
    }
    translation_operations.set(translation_host.data(), translation_state);
    translation_codec.unpack(translation_state, spectrum);
    translation_codec.pack(spectrum, translation_roundtrip);
    std::vector<scalar_type> translation_roundtrip_host(translation_host.size());
    translation_operations.get(translation_roundtrip, translation_roundtrip_host.data());
    for(std::size_t index = 0; index < translation_host.size(); ++index)
    {
        report.near(
            translation_roundtrip_host[index],
            translation_host[index],
            tolerance,
            "translation-equivariant codec round trip"
        );
    }
    copy_type()(static_cast<std::ptrdiff_t>(spectrum_host.size()), spectrum.data(), spectrum_host.data());
    for(std::size_t iy = 0; iy < index_space.my(); ++iy)
    {
        const complex_type value = spectrum_host[index_space.flat_index(nx/2, iy)];
        report.near(complex_traits::real(value), scalar_type(0), tolerance, "x Nyquist plane removed real part");
        report.near(complex_traits::imag(value), scalar_type(0), tolerance, "x Nyquist plane removed imaginary part");
    }
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        const complex_type value = spectrum_host[index_space.flat_index(ix, ny/2)];
        report.near(complex_traits::real(value), scalar_type(0), tolerance, "y Nyquist plane removed real part");
        report.near(complex_traits::imag(value), scalar_type(0), tolerance, "y Nyquist plane removed imaginary part");
    }

    vector_operations_type odd_operations(index_space.inversion_odd_state_size());
    typename vector_operations_type::vector_type odd_state;
    typename vector_operations_type::vector_type odd_roundtrip;
    odd_operations.init_vectors(odd_state, odd_roundtrip);
    odd_operations.start_use_vectors(odd_state, odd_roundtrip);
    codecs::inversion_odd_field_2d<vector_operations_type, complex_type> odd_codec(&odd_operations, index_space);
    std::vector<scalar_type> odd_host(odd_codec.state_size());
    for(std::size_t index = 0; index < odd_host.size(); ++index)
    {
        odd_host[index] = scalar_type(0.02)*std::sin(scalar_type(index + 1));
    }
    odd_operations.set(odd_host.data(), odd_state);
    odd_codec.unpack(odd_state, spectrum);
    odd_codec.pack(spectrum, odd_roundtrip);
    std::vector<scalar_type> odd_roundtrip_host(odd_host.size());
    odd_operations.get(odd_roundtrip, odd_roundtrip_host.data());
    for(std::size_t index = 0; index < odd_host.size(); ++index)
    {
        report.near(odd_roundtrip_host[index], odd_host[index], tolerance, "odd codec round trip");
    }

    copy_type()(static_cast<std::ptrdiff_t>(spectrum_host.size()), spectrum.data(), spectrum_host.data());
    for(const complex_type& value: spectrum_host)
    {
        report.near(complex_traits::real(value), scalar_type(0), tolerance, "odd spectrum is purely imaginary");
    }
    for(const std::size_t iy: {std::size_t(0), ny/2})
    {
        report.near(complex_traits::imag(spectrum_host[index_space.flat_index(0, iy)]), scalar_type(0), tolerance, "odd self mode is zero");
        report.near(complex_traits::imag(spectrum_host[index_space.flat_index(nx/2, iy)]), scalar_type(0), tolerance, "odd Nyquist self mode is zero");
        for(std::size_t ix = 1; ix < nx/2; ++ix)
        {
            report.near(
                complex_traits::imag(spectrum_host[index_space.flat_index(nx - ix, iy)]),
                -complex_traits::imag(spectrum_host[index_space.flat_index(ix, iy)]),
                tolerance,
                "odd Hermitian boundary pair"
            );
        }
    }

    transform.inverse(spectrum, restored);
    copy_type()(static_cast<std::ptrdiff_t>(restored_host.size()), restored.data(), restored_host.data());
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            const std::size_t reflected_x = (nx - ix)%nx;
            const std::size_t reflected_y = (ny - iy)%ny;
            report.near(
                restored_host[reflected_x*ny + reflected_y],
                -restored_host[ix*ny + iy],
                tolerance,
                "odd physical inversion symmetry"
            );
        }
    }

    const std::vector<scalar_type> legacy_state = legacy_ks2d_pack(spectrum_host);
    const std::vector<complex_type> legacy_spectrum = legacy_ks2d_unpack<complex_type>(legacy_state);
    report.check(legacy_state.size() == nx*index_space.my() - 1, "legacy compatibility state retains Nx*My-1 entries");
    for(std::size_t index = 0; index < spectrum_host.size(); ++index)
    {
        report.near(complex_traits::real(legacy_spectrum[index]), complex_traits::real(spectrum_host[index]), tolerance, "legacy valid-state real compatibility");
        report.near(complex_traits::imag(legacy_spectrum[index]), complex_traits::imag(spectrum_host[index]), tolerance, "legacy valid-state imaginary compatibility");
    }

    full_operations.stop_use_vectors(full_state, full_roundtrip);
    full_operations.free_vectors(full_state, full_roundtrip);
    translation_operations.stop_use_vectors(translation_state, translation_roundtrip);
    translation_operations.free_vectors(translation_state, translation_roundtrip);
    odd_operations.stop_use_vectors(odd_state, odd_roundtrip);
    odd_operations.free_vectors(odd_state, odd_roundtrip);
    return report.finish(backend_name);
}

} // namespace tests
} // namespace fourier
} // namespace discretization

#endif
