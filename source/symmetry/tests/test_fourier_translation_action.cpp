#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#if defined(TEST_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#include <external_libraries/fft_facade_cufft.h>
#include <scfd/backend/cuda.h>
#else
#include <external_libraries/fft_facade_fftw.h>
#include <scfd/backend/omp.h>
#endif

#include <common/scfd_backend_ext/complex.h>
#include <discretization/fourier/normalized_fft.h>
#include <discretization/fourier/periodic_grid.h>
#include <discretization/fourier/r2c_index_space.h>
#include <discretization/fourier/wavevector_table.h>
#include <symmetry/fourier/active_mode_basis.h>
#include <symmetry/fourier/translation_action.h>

namespace
{

class report
{
public:
    void check(const bool condition, const std::string& message)
    {
        ++checks_;
        if(!condition)
        {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template<class T>
    void near(const T actual, const T expected, const T tolerance, const std::string& message)
    {
        check(std::abs(actual - expected) <= tolerance, message);
    }

    int finish(const std::string& backend_name) const
    {
        std::cout << backend_name << " Fourier translation: " << checks_
                  << " checks, " << failures_ << " failures\n";
        return failures_ == 0 ? 0 : 1;
    }

private:
    std::size_t checks_ = 0;
    std::size_t failures_ = 0;
};

void test_active_mode_selection(report& result)
{
    using observation = symmetry::fourier::active_mode_observation<double, 2>;
    std::vector<observation> full_rank{
        {symmetry::fourier::mode_index<2>{6, 0}, 4.0, 6},
        {symmetry::fourier::mode_index<2>{0, 2}, 3.0, 2},
        {symmetry::fourier::mode_index<2>{3, 0}, 5.0, 3},
        {symmetry::fourier::mode_index<2>{1, 1}, 1.0e-14, 1}
    };
    const auto rank_two = symmetry::fourier::select_active_mode_basis(
        full_rank,
        1.0e-12,
        1.0e-10,
        1.0e-12
    );
    result.check(rank_two.group_dimension == 2, "mode basis records group dimension");
    result.check(rank_two.active_rank == 2, "independent modes produce active rank two");
    result.check(
        rank_two.selected[0].mode[0] == 3 && rank_two.selected[0].mode[1] == 0,
        "strongest active mode is selected first"
    );
    result.check(
        rank_two.selected[1].mode[0] == 0 && rank_two.selected[1].mode[1] == 2,
        "collinear mode is skipped in favor of an independent mode"
    );

    std::vector<observation> rank_one{
        {symmetry::fourier::mode_index<2>{3, 0}, 2.0, 3},
        {symmetry::fourier::mode_index<2>{6, 0}, 1.0, 6},
        {symmetry::fourier::mode_index<2>{9, 0}, 0.5, 9}
    };
    const auto degenerate = symmetry::fourier::select_active_mode_basis(
        rank_one,
        0.0,
        0.0,
        1.0e-12
    );
    result.check(degenerate.active_rank == 1, "collinear modes expose rho less than group dimension");

    std::vector<observation> tied{
        {symmetry::fourier::mode_index<2>{1, 0}, -2.0, 4},
        {symmetry::fourier::mode_index<2>{0, 1}, 2.0, 5}
    };
    const auto deterministic = symmetry::fourier::select_active_mode_basis(
        tied,
        0.0,
        0.0,
        1.0e-12
    );
    result.check(deterministic.active_rank == 2, "absolute amplitudes participate in mode selection");
    result.check(
        deterministic.selected[0].mode[0] == 0 && deterministic.selected[0].mode[1] == 1,
        "equal amplitudes use deterministic lexicographic ordering"
    );
}

template<class Backend, class FFTBackend>
int run(const std::string& backend_name)
{
    using scalar_type = double;
    using extent_type = discretization::common::structured_extent<2>;
    using transform_type = discretization::fourier::normalized_r2c_transform<
        Backend,
        FFTBackend,
        scalar_type,
        2
    >;
    using complex_type = typename transform_type::complex_type;
    using complex_traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using physical_field_type = typename transform_type::physical_field_type;
    using spectral_field_type = typename transform_type::spectral_field_type;
    using copy_type = typename Backend::copy_type;

    constexpr std::size_t nx = 12;
    constexpr std::size_t ny = 16;
    const scalar_type pi = std::acos(scalar_type(-1));
    const scalar_type tolerance = 8.0e-11;
    const extent_type physical_extent(nx, ny);
    const discretization::fourier::r2c_index_space_2d index_space(physical_extent);
    const discretization::fourier::periodic_grid<scalar_type, 2> grid(
        physical_extent,
        {scalar_type(2)*pi, scalar_type(2)*pi}
    );
    const discretization::fourier::wavevector_table_2d<Backend, scalar_type>
        wavevectors(grid, index_space);
    transform_type transform(physical_extent);
    physical_field_type physical(physical_extent);
    physical_field_type translated_physical(physical_extent);
    physical_field_type generator_physical(physical_extent);
    spectral_field_type spectrum(index_space.spectral_extent());
    spectral_field_type translated(index_space.spectral_extent());
    spectral_field_type translated_twice(index_space.spectral_extent());
    spectral_field_type translated_once(index_space.spectral_extent());
    spectral_field_type generator(index_space.spectral_extent());
    report result;

    std::vector<scalar_type> physical_host(nx*ny);
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        const scalar_type x = scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx);
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            const scalar_type y = scalar_type(2)*pi*static_cast<scalar_type>(iy)/static_cast<scalar_type>(ny);
            physical_host[ix*ny + iy] =
                std::sin(x) + scalar_type(0.3)*std::cos(scalar_type(2)*y) +
                scalar_type(0.2)*std::sin(scalar_type(2)*x - y);
        }
    }
    copy_type()(static_cast<std::ptrdiff_t>(physical_host.size()), physical_host.data(), physical.data());
    transform.forward(physical, spectrum);

    const std::array<scalar_type, 2> shift{0.37, -0.21};
    symmetry::fourier::apply_translation<Backend>(spectrum, wavevectors, shift, translated);
    transform.inverse(translated, translated_physical);
    std::vector<scalar_type> translated_host(nx*ny);
    copy_type()(
        static_cast<std::ptrdiff_t>(translated_host.size()),
        translated_physical.data(),
        translated_host.data()
    );
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        const scalar_type x = scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx);
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            const scalar_type y = scalar_type(2)*pi*static_cast<scalar_type>(iy)/static_cast<scalar_type>(ny);
            const scalar_type expected =
                std::sin(x + shift[0]) +
                scalar_type(0.3)*std::cos(scalar_type(2)*(y + shift[1])) +
                scalar_type(0.2)*std::sin(scalar_type(2)*(x + shift[0]) - (y + shift[1]));
            result.near(
                translated_host[ix*ny + iy],
                expected,
                tolerance,
                "two-dimensional translation action"
            );
        }
    }

    const std::array<scalar_type, 2> first_shift{0.11, 0.07};
    const std::array<scalar_type, 2> second_shift{-0.29, 0.13};
    const std::array<scalar_type, 2> combined_shift{
        first_shift[0] + second_shift[0],
        first_shift[1] + second_shift[1]
    };
    symmetry::fourier::apply_translation<Backend>(spectrum, wavevectors, first_shift, translated_once);
    symmetry::fourier::apply_translation<Backend>(translated_once, wavevectors, second_shift, translated_twice);
    symmetry::fourier::apply_translation<Backend>(spectrum, wavevectors, combined_shift, translated);
    std::vector<complex_type> twice_host(index_space.complex_size());
    std::vector<complex_type> combined_host(index_space.complex_size());
    copy_type()(static_cast<std::ptrdiff_t>(twice_host.size()), translated_twice.data(), twice_host.data());
    copy_type()(static_cast<std::ptrdiff_t>(combined_host.size()), translated.data(), combined_host.data());
    for(std::size_t index = 0; index < twice_host.size(); ++index)
    {
        result.near(
            complex_traits::real(twice_host[index]),
            complex_traits::real(combined_host[index]),
            tolerance,
            "translation composition real part"
        );
        result.near(
            complex_traits::imag(twice_host[index]),
            complex_traits::imag(combined_host[index]),
            tolerance,
            "translation composition imaginary part"
        );
    }

    for(std::size_t dimension = 0; dimension < 2; ++dimension)
    {
        symmetry::fourier::translation_generator<Backend>(spectrum, wavevectors, dimension, generator);
        transform.inverse(generator, generator_physical);
        std::vector<scalar_type> generator_host(nx*ny);
        copy_type()(
            static_cast<std::ptrdiff_t>(generator_host.size()),
            generator_physical.data(),
            generator_host.data()
        );
        for(std::size_t ix = 0; ix < nx; ++ix)
        {
            const scalar_type x = scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx);
            for(std::size_t iy = 0; iy < ny; ++iy)
            {
                const scalar_type y = scalar_type(2)*pi*static_cast<scalar_type>(iy)/static_cast<scalar_type>(ny);
                const scalar_type expected = dimension == 0
                    ? std::cos(x) + scalar_type(0.4)*std::cos(scalar_type(2)*x - y)
                    : -scalar_type(0.6)*std::sin(scalar_type(2)*y) -
                        scalar_type(0.2)*std::cos(scalar_type(2)*x - y);
                result.near(
                    generator_host[ix*ny + iy],
                    expected,
                    tolerance,
                    dimension == 0 ? "x translation generator" : "y translation generator"
                );
            }
        }
    }

    test_active_mode_selection(result);
    return result.finish(backend_name);
}

} // namespace

int main()
{
    try
    {
#if defined(TEST_VECTOR_BACKEND_CUDA)
        common::init_cuda_from_scfd_selector("auto");
        return run<scfd::backend::cuda, external_libraries::fft::cufft_backend>("scfd_cuda_cufft");
#else
        return run<scfd::backend::omp, external_libraries::fft::fftw_backend>("scfd_omp_fftw");
#endif
    }
    catch(const std::exception& error)
    {
        std::cerr << "Fourier translation test failed: " << error.what() << '\n';
        return 1;
    }
}
