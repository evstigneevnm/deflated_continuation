#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <symmetry/fourier/fourier_slice_1d.h>
#include <symmetry/fourier/fourier_slice_differential_1d.h>

namespace
{

using real = double;
using complex_type = std::complex<real>;
using slice_type = symmetry::fourier::fourier_slice_1d<complex_type>;

int checks = 0;
int failures = 0;

void require_close(const char* label, const real value, const real expected, const real tolerance)
{
    ++checks;
    const real error = std::abs(value - expected);
    if(error > tolerance)
    {
        ++failures;
        std::cerr << "FAIL " << label
                  << " value=" << value
                  << " expected=" << expected
                  << " error=" << error
                  << " tolerance=" << tolerance << std::endl;
    }
}

std::vector<complex_type> shifted(
    const slice_type& slice,
    const std::vector<complex_type>& source,
    const real shift)
{
    std::vector<complex_type> destination(source.size());
    slice.apply_shift(source.data(), destination.data(), source.size(), shift);
    return destination;
}

std::vector<complex_type> add_scaled(
    const std::vector<complex_type>& x,
    const std::vector<complex_type>& v,
    const real scale)
{
    std::vector<complex_type> y(x.size());
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        y[i] = x[i] + scale*v[i];
    }
    return y;
}

real objective(
    const slice_type& slice,
    const std::vector<complex_type>& source,
    const std::vector<complex_type>& reference)
{
    auto data = slice.choose_slice_data(source.data(), source.size());
    const auto stabilized = slice.stabilize(source, data);
    real value = real(0);
    for(std::size_t i = 0; i < stabilized.size(); ++i)
    {
        const complex_type diff = stabilized[i] - reference[i];
        value += diff.real()*diff.real() + diff.imag()*diff.imag();
    }
    return real(0.5)*value;
}

std::vector<complex_type> projected_vector_field(
    const slice_type& slice,
    const std::vector<complex_type>& state_on_slice,
    const std::vector<complex_type>& field_on_slice)
{
    auto data = slice.choose_slice_data(state_on_slice.data(), state_on_slice.size());
    const auto generator = slice.translation_generator(state_on_slice);
    typename slice_type::projection_info info;
    auto projected = slice.project(data, generator, field_on_slice, info);
    if(!info.ok())
    {
        throw std::runtime_error("projected_vector_field test helper failed");
    }
    return projected;
}

real vector_error(
    const std::vector<complex_type>& left,
    const std::vector<complex_type>& right)
{
    real value = real(0);
    for(std::size_t i = 0; i < left.size(); ++i)
    {
        const complex_type diff = left[i] - right[i];
        value += diff.real()*diff.real() + diff.imag()*diff.imag();
    }
    return std::sqrt(value);
}

} // namespace

int main()
{
    slice_type slice;
    std::vector<complex_type> source{
        {0.0, 0.0},
        {1.2, 0.4},
        {-0.3, 0.8},
        {0.5, -0.2}
    };
    auto data = slice.choose_slice_data(source.data(), source.size());
    const auto state_on_slice = slice.stabilize(source, data);
    const std::vector<complex_type> source_tangent{
        {0.0, 0.0},
        {-0.2, 0.7},
        {0.4, -0.1},
        {-0.3, 0.2}
    };
    const std::vector<complex_type> gradient_on_slice{
        {0.0, 0.0},
        {0.31, -0.17},
        {-0.44, 0.25},
        {0.19, 0.33}
    };

    std::vector<complex_type> work(source.size());
    std::vector<complex_type> stabilized_tangent(source.size());
    symmetry::fourier::stabilizer_differential_1d(
        data,
        state_on_slice.data(),
        source_tangent.data(),
        work.data(),
        stabilized_tangent.data(),
        source.size());

    require_close(
        "projected tangent satisfies slice phase condition",
        stabilized_tangent[data.mode].imag(),
        0.0,
        1e-13);

    std::vector<complex_type> work_gradient(source.size());
    std::vector<complex_type> pulled_gradient(source.size());
    symmetry::fourier::stabilizer_adjoint_pullback_1d(
        data,
        state_on_slice.data(),
        gradient_on_slice.data(),
        work_gradient.data(),
        pulled_gradient.data(),
        source.size());

    const real left = symmetry::fourier::real_inner_product_packed_positive_modes(
        gradient_on_slice.data(), stabilized_tangent.data(), source.size());
    const real right = symmetry::fourier::real_inner_product_packed_positive_modes(
        pulled_gradient.data(), source_tangent.data(), source.size());
    require_close("stabilizer adjoint identity", left, right, 1e-13);

    std::vector<complex_type> reference = state_on_slice;
    reference[1] += complex_type(0.03, 0.0);
    reference[2] += complex_type(-0.07, 0.11);
    reference[3] += complex_type(0.05, -0.02);

    std::vector<complex_type> distance_gradient(state_on_slice.size());
    for(std::size_t i = 0; i < state_on_slice.size(); ++i)
    {
        distance_gradient[i] = state_on_slice[i] - reference[i];
    }
    symmetry::fourier::stabilizer_adjoint_pullback_1d(
        data,
        state_on_slice.data(),
        distance_gradient.data(),
        work_gradient.data(),
        pulled_gradient.data(),
        source.size());

    const real eps = 1e-6;
    const real finite_difference = (
        objective(slice, add_scaled(source, source_tangent, eps), reference) -
        objective(slice, add_scaled(source, source_tangent, -eps), reference))/(real(2)*eps);
    const real predicted = symmetry::fourier::real_inner_product_packed_positive_modes(
        pulled_gradient.data(), source_tangent.data(), source.size());
    require_close("stabilized distance finite-difference gradient", predicted, finite_difference, 5e-10);

    const auto group_tangent = slice.translation_generator(state_on_slice);
    symmetry::fourier::project_slice_tangent_1d(
        data,
        state_on_slice.data(),
        group_tangent.data(),
        stabilized_tangent.data(),
        state_on_slice.size());
    require_close(
        "slice projector removes group tangent",
        symmetry::fourier::real_inner_product_packed_positive_modes(
            stabilized_tangent.data(), stabilized_tangent.data(), stabilized_tangent.size()),
        0.0,
        1e-24);

    std::vector<complex_type> tangent_on_slice(source.size());
    symmetry::fourier::project_slice_tangent_1d(
        data,
        state_on_slice.data(),
        source_tangent.data(),
        tangent_on_slice.data(),
        source.size());
    const std::vector<complex_type> field_on_slice{
        {0.0, 0.0},
        {0.12, 0.34},
        {-0.23, 0.41},
        {0.29, -0.18}
    };
    const std::vector<complex_type> zero_field_derivative(source.size(), complex_type(0.0, 0.0));
    std::vector<complex_type> projected_field_derivative(source.size());
    symmetry::fourier::projected_vector_field_differential_1d(
        data,
        state_on_slice.data(),
        tangent_on_slice.data(),
        field_on_slice.data(),
        zero_field_derivative.data(),
        work.data(),
        projected_field_derivative.data(),
        source.size());
    const auto projected_plus = projected_vector_field(
        slice,
        add_scaled(state_on_slice, tangent_on_slice, eps),
        field_on_slice);
    const auto projected_minus = projected_vector_field(
        slice,
        add_scaled(state_on_slice, tangent_on_slice, -eps),
        field_on_slice);
    std::vector<complex_type> projected_finite_difference(source.size());
    for(std::size_t i = 0; i < source.size(); ++i)
    {
        projected_finite_difference[i] = (projected_plus[i] - projected_minus[i])/(real(2)*eps);
    }
    require_close(
        "projected vector field differential finite difference",
        vector_error(projected_field_derivative, projected_finite_difference),
        0.0,
        1e-9);

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures == 0)
    {
        std::cout << "PASSED" << std::endl;
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
