#ifndef __SYMMETRY_FOURIER_FOURIER_SLICE_DIFFERENTIAL_1D_H__
#define __SYMMETRY_FOURIER_FOURIER_SLICE_DIFFERENTIAL_1D_H__

#include <cstddef>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <symmetry/fourier/translation_generators.h>

namespace symmetry
{
namespace fourier
{

template<class Complex>
typename common::scfd_backend_ext::complex_value_traits<Complex>::real_type
real_inner_product_packed_positive_modes(
    const Complex* left,
    const Complex* right,
    const std::size_t size)
{
    if(left == nullptr || right == nullptr)
    {
        throw std::invalid_argument("real_inner_product_packed_positive_modes got null pointer");
    }

    using traits_type = common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits_type::real_type;

    real_type result = real_type(0);
    for(std::size_t i = 0; i < size; ++i)
    {
        result += traits_type::real(left[i])*traits_type::real(right[i]) +
                  traits_type::imag(left[i])*traits_type::imag(right[i]);
    }
    return result;
}

template<class SliceData, class Complex>
void project_slice_tangent_1d(
    const SliceData& data,
    const Complex* state_on_slice,
    const Complex* tangent_on_slice,
    Complex* projected,
    const std::size_t size)
{
    if(state_on_slice == nullptr || tangent_on_slice == nullptr || projected == nullptr)
    {
        throw std::invalid_argument("project_slice_tangent_1d got null pointer");
    }

    using traits_type = common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits_type::real_type;

    if(!data.active())
    {
        for(std::size_t i = 0; i < size; ++i)
        {
            projected[i] = tangent_on_slice[i];
        }
        return;
    }
    if(data.mode >= size)
    {
        throw std::out_of_range("project_slice_tangent_1d mode exceeds vector size");
    }
    if(data.slice_matrix == real_type(0))
    {
        throw std::runtime_error("project_slice_tangent_1d got singular slice matrix");
    }

    const real_type alpha = traits_type::imag(tangent_on_slice[data.mode])/data.slice_matrix;
    for(std::size_t mode = 0; mode < size; ++mode)
    {
        const Complex generator = translation_generator_1d_mode(mode, state_on_slice[mode]);
        const Complex correction = scale_complex(generator, -alpha);
        projected[mode] = traits_type::add(tangent_on_slice[mode], correction);
    }
}

template<class SliceData, class Complex>
void pullback_slice_tangent_projector_1d(
    const SliceData& data,
    const Complex* state_on_slice,
    const Complex* gradient_on_slice,
    Complex* pulled_back_on_slice,
    const std::size_t size)
{
    if(state_on_slice == nullptr || gradient_on_slice == nullptr || pulled_back_on_slice == nullptr)
    {
        throw std::invalid_argument("pullback_slice_tangent_projector_1d got null pointer");
    }

    using traits_type = common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits_type::real_type;

    if(!data.active())
    {
        for(std::size_t i = 0; i < size; ++i)
        {
            pulled_back_on_slice[i] = gradient_on_slice[i];
        }
        return;
    }
    if(data.mode >= size)
    {
        throw std::out_of_range("pullback_slice_tangent_projector_1d mode exceeds vector size");
    }
    if(data.slice_matrix == real_type(0))
    {
        throw std::runtime_error("pullback_slice_tangent_projector_1d got singular slice matrix");
    }

    real_type generator_inner_product = real_type(0);
    for(std::size_t mode = 0; mode < size; ++mode)
    {
        const Complex generator = translation_generator_1d_mode(mode, state_on_slice[mode]);
        generator_inner_product += traits_type::real(gradient_on_slice[mode])*traits_type::real(generator) +
                                   traits_type::imag(gradient_on_slice[mode])*traits_type::imag(generator);
    }

    const real_type phase_gradient_scale = generator_inner_product/data.slice_matrix;
    for(std::size_t mode = 0; mode < size; ++mode)
    {
        pulled_back_on_slice[mode] = gradient_on_slice[mode];
    }
    pulled_back_on_slice[data.mode] = traits_type::make(
        traits_type::real(pulled_back_on_slice[data.mode]),
        traits_type::imag(pulled_back_on_slice[data.mode]) - phase_gradient_scale);
}

template<class SliceData, class Complex>
void stabilizer_differential_1d(
    const SliceData& data,
    const Complex* state_on_slice,
    const Complex* source_tangent,
    Complex* work_tangent_on_slice,
    Complex* stabilized_tangent,
    const std::size_t size)
{
    if(source_tangent == nullptr || work_tangent_on_slice == nullptr)
    {
        throw std::invalid_argument("stabilizer_differential_1d got null tangent pointer");
    }
    apply_shift_1d(source_tangent, work_tangent_on_slice, size, data.shift);
    project_slice_tangent_1d(data, state_on_slice, work_tangent_on_slice, stabilized_tangent, size);
}

template<class SliceData, class Complex>
void stabilizer_adjoint_pullback_1d(
    const SliceData& data,
    const Complex* state_on_slice,
    const Complex* gradient_on_slice,
    Complex* work_gradient_on_slice,
    Complex* pulled_back_to_source,
    const std::size_t size)
{
    if(gradient_on_slice == nullptr || work_gradient_on_slice == nullptr || pulled_back_to_source == nullptr)
    {
        throw std::invalid_argument("stabilizer_adjoint_pullback_1d got null gradient pointer");
    }
    pullback_slice_tangent_projector_1d(data, state_on_slice, gradient_on_slice, work_gradient_on_slice, size);
    apply_shift_1d(work_gradient_on_slice, pulled_back_to_source, size, -data.shift);
}

template<class SliceData, class Complex>
void projected_vector_field_differential_1d(
    const SliceData& data,
    const Complex* state_on_slice,
    const Complex* tangent_on_slice,
    const Complex* vector_field_on_slice,
    const Complex* vector_field_derivative_on_slice,
    Complex* work_projected_derivative,
    Complex* projected_derivative,
    const std::size_t size)
{
    if(state_on_slice == nullptr || tangent_on_slice == nullptr ||
       vector_field_on_slice == nullptr || vector_field_derivative_on_slice == nullptr ||
       work_projected_derivative == nullptr || projected_derivative == nullptr)
    {
        throw std::invalid_argument("projected_vector_field_differential_1d got null pointer");
    }

    project_slice_tangent_1d(
        data,
        state_on_slice,
        vector_field_derivative_on_slice,
        work_projected_derivative,
        size);

    using traits_type = common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits_type::real_type;

    if(!data.active())
    {
        for(std::size_t i = 0; i < size; ++i)
        {
            projected_derivative[i] = work_projected_derivative[i];
        }
        return;
    }
    if(data.mode >= size)
    {
        throw std::out_of_range("projected_vector_field_differential_1d mode exceeds vector size");
    }
    if(data.slice_matrix == real_type(0))
    {
        throw std::runtime_error("projected_vector_field_differential_1d got singular slice matrix");
    }

    const real_type phase_vector_field = traits_type::imag(vector_field_on_slice[data.mode]);
    const real_type alpha = phase_vector_field/data.slice_matrix;
    const Complex tangent_generator_at_slice_mode =
        translation_generator_1d_mode(data.mode, tangent_on_slice[data.mode]);
    const real_type slice_matrix_derivative = traits_type::imag(tangent_generator_at_slice_mode);
    const real_type state_generator_scale = alpha*slice_matrix_derivative/data.slice_matrix;

    for(std::size_t mode = 0; mode < size; ++mode)
    {
        const Complex state_generator = translation_generator_1d_mode(mode, state_on_slice[mode]);
        const Complex tangent_generator = translation_generator_1d_mode(mode, tangent_on_slice[mode]);
        Complex value = work_projected_derivative[mode];
        value = traits_type::add(value, scale_complex(tangent_generator, -alpha));
        value = traits_type::add(value, scale_complex(state_generator, state_generator_scale));
        projected_derivative[mode] = value;
    }
}

} // namespace fourier
} // namespace symmetry

#endif
