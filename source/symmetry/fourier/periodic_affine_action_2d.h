#ifndef __SYMMETRY_FOURIER_PERIODIC_AFFINE_ACTION_2D_H__
#define __SYMMETRY_FOURIER_PERIODIC_AFFINE_ACTION_2D_H__

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_backend_ext/math.h>
#include <discretization/fourier/r2c_index_space.h>
#include <discretization/fourier/spectral_field.h>
#include <scfd/utils/device_tag.h>
#include <symmetry/fourier/periodic_affine_element_2d.h>

namespace symmetry
{
namespace fourier
{

template<class VectorOperations, class Codec, class Complex>
class periodic_affine_action_2d
{
public:
    using vector_operations_type = VectorOperations;
    using codec_type = Codec;
    using vector_type = typename vector_operations_type::vector_type;
    using scalar_type = typename vector_operations_type::scalar_type;
    using backend_type = typename vector_operations_type::backend_type;
    using ordinal_type = typename vector_operations_type::ordinal_type;
    using for_each_type = typename vector_operations_type::for_each_type;
    using complex_type = Complex;
    using complex_traits =
        ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using math_type =
        ::common::scfd_backend_ext::math<backend_type, scalar_type>;
    using spectral_field_type =
        discretization::fourier::spectral_field<backend_type, complex_type, 2>;
    using element_type = periodic_affine_element_2d;

    periodic_affine_action_2d(
        vector_operations_type* vector_operations,
        const codec_type* codec,
        const discretization::fourier::r2c_index_space_2d& index_space,
        const std::array<scalar_type, 2>& lengths):
        vector_operations_(vector_operations),
        codec_(codec),
        index_space_(index_space),
        lengths_(lengths),
        source_spectrum_(index_space_.spectral_extent()),
        destination_spectrum_(index_space_.spectral_extent())
    {
        if(vector_operations_ == nullptr || codec_ == nullptr)
        {
            throw std::invalid_argument(
                "periodic_affine_action_2d requires vector operations and a codec");
        }
        if(codec_->state_size() != vector_operations_->get_default_size())
        {
            throw std::invalid_argument(
                "periodic_affine_action_2d codec and vector sizes differ");
        }
        if(!(lengths_[0] > scalar_type(0)) ||
           !(lengths_[1] > scalar_type(0)))
        {
            throw std::invalid_argument(
                "periodic_affine_action_2d domain lengths must be positive");
        }
    }

    periodic_affine_action_2d(const periodic_affine_action_2d&) = delete;
    periodic_affine_action_2d& operator=(const periodic_affine_action_2d&) = delete;

    void apply(
        const element_type& element,
        const vector_type& source,
        vector_type& destination)
    {
        validate_element(element);
        codec_->unpack(source, source_spectrum_);

        const complex_type* input = source_spectrum_.data();
        complex_type* output = destination_spectrum_.data();
        const int nx = static_cast<int>(index_space_.nx());
        const int ny = static_cast<int>(index_space_.ny());
        const int my = static_cast<int>(index_space_.my());
        const auto matrix = element.linear().values;
        const scalar_type shift_x =
            element.translation()[0].template value<scalar_type>();
        const scalar_type shift_y =
            element.translation()[1].template value<scalar_type>();
        const std::int64_t shift_x_numerator =
            element.translation()[0].numerator();
        const std::int64_t shift_x_denominator =
            element.translation()[0].denominator();
        const std::int64_t shift_y_numerator =
            element.translation()[1].numerator();
        const std::int64_t shift_y_denominator =
            element.translation()[1].denominator();
        const bool exact_binary_shift =
            (shift_x_denominator == 1 || shift_x_denominator == 2) &&
            (shift_y_denominator == 1 || shift_y_denominator == 2);
        const scalar_type field_sign =
            static_cast<scalar_type>(element.scalar_sign());
        const scalar_type two_pi =
            scalar_type(2)*std::acos(scalar_type(-1));

        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const ordinal_type raw_index)
            {
                const int index = static_cast<int>(raw_index);
                const int ix = index/my;
                const int iy = index - ix*my;
                const int kx = ix <= nx/2 ? ix : ix - nx;
                const int ky = iy;

                int source_kx = matrix[0]*kx + matrix[2]*ky;
                int source_ky = matrix[1]*kx + matrix[3]*ky;
                bool take_conjugate = false;
                if(source_ky < 0)
                {
                    source_kx = -source_kx;
                    source_ky = -source_ky;
                    take_conjugate = true;
                }
                const int source_ix =
                    source_kx < 0 ? source_kx + nx : source_kx;
                const int source_index = source_ix*my + source_ky;
                complex_type value = input[source_index];
                if(take_conjugate)
                {
                    value = complex_traits::conj(value);
                }

                complex_type phase;
                if(exact_binary_shift)
                {
                    const std::int64_t parity_sum =
                        static_cast<std::int64_t>(kx)*
                            (shift_x_denominator == 2
                                ? shift_x_numerator
                                : 0) +
                        static_cast<std::int64_t>(ky)*
                            (shift_y_denominator == 2
                                ? shift_y_numerator
                                : 0);
                    const std::int64_t parity =
                        (parity_sum%2 + 2)%2;
                    phase = complex_traits::make(
                        parity == 0 ? field_sign : -field_sign,
                        scalar_type(0));
                }
                else
                {
                    const scalar_type angle = -two_pi*(
                        static_cast<scalar_type>(kx)*shift_x +
                        static_cast<scalar_type>(ky)*shift_y);
                    phase = complex_traits::make(
                        field_sign*math_type::cos(angle),
                        field_sign*math_type::sin(angle));
                }
                output[index] = complex_traits::mul(phase, value);
            },
            static_cast<ordinal_type>(index_space_.complex_size()));
        for_each.wait();
        codec_->pack(destination_spectrum_, destination);
    }

private:
    void validate_element(const element_type& element) const
    {
        const auto matrix = element.linear().values;
        const std::array<std::size_t, 2> sizes{{
            index_space_.nx(),
            index_space_.ny()}};
        for(std::size_t destination_axis = 0;
            destination_axis < 2;
            ++destination_axis)
        {
            for(std::size_t source_axis = 0;
                source_axis < 2;
                ++source_axis)
            {
                if(matrix[2*destination_axis + source_axis] == 0)
                {
                    continue;
                }
                if(sizes[destination_axis] != sizes[source_axis])
                {
                    throw std::invalid_argument(
                        "periodic affine axis permutation requires matching grid sizes");
                }
                const scalar_type scale =
                    lengths_[destination_axis] > lengths_[source_axis]
                        ? lengths_[destination_axis]
                        : lengths_[source_axis];
                if(math_type::abs(
                       lengths_[destination_axis] -
                       lengths_[source_axis]) >
                   scalar_type(64)*
                       std::numeric_limits<scalar_type>::epsilon()*scale)
                {
                    throw std::invalid_argument(
                        "periodic affine axis permutation requires matching domain lengths");
                }
            }
        }
    }

    vector_operations_type* vector_operations_;
    const codec_type* codec_;
    discretization::fourier::r2c_index_space_2d index_space_;
    std::array<scalar_type, 2> lengths_;
    spectral_field_type source_spectrum_;
    spectral_field_type destination_spectrum_;
};

template<class Registry, class Group, class Codec, class Complex>
auto register_periodic_affine_group_2d(
    Registry& registry,
    const Group& group,
    const Codec* codec,
    const discretization::fourier::r2c_index_space_2d& index_space,
    const std::array<typename Registry::scalar_type, 2>& lengths)
{
    using action_type = periodic_affine_action_2d<
        typename Registry::vector_operations_type,
        Codec,
        Complex>;
    auto action = std::make_shared<action_type>(
        registry.vector_operations(),
        codec,
        index_space,
        lengths);

    registry.clear();
    for(const auto& entry: group.entries())
    {
        const auto element = entry.element;
        const auto inverse =
            group.entry(entry.inverse_index).element;
        registry.add(
            entry.name,
            [action, element](const auto& source, auto& destination)
            {
                action->apply(element, source, destination);
            },
            [action, inverse](const auto& source, auto& destination)
            {
                action->apply(inverse, source, destination);
            });
    }
    registry.set_definition_fingerprint(group.fingerprint());
    return action;
}

} // namespace fourier
} // namespace symmetry

#endif
