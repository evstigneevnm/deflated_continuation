#ifndef __DISCRETIZATION_FOURIER_CODECS_TRANSLATION_EQUIVARIANT_REAL_FIELD_H__
#define __DISCRETIZATION_FOURIER_CODECS_TRANSLATION_EQUIVARIANT_REAL_FIELD_H__

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <common/scfd_backend_ext/complex.h>
#include <discretization/fourier/codecs/codec_descriptor_2d.h>
#include <discretization/fourier/r2c_index_space.h>
#include <scfd/arrays/array.h>
#include <scfd/utils/device_tag.h>

namespace discretization
{
namespace fourier
{
namespace codecs
{

// A real mean-zero Fourier field with both Nyquist planes removed. Unlike the
// full collocation-grid codec, this space is closed under continuous translations.
template<class VectorOperations, class Complex>
class translation_equivariant_real_field_2d
{
public:
    using vector_operations_type = VectorOperations;
    using scalar_type = typename vector_operations_type::scalar_type;
    using vector_type = typename vector_operations_type::vector_type;
    using backend_type = typename vector_operations_type::backend_type;
    using memory_type = typename backend_type::memory_type;
    using complex_type = Complex;
    using complex_traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using descriptor_type = detail::coefficient_orbit_2d;
    using descriptor_array_type = scfd::arrays::array<descriptor_type, memory_type>;
    using for_each_type = typename vector_operations_type::for_each_type;
    using copy_type = typename vector_operations_type::copy_type;
    using ordinal_type = typename vector_operations_type::ordinal_type;

    translation_equivariant_real_field_2d(
        vector_operations_type* vector_operations,
        const r2c_index_space_2d& index_space
    ):
        vector_operations_(vector_operations),
        index_space_(index_space)
    {
        static_assert(std::is_same<typename complex_traits::real_type, scalar_type>::value,
            "translation-equivariant codec scalar and complex value types must match");
        if(vector_operations_ == nullptr)
        {
            throw std::invalid_argument("translation_equivariant_real_field_2d requires vector operations");
        }
        if(vector_operations_->get_default_size() != state_size())
        {
            throw std::invalid_argument(
                "translation_equivariant_real_field_2d vector size does not match its Fourier space"
            );
        }
        initialize_descriptors();
    }

    std::size_t state_size() const
    {
        return index_space_.translation_equivariant_mean_zero_state_size();
    }

    std::size_t complex_size() const
    {
        return index_space_.complex_size();
    }

    template<class SpectralField>
    void unpack(const vector_type& state, SpectralField& spectrum) const
    {
        require_sizes(state, spectrum);
        complex_type* output = spectrum.data();
        const scalar_type* input = state.raw_ptr();
        const descriptor_type* descriptors = descriptors_.raw_ptr();
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const ordinal_type index)
            {
                output[index] = complex_traits::make(scalar_type(0), scalar_type(0));
            },
            static_cast<ordinal_type>(complex_size())
        );
        for_each.wait();
        for_each(
            [=] __DEVICE_TAG__ (const ordinal_type descriptor_index)
            {
                const descriptor_type descriptor = descriptors[descriptor_index];
                const complex_type value = complex_traits::make(
                    input[descriptor.state_offset],
                    input[descriptor.state_offset + 1]
                );
                output[descriptor.spectrum_index] = value;
                if(descriptor.partner_index >= 0)
                {
                    output[descriptor.partner_index] = complex_traits::conj(value);
                }
            },
            static_cast<ordinal_type>(descriptor_count_)
        );
        for_each.wait();
    }

    template<class SpectralField>
    void pack(const SpectralField& spectrum, vector_type& state) const
    {
        require_sizes(state, spectrum);
        const complex_type* input = spectrum.data();
        scalar_type* output = state.raw_ptr();
        const descriptor_type* descriptors = descriptors_.raw_ptr();
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const ordinal_type descriptor_index)
            {
                const descriptor_type descriptor = descriptors[descriptor_index];
                const complex_type value = input[descriptor.spectrum_index];
                output[descriptor.state_offset] = complex_traits::real(value);
                output[descriptor.state_offset + 1] = complex_traits::imag(value);
            },
            static_cast<ordinal_type>(descriptor_count_)
        );
        for_each.wait();
    }

    template<class SpectralField>
    void pack_adjoint(const vector_type& state, SpectralField& spectrum) const
    {
        require_sizes(state, spectrum);
        complex_type* output = spectrum.data();
        const scalar_type* input = state.raw_ptr();
        const descriptor_type* descriptors = descriptors_.raw_ptr();
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const ordinal_type index)
            {
                output[index] = complex_traits::make(scalar_type(0), scalar_type(0));
            },
            static_cast<ordinal_type>(complex_size()));
        for_each.wait();
        for_each(
            [=] __DEVICE_TAG__ (const ordinal_type descriptor_index)
            {
                const descriptor_type descriptor = descriptors[descriptor_index];
                const scalar_type weight =
                    descriptor.partner_index >= 0 ? scalar_type(0.5) : scalar_type(1);
                const complex_type value = complex_traits::make(
                    weight*input[descriptor.state_offset],
                    weight*input[descriptor.state_offset + 1]);
                output[descriptor.spectrum_index] = value;
                if(descriptor.partner_index >= 0)
                {
                    output[descriptor.partner_index] = complex_traits::conj(value);
                }
            },
            static_cast<ordinal_type>(descriptor_count_));
        for_each.wait();
    }

    template<class SpectralField>
    void unpack_adjoint(const SpectralField& spectrum, vector_type& state) const
    {
        require_sizes(state, spectrum);
        const complex_type* input = spectrum.data();
        scalar_type* output = state.raw_ptr();
        const descriptor_type* descriptors = descriptors_.raw_ptr();
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const ordinal_type descriptor_index)
            {
                const descriptor_type descriptor = descriptors[descriptor_index];
                const complex_type value = input[descriptor.spectrum_index];
                scalar_type real = complex_traits::real(value);
                scalar_type imag = complex_traits::imag(value);
                if(descriptor.partner_index >= 0)
                {
                    const complex_type partner = input[descriptor.partner_index];
                    real += complex_traits::real(partner);
                    imag -= complex_traits::imag(partner);
                }
                output[descriptor.state_offset] = real;
                output[descriptor.state_offset + 1] = imag;
            },
            static_cast<ordinal_type>(descriptor_count_));
        for_each.wait();
    }

private:
    void initialize_descriptors()
    {
        std::vector<descriptor_type> host;
        std::ptrdiff_t offset = 0;
        for(std::size_t ix = 1; ix < index_space_.nx()/2; ++ix)
        {
            host.push_back(descriptor_type{
                static_cast<std::ptrdiff_t>(index_space_.flat_index(ix, 0)),
                static_cast<std::ptrdiff_t>(index_space_.flat_index(index_space_.conjugate_x(ix), 0)),
                offset,
                detail::coefficient_kind_2d::complex
            });
            offset += 2;
        }
        for(std::size_t iy = 1; iy < index_space_.ny()/2; ++iy)
        {
            for(std::size_t ix = 0; ix < index_space_.nx(); ++ix)
            {
                if(ix == index_space_.nx()/2)
                {
                    continue;
                }
                host.push_back(descriptor_type{
                    static_cast<std::ptrdiff_t>(index_space_.flat_index(ix, iy)),
                    -1,
                    offset,
                    detail::coefficient_kind_2d::complex
                });
                offset += 2;
            }
        }
        if(static_cast<std::size_t>(offset) != state_size())
        {
            throw std::logic_error("translation-equivariant descriptor count is inconsistent");
        }
        descriptor_count_ = host.size();
        descriptors_.init(static_cast<ordinal_type>(descriptor_count_));
        copy_type()(static_cast<ordinal_type>(descriptor_count_), host.data(), descriptors_.raw_ptr());
    }

    template<class SpectralField>
    void require_sizes(const vector_type& state, const SpectralField& spectrum) const
    {
        if(vector_operations_->get_size(state) != state_size() || spectrum.size() != complex_size())
        {
            throw std::invalid_argument("translation_equivariant_real_field_2d pack/unpack size mismatch");
        }
    }

    vector_operations_type* vector_operations_;
    r2c_index_space_2d index_space_;
    descriptor_array_type descriptors_;
    std::size_t descriptor_count_ = 0;
};

} // namespace codecs
} // namespace fourier
} // namespace discretization

#endif
