#ifndef __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_ADAPTER_H__
#define __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_ADAPTER_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <symmetry/fourier/fourier_slice_differential_1d.h>
#include <symmetry/fourier/fourier_slice_1d.h>

namespace symmetry
{
namespace fourier
{

enum class real_packed_fourier_1d_layout
{
    interleaved_real_imag
};

template<class VectorOperations>
class real_packed_fourier_slice_1d_adapter
{
public:
    using vector_operations_type = VectorOperations;
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using complex_type = std::complex<scalar_type>;
    using slice_type = fourier_slice_1d<complex_type>;
    using slice_data_type = typename slice_type::slice_data;

    real_packed_fourier_slice_1d_adapter(
        VectorOperations* vec_ops_,
        const std::size_t positive_modes_,
        const real_packed_fourier_1d_layout layout_ = real_packed_fourier_1d_layout::interleaved_real_imag,
        const scalar_type active_mode_tolerance = slice_type::default_active_mode_tolerance()):
        vec_ops(vec_ops_),
        positive_modes_(positive_modes_),
        layout(layout_),
        slice(active_mode_tolerance),
        host_source(expected_vector_size(), scalar_type(0)),
        host_destination(expected_vector_size(), scalar_type(0)),
        spectrum(positive_modes_ + 1, complex_type(0)),
        reference_spectrum(positive_modes_ + 1, complex_type(0)),
        stabilized_spectrum(positive_modes_ + 1, complex_type(0)),
        candidate_spectrum(positive_modes_ + 1, complex_type(0)),
        best_spectrum(positive_modes_ + 1, complex_type(0)),
        tangent_spectrum(positive_modes_ + 1, complex_type(0)),
        vector_field_spectrum(positive_modes_ + 1, complex_type(0)),
        vector_field_derivative_spectrum(positive_modes_ + 1, complex_type(0)),
        gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        work_gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        source_gradient_spectrum(positive_modes_ + 1, complex_type(0))
    {
        if(vec_ops == nullptr)
        {
            throw std::invalid_argument("real_packed_fourier_slice_1d_adapter got null vector operations");
        }
        if(positive_modes_ == 0)
        {
            throw std::invalid_argument("real_packed_fourier_slice_1d_adapter needs at least one positive mode");
        }
    }

    std::size_t positive_modes() const
    {
        return positive_modes_;
    }

    std::size_t expected_vector_size() const
    {
        switch(layout)
        {
            case real_packed_fourier_1d_layout::interleaved_real_imag:
                return 2*positive_modes_;
        }
        throw std::runtime_error("real_packed_fourier_slice_1d_adapter has an unknown layout");
    }

    const slice_data_type& last_slice_data() const
    {
        return last_data;
    }

    void stabilize(const vector_type& source, vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        slice.stabilize(spectrum.data(), stabilized_spectrum.data(), stabilized_spectrum.size(), last_data);
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void stabilize_closest_to_reference(const vector_type& reference, const vector_type& source, vector_type& destination)
    {
        check_vector_size(reference);
        check_vector_size(source);
        check_vector_size(destination);

        vector_to_spectrum(source, spectrum);
        vector_to_spectrum(reference, reference_spectrum);

        last_data = slice.choose_slice_data(spectrum.data(), spectrum.size(), last_data.mode);
        if(!last_data.active())
        {
            stabilized_spectrum = spectrum;
            spectrum_to_vector(stabilized_spectrum, destination);
            return;
        }

        const scalar_type base_shift = last_data.shift;
        const std::size_t order = last_data.residual_group_order() == 0 ? std::size_t(1) : last_data.residual_group_order();
        const scalar_type two_pi = scalar_type(2)*static_cast<scalar_type>(std::acos(static_cast<scalar_type>(-1)));
        scalar_type best_distance = std::numeric_limits<scalar_type>::max();
        scalar_type best_shift = base_shift;

        for(std::size_t j = 0; j < order; ++j)
        {
            const scalar_type shift = base_shift + two_pi*static_cast<scalar_type>(j)/static_cast<scalar_type>(order);
            slice.apply_shift(spectrum.data(), candidate_spectrum.data(), candidate_spectrum.size(), shift);
            const scalar_type distance = spectrum_distance_sq(candidate_spectrum, reference_spectrum);
            if(j == 0 || distance < best_distance)
            {
                best_distance = distance;
                best_shift = shift;
                best_spectrum = candidate_spectrum;
            }
        }

        last_data.shift = best_shift;
        last_data.set_shift(0, best_shift);
        stabilized_spectrum = best_spectrum;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void pullback_distance_gradient(
        const vector_type& source,
        const vector_type&,
        const vector_type& slice_gradient,
        vector_type& gradient)
    {
        check_vector_size(source);
        check_vector_size(slice_gradient);
        check_vector_size(gradient);

        vector_to_spectrum(source, spectrum);
        slice.stabilize(spectrum.data(), stabilized_spectrum.data(), stabilized_spectrum.size(), last_data);
        vector_to_spectrum(slice_gradient, gradient_spectrum);
        stabilizer_adjoint_pullback_1d(
            last_data,
            stabilized_spectrum.data(),
            gradient_spectrum.data(),
            work_gradient_spectrum.data(),
            source_gradient_spectrum.data(),
            source_gradient_spectrum.size());
        spectrum_to_vector(source_gradient_spectrum, gradient);
    }

    void stabilizer_differential_from_last(const vector_type& source_tangent, vector_type& tangent_on_slice)
    {
        check_vector_size(source_tangent);
        check_vector_size(tangent_on_slice);

        vector_to_spectrum(source_tangent, gradient_spectrum);
        stabilizer_differential_1d(
            last_data,
            stabilized_spectrum.data(),
            gradient_spectrum.data(),
            work_gradient_spectrum.data(),
            source_gradient_spectrum.data(),
            source_gradient_spectrum.size());
        spectrum_to_vector(source_gradient_spectrum, tangent_on_slice);
    }

    void project_tangent(const vector_type& state_on_slice, const vector_type& vector_on_slice, vector_type& projected)
    {
        check_vector_size(state_on_slice);
        check_vector_size(vector_on_slice);
        check_vector_size(projected);

        vector_to_spectrum(state_on_slice, spectrum);
        const slice_data_type local_data = slice.choose_slice_data(spectrum.data(), spectrum.size(), last_data.mode);
        vector_to_spectrum(vector_on_slice, gradient_spectrum);
        slice.translation_generator(spectrum.data(), work_gradient_spectrum.data(), work_gradient_spectrum.size());
        typename slice_type::projection_info info;
        auto projected_spectrum = slice.project(
            local_data,
            work_gradient_spectrum,
            gradient_spectrum,
            info);
        if(!info.ok())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter::project_tangent failed");
        }
        spectrum_to_vector(projected_spectrum, projected);
    }

    void projected_vector_field_differential_from_last(
        const vector_type& tangent_on_slice,
        const vector_type& vector_field_on_slice,
        const vector_type& vector_field_derivative_on_slice,
        vector_type& projected_derivative)
    {
        check_vector_size(tangent_on_slice);
        check_vector_size(vector_field_on_slice);
        check_vector_size(vector_field_derivative_on_slice);
        check_vector_size(projected_derivative);

        vector_to_spectrum(tangent_on_slice, tangent_spectrum);
        vector_to_spectrum(vector_field_on_slice, vector_field_spectrum);
        vector_to_spectrum(vector_field_derivative_on_slice, vector_field_derivative_spectrum);
        projected_vector_field_differential_1d(
            last_data,
            stabilized_spectrum.data(),
            tangent_spectrum.data(),
            vector_field_spectrum.data(),
            vector_field_derivative_spectrum.data(),
            work_gradient_spectrum.data(),
            source_gradient_spectrum.data(),
            source_gradient_spectrum.size());
        spectrum_to_vector(source_gradient_spectrum, projected_derivative);
    }

    void apply_shift(const vector_type& source, vector_type& destination, const scalar_type shift)
    {
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        slice.apply_shift(spectrum.data(), stabilized_spectrum.data(), stabilized_spectrum.size(), shift);
        spectrum_to_vector(stabilized_spectrum, destination);
    }

private:
    void check_vector_size(const vector_type& x) const
    {
        if(vec_ops->get_size(x) != expected_vector_size())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter vector size does not match layout");
        }
    }

    void vector_to_spectrum(const vector_type& source, std::vector<complex_type>& destination)
    {
        vec_ops->get(source, host_source.data(), host_source.size());
        for(std::size_t i = 0; i < destination.size(); ++i)
        {
            destination[i] = complex_type(0);
        }
        switch(layout)
        {
            case real_packed_fourier_1d_layout::interleaved_real_imag:
                for(std::size_t mode = 1; mode <= positive_modes_; ++mode)
                {
                    const std::size_t offset = 2*(mode - 1);
                    destination[mode] = complex_type(host_source[offset], host_source[offset + 1]);
                }
                break;
        }
    }

    void spectrum_to_vector(const std::vector<complex_type>& source, vector_type& destination)
    {
        switch(layout)
        {
            case real_packed_fourier_1d_layout::interleaved_real_imag:
                for(std::size_t mode = 1; mode <= positive_modes_; ++mode)
                {
                    const std::size_t offset = 2*(mode - 1);
                    host_destination[offset] = source[mode].real();
                    host_destination[offset + 1] = source[mode].imag();
                }
                break;
        }
        vec_ops->set(host_destination.data(), destination, host_destination.size());
    }

    scalar_type spectrum_distance_sq(const std::vector<complex_type>& x, const std::vector<complex_type>& y) const
    {
        if(x.size() != y.size())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter spectrum sizes do not match");
        }
        scalar_type result = scalar_type(0);
        for(std::size_t i = 0; i < x.size(); ++i)
        {
            const scalar_type real_delta = x[i].real() - y[i].real();
            const scalar_type imag_delta = x[i].imag() - y[i].imag();
            result += real_delta*real_delta + imag_delta*imag_delta;
        }
        return result;
    }

private:
    VectorOperations* vec_ops;
    std::size_t positive_modes_;
    real_packed_fourier_1d_layout layout;
    slice_type slice;
    slice_data_type last_data;
    std::vector<scalar_type> host_source;
    std::vector<scalar_type> host_destination;
    std::vector<complex_type> spectrum;
    std::vector<complex_type> reference_spectrum;
    std::vector<complex_type> stabilized_spectrum;
    std::vector<complex_type> candidate_spectrum;
    std::vector<complex_type> best_spectrum;
    std::vector<complex_type> tangent_spectrum;
    std::vector<complex_type> vector_field_spectrum;
    std::vector<complex_type> vector_field_derivative_spectrum;
    std::vector<complex_type> gradient_spectrum;
    std::vector<complex_type> work_gradient_spectrum;
    std::vector<complex_type> source_gradient_spectrum;
};

} // namespace fourier
} // namespace symmetry

#endif
