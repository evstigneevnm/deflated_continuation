#ifndef __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_ADAPTER_H__
#define __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_ADAPTER_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <symmetry/fourier/fourier_slice_differential_1d.h>
#include <symmetry/fourier/fourier_slice_1d.h>
#include <symmetry/fourier/lsq_phase_solver_1d.h>

namespace symmetry
{
namespace fourier
{

enum class real_packed_fourier_1d_layout
{
    interleaved_real_imag
};

enum class real_packed_fourier_1d_discrete_action
{
    identity,
    negative_reflection
};

enum class real_packed_fourier_1d_stabilizer_policy
{
    single_mode,
    lsq_multimode
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
    using lsq_solver_type = lsq_phase_solver_1d<complex_type>;
    using lsq_options_type = typename lsq_solver_type::options;

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
        zero_spectrum(positive_modes_ + 1, complex_type(0)),
        action_spectrum(positive_modes_ + 1, complex_type(0)),
        candidate_spectrum(positive_modes_ + 1, complex_type(0)),
        best_spectrum(positive_modes_ + 1, complex_type(0)),
        tangent_spectrum(positive_modes_ + 1, complex_type(0)),
        vector_field_spectrum(positive_modes_ + 1, complex_type(0)),
        vector_field_derivative_spectrum(positive_modes_ + 1, complex_type(0)),
        gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        work_gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        source_gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        continuation_tangent_spectrum(positive_modes_ + 1, complex_type(0)),
        physical_pullback_spectrum(positive_modes_ + 1, complex_type(0))
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

    void enable_negative_reflection_symmetry(const bool enabled = true)
    {
        negative_reflection_symmetry_enabled = enabled;
    }

    bool negative_reflection_symmetry() const
    {
        return negative_reflection_symmetry_enabled;
    }

    void set_relative_active_mode_tolerance(const scalar_type tolerance)
    {
        if(tolerance < scalar_type(0))
        {
            throw std::invalid_argument("relative active mode tolerance must be non-negative");
        }
        relative_active_mode_tolerance = tolerance;
    }

    scalar_type get_relative_active_mode_tolerance() const
    {
        return relative_active_mode_tolerance;
    }

    void set_continuation_mode_switch_ratio(const scalar_type ratio)
    {
        if(ratio < scalar_type(0) || ratio > scalar_type(1))
        {
            throw std::invalid_argument("continuation mode switch ratio must be in [0,1]");
        }
        continuation_mode_switch_ratio = ratio;
    }

    scalar_type get_continuation_mode_switch_ratio() const
    {
        return continuation_mode_switch_ratio;
    }

    void set_tangent_continuity_weight(const scalar_type weight)
    {
        if(weight < scalar_type(0))
        {
            throw std::invalid_argument("tangent continuity weight must be non-negative");
        }
        tangent_continuity_weight = weight;
    }

    scalar_type get_tangent_continuity_weight() const
    {
        return tangent_continuity_weight;
    }

    void set_stabilizer_policy(const real_packed_fourier_1d_stabilizer_policy policy)
    {
        stabilizer_policy = policy;
    }

    real_packed_fourier_1d_stabilizer_policy get_stabilizer_policy() const
    {
        return stabilizer_policy;
    }

    void set_lsq_mode_range(const std::size_t mode_min, const std::size_t mode_max)
    {
        if(mode_min == 0)
        {
            throw std::invalid_argument("LSQ mode_min must be positive");
        }
        if(mode_max != 0 && mode_max < mode_min)
        {
            throw std::invalid_argument("LSQ mode_max must be zero or not smaller than mode_min");
        }
        lsq_options.mode_min = mode_min;
        lsq_options.mode_max = mode_max;
    }

    void set_lsq_max_active_modes(const std::size_t value)
    {
        lsq_options.max_active_modes = value;
    }

    void set_lsq_grid_points(const std::size_t value)
    {
        if(value < 8)
        {
            throw std::invalid_argument("LSQ grid_points must be at least 8");
        }
        lsq_options.grid_points = value;
    }

    void set_lsq_newton_iterations(const std::size_t value)
    {
        lsq_options.newton_iterations = value;
    }

    real_packed_fourier_1d_discrete_action last_discrete_action() const
    {
        return last_action;
    }

    void stabilize(const vector_type& source, vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        last_action = real_packed_fourier_1d_discrete_action::identity;
        last_data = choose_reliable_slice_data(spectrum, last_data.mode);
        last_data_uses_lsq = false;
        if(last_data.active())
        {
            slice.apply_shift(spectrum.data(), stabilized_spectrum.data(), stabilized_spectrum.size(), last_data.shift);
        }
        else
        {
            stabilized_spectrum = spectrum;
        }
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void stabilize_chart(const vector_type& source, vector_type& destination)
    {
        stabilize(source, destination);
    }

    void stabilize_canonical(const vector_type& source, vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        canonicalize_spectrum(spectrum, stabilized_spectrum, last_data);
        last_data_uses_lsq = false;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void stabilize_closest_to_reference(const vector_type& reference, const vector_type& source, vector_type& destination)
    {
        check_vector_size(reference);
        check_vector_size(source);
        check_vector_size(destination);

        vector_to_spectrum(source, spectrum);
        vector_to_spectrum(reference, reference_spectrum);

        const scalar_type two_pi = scalar_type(2)*static_cast<scalar_type>(std::acos(static_cast<scalar_type>(-1)));
        scalar_type best_distance = std::numeric_limits<scalar_type>::max();
        slice_data_type best_data;
        real_packed_fourier_1d_discrete_action best_action = real_packed_fourier_1d_discrete_action::identity;
        bool have_best = false;

        for(std::size_t action_index = 0; action_index < discrete_action_count(); ++action_index)
        {
            const auto action = discrete_action_at(action_index);
            apply_discrete_action_spectrum(spectrum, action_spectrum, action);
            slice_data_type candidate_data = choose_reliable_slice_data(action_spectrum, last_data.mode);

            if(!candidate_data.active())
            {
                const scalar_type distance = spectrum_distance_sq(action_spectrum, reference_spectrum);
                if(!have_best || distance < best_distance)
                {
                    best_distance = distance;
                    best_spectrum = action_spectrum;
                    best_data = candidate_data;
                    best_action = action;
                    have_best = true;
                }
                continue;
            }

            const scalar_type base_shift = candidate_data.shift;
            const std::size_t order =
                candidate_data.residual_group_order() == 0 ? std::size_t(1) : candidate_data.residual_group_order();

            for(std::size_t j = 0; j < order; ++j)
            {
                const scalar_type shift =
                    base_shift + two_pi*static_cast<scalar_type>(j)/static_cast<scalar_type>(order);
                slice.apply_shift(action_spectrum.data(), candidate_spectrum.data(), candidate_spectrum.size(), shift);
                const scalar_type distance = spectrum_distance_sq(candidate_spectrum, reference_spectrum);
                if(!have_best || distance < best_distance)
                {
                    best_distance = distance;
                    best_spectrum = candidate_spectrum;
                    best_data = candidate_data;
                    best_data.shift = shift;
                    best_data.set_shift(0, shift);
                    best_action = action;
                    have_best = true;
                }
            }
        }

        last_data = best_data;
        last_action = best_action;
        last_data_uses_lsq = false;
        stabilized_spectrum = best_spectrum;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void begin_continuation_chart(const vector_type& reference, const vector_type& reference_tangent)
    {
        check_vector_size(reference);
        check_vector_size(reference_tangent);
        vector_to_spectrum(reference, reference_spectrum);
        vector_to_spectrum(reference_tangent, continuation_tangent_spectrum);
        last_action = real_packed_fourier_1d_discrete_action::identity;
        if(stabilizer_policy == real_packed_fourier_1d_stabilizer_policy::lsq_multimode)
        {
            last_data = make_lsq_slice_data(reference_spectrum, reference_spectrum, active_mode_threshold(reference_spectrum));
            last_data_uses_lsq = last_data.active();
        }
        else
        {
            last_data = choose_continuation_slice_data(reference_spectrum, last_data.mode);
            last_data_uses_lsq = false;
        }
        if(last_data.active())
        {
            slice.apply_shift(
                reference_spectrum.data(),
                stabilized_spectrum.data(),
                stabilized_spectrum.size(),
                last_data.shift);
        }
        else
        {
            stabilized_spectrum = reference_spectrum;
        }
        continuation_chart_active = true;
    }

    void stabilize_continuation_chart(
        const vector_type& reference,
        const vector_type& reference_tangent,
        const vector_type& source,
        vector_type& destination)
    {
        check_vector_size(reference);
        check_vector_size(reference_tangent);
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        vector_to_spectrum(reference, reference_spectrum);
        vector_to_spectrum(reference_tangent, continuation_tangent_spectrum);
        continuation_chart_active = true;
        stabilize_continuation_spectra(true, destination);
    }

    void stabilize_continuation_chart(
        const vector_type& reference,
        const vector_type& source,
        vector_type& destination)
    {
        check_vector_size(reference);
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        vector_to_spectrum(reference, reference_spectrum);
        stabilize_continuation_spectra(continuation_chart_active, destination);
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
        last_action = real_packed_fourier_1d_discrete_action::identity;
        last_data = choose_reliable_slice_data(spectrum, last_data.mode);
        last_data_uses_lsq = false;
        if(last_data.active())
        {
            slice.apply_shift(spectrum.data(), stabilized_spectrum.data(), stabilized_spectrum.size(), last_data.shift);
        }
        else
        {
            stabilized_spectrum = spectrum;
        }
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

    void pullback_canonical_distance_gradient(
        const vector_type& source,
        const vector_type&,
        const vector_type& slice_gradient,
        vector_type& gradient)
    {
        check_vector_size(source);
        check_vector_size(slice_gradient);
        check_vector_size(gradient);

        vector_to_spectrum(source, spectrum);
        canonicalize_spectrum(spectrum, stabilized_spectrum, last_data);
        last_data_uses_lsq = false;
        apply_discrete_action_spectrum(spectrum, action_spectrum, last_action);
        vector_to_spectrum(slice_gradient, gradient_spectrum);
        stabilizer_adjoint_pullback_1d(
            last_data,
            stabilized_spectrum.data(),
            gradient_spectrum.data(),
            work_gradient_spectrum.data(),
            source_gradient_spectrum.data(),
            source_gradient_spectrum.size());
        apply_discrete_action_spectrum(source_gradient_spectrum, physical_pullback_spectrum, last_action);
        spectrum_to_vector(physical_pullback_spectrum, gradient);
    }

    void stabilizer_differential_from_last(const vector_type& source_tangent, vector_type& tangent_on_slice)
    {
        check_vector_size(source_tangent);
        check_vector_size(tangent_on_slice);

        vector_to_spectrum(source_tangent, gradient_spectrum);
        if(last_action != real_packed_fourier_1d_discrete_action::identity)
        {
            apply_discrete_action_spectrum(gradient_spectrum, action_spectrum, last_action);
            gradient_spectrum = action_spectrum;
        }
        if(last_data_uses_lsq &&
           last_data.active() && !last_data.active_modes.empty())
        {
            slice.apply_shift(
                gradient_spectrum.data(),
                work_gradient_spectrum.data(),
                work_gradient_spectrum.size(),
                last_data.shift);
            project_lsq_slice_tangent(
                stabilized_spectrum,
                work_gradient_spectrum,
                source_gradient_spectrum);
        }
        else
        {
            stabilizer_differential_1d(
                last_data,
                stabilized_spectrum.data(),
                gradient_spectrum.data(),
                work_gradient_spectrum.data(),
                source_gradient_spectrum.data(),
                source_gradient_spectrum.size());
        }
        spectrum_to_vector(source_gradient_spectrum, tangent_on_slice);
    }

    void project_tangent(const vector_type& state_on_slice, const vector_type& vector_on_slice, vector_type& projected)
    {
        check_vector_size(state_on_slice);
        check_vector_size(vector_on_slice);
        check_vector_size(projected);

        vector_to_spectrum(state_on_slice, spectrum);
        const slice_data_type local_data = choose_reliable_slice_data(spectrum, last_data.mode);
        vector_to_spectrum(vector_on_slice, gradient_spectrum);
        if(last_data_uses_lsq &&
           local_data.active() && !last_data.active_modes.empty())
        {
            project_lsq_slice_tangent(spectrum, gradient_spectrum, source_gradient_spectrum);
        }
        else
        {
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
            source_gradient_spectrum = projected_spectrum;
        }
        spectrum_to_vector(source_gradient_spectrum, projected);
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
        if(last_data_uses_lsq &&
           last_data.active() && !last_data.active_modes.empty())
        {
            projected_lsq_vector_field_differential(
                tangent_spectrum,
                vector_field_spectrum,
                vector_field_derivative_spectrum,
                source_gradient_spectrum);
        }
        else
        {
            projected_vector_field_differential_1d(
                last_data,
                stabilized_spectrum.data(),
                tangent_spectrum.data(),
                vector_field_spectrum.data(),
                vector_field_derivative_spectrum.data(),
                work_gradient_spectrum.data(),
                source_gradient_spectrum.data(),
                source_gradient_spectrum.size());
        }
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

    void apply_negative_reflection(const vector_type& source, vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        apply_discrete_action_spectrum(
            spectrum,
            stabilized_spectrum,
            real_packed_fourier_1d_discrete_action::negative_reflection);
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

    scalar_type spectrum_inner(
        const std::vector<complex_type>& x,
        const std::vector<complex_type>& y) const
    {
        if(x.size() != y.size())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter spectrum sizes do not match");
        }
        scalar_type result = scalar_type(0);
        for(std::size_t i = 0; i < x.size(); ++i)
        {
            result += x[i].real()*y[i].real() + x[i].imag()*y[i].imag();
        }
        return result;
    }

    scalar_type spectrum_delta_inner(
        const std::vector<complex_type>& x,
        const std::vector<complex_type>& y,
        const std::vector<complex_type>& tangent) const
    {
        if(x.size() != y.size() || x.size() != tangent.size())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter spectrum sizes do not match");
        }
        scalar_type result = scalar_type(0);
        for(std::size_t i = 0; i < x.size(); ++i)
        {
            result += (x[i].real() - y[i].real())*tangent[i].real() +
                      (x[i].imag() - y[i].imag())*tangent[i].imag();
        }
        return result;
    }

    void project_lsq_slice_tangent(
        const std::vector<complex_type>& state_on_slice,
        const std::vector<complex_type>& tangent_on_slice,
        std::vector<complex_type>& projected)
    {
        if(projected.size() != tangent_on_slice.size() || state_on_slice.size() != tangent_on_slice.size())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter LSQ projection spectrum sizes do not match");
        }

        if(!last_data.active() || last_data.active_modes.empty())
        {
            projected = tangent_on_slice;
            return;
        }

        const scalar_type matrix = lsq_solver_type::slice_matrix(
            state_on_slice,
            reference_spectrum,
            last_data.active_modes);
        if(std::abs(matrix) <= std::numeric_limits<scalar_type>::epsilon())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter LSQ projection got singular slice matrix");
        }

        const scalar_type phase = lsq_solver_type::phase_value(
            tangent_on_slice,
            reference_spectrum,
            last_data.active_modes);
        const scalar_type alpha = phase/matrix;

        for(std::size_t mode = 0; mode < tangent_on_slice.size(); ++mode)
        {
            const complex_type generator = translation_generator_1d_mode(mode, state_on_slice[mode]);
            projected[mode] = tangent_on_slice[mode] + scale_complex(generator, -alpha);
        }
    }

    void projected_lsq_vector_field_differential(
        const std::vector<complex_type>& tangent_on_slice,
        const std::vector<complex_type>& vector_field_on_slice,
        const std::vector<complex_type>& vector_field_derivative_on_slice,
        std::vector<complex_type>& projected_derivative)
    {
        project_lsq_slice_tangent(
            stabilized_spectrum,
            vector_field_derivative_on_slice,
            work_gradient_spectrum);

        if(!last_data.active() || last_data.active_modes.empty())
        {
            projected_derivative = work_gradient_spectrum;
            return;
        }

        const scalar_type matrix = lsq_solver_type::slice_matrix(
            stabilized_spectrum,
            reference_spectrum,
            last_data.active_modes);
        if(std::abs(matrix) <= std::numeric_limits<scalar_type>::epsilon())
        {
            throw std::runtime_error("real_packed_fourier_slice_1d_adapter LSQ vector field differential got singular slice matrix");
        }

        const scalar_type phase_vector_field = lsq_solver_type::phase_value(
            vector_field_on_slice,
            reference_spectrum,
            last_data.active_modes);
        const scalar_type alpha = phase_vector_field/matrix;
        const scalar_type matrix_derivative = lsq_solver_type::slice_matrix(
            tangent_on_slice,
            reference_spectrum,
            last_data.active_modes);
        const scalar_type state_generator_scale = phase_vector_field*matrix_derivative/(matrix*matrix);

        for(std::size_t mode = 0; mode < projected_derivative.size(); ++mode)
        {
            const complex_type state_generator = translation_generator_1d_mode(mode, stabilized_spectrum[mode]);
            const complex_type tangent_generator = translation_generator_1d_mode(mode, tangent_on_slice[mode]);
            complex_type value = work_gradient_spectrum[mode];
            value += scale_complex(tangent_generator, -alpha);
            value += scale_complex(state_generator, state_generator_scale);
            projected_derivative[mode] = value;
        }
    }

    scalar_type active_mode_threshold(const std::vector<complex_type>& values) const
    {
        const scalar_type norm = std::sqrt(spectrum_distance_sq(values, zero_spectrum));
        return std::max(slice.active_mode_tolerance(), relative_active_mode_tolerance*norm);
    }

    slice_data_type make_slice_data(
        const std::vector<complex_type>& values,
        const std::size_t selected_mode,
        const scalar_type tolerance) const
    {
        slice_data_type data;
        data.group_dimension = 1;
        data.tolerance = tolerance;

        if(selected_mode == 0)
        {
            return data;
        }

        const complex_type coeff = values[selected_mode];
        const scalar_type real_part = coeff.real();
        const scalar_type imag_part = coeff.imag();
        const scalar_type magnitude = std::abs(coeff);

        data.mode = selected_mode;
        data.active_rank = 1;
        data.residual_group_order_value = selected_mode;
        data.selected_abs = magnitude;
        data.selected_real_on_slice = magnitude;
        data.shift = -std::atan2(imag_part, real_part)/static_cast<scalar_type>(selected_mode);
        data.set_shift(0, data.shift);
        data.slice_matrix = static_cast<scalar_type>(selected_mode)*magnitude;
        data.active_modes = {selected_mode};
        return data;
    }

    slice_data_type make_lsq_slice_data(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const scalar_type tolerance)
    {
        slice_data_type data;
        data.group_dimension = 1;
        data.tolerance = tolerance;

        const auto result = lsq_solver.solve(source, reference, tolerance, lsq_options);
        if(!result.active)
        {
            return data;
        }

        data.active_rank = 1;
        data.active_modes = result.active_modes;
        data.mode = result.active_modes.empty() ? std::size_t(0) : result.active_modes.front();
        data.residual_group_order_value = result.residual_group_order;
        data.shift = result.shift;
        data.set_shift(0, result.shift);
        data.lsq_objective = result.objective;
        data.slice_matrix = result.slice_matrix;
        data.selected_abs = scalar_type(0);
        data.selected_real_on_slice = scalar_type(0);
        for(const auto mode: result.active_modes)
        {
            data.selected_abs = std::max<scalar_type>(data.selected_abs, std::abs(source[mode]));
        }
        return data;
    }

    slice_data_type choose_reliable_slice_data(
        const std::vector<complex_type>& values,
        const std::size_t preferred_mode) const
    {
        const scalar_type threshold = active_mode_threshold(values);

        std::size_t selected_mode = 0;
        if(preferred_mode != 0 && preferred_mode < values.size() &&
           std::abs(values[preferred_mode]) > threshold)
        {
            selected_mode = preferred_mode;
        }
        else
        {
            for(std::size_t mode = 1; mode < values.size(); ++mode)
            {
                if(std::abs(values[mode]) > threshold)
                {
                    selected_mode = mode;
                    break;
                }
            }
        }

        return make_slice_data(values, selected_mode, threshold);
    }

    slice_data_type choose_continuation_slice_data(
        const std::vector<complex_type>& values,
        const std::size_t preferred_mode) const
    {
        const scalar_type threshold = active_mode_threshold(values);
        std::size_t best_mode = 0;
        scalar_type best_matrix = scalar_type(0);
        for(std::size_t mode = 1; mode < values.size(); ++mode)
        {
            const scalar_type magnitude = std::abs(values[mode]);
            if(magnitude <= threshold)
            {
                continue;
            }
            const scalar_type slice_matrix = static_cast<scalar_type>(mode)*magnitude;
            if(best_mode == 0 || slice_matrix > best_matrix)
            {
                best_mode = mode;
                best_matrix = slice_matrix;
            }
        }

        std::size_t selected_mode = best_mode;
        if(preferred_mode != 0 && preferred_mode < values.size())
        {
            const scalar_type preferred_abs = std::abs(values[preferred_mode]);
            const scalar_type preferred_matrix = static_cast<scalar_type>(preferred_mode)*preferred_abs;
            if(preferred_abs > threshold &&
               (best_mode == 0 || preferred_matrix >= continuation_mode_switch_ratio*best_matrix))
            {
                selected_mode = preferred_mode;
            }
        }

        return make_slice_data(values, selected_mode, threshold);
    }

    scalar_type continuation_candidate_score(
        const std::vector<complex_type>& candidate,
        const bool use_tangent) const
    {
        const scalar_type distance_sq = spectrum_distance_sq(candidate, reference_spectrum);
        if(!use_tangent)
        {
            return distance_sq;
        }

        const scalar_type tangent_norm_sq = spectrum_inner(continuation_tangent_spectrum, continuation_tangent_spectrum);
        if(tangent_norm_sq <= scalar_type(0) || distance_sq <= scalar_type(0))
        {
            return distance_sq;
        }

        const scalar_type progress = spectrum_delta_inner(candidate, reference_spectrum, continuation_tangent_spectrum);
        const scalar_type alignment = std::max(
            scalar_type(-1),
            std::min(
                scalar_type(1),
                progress/std::sqrt(distance_sq*tangent_norm_sq)));
        scalar_type score = distance_sq*(scalar_type(1) + tangent_continuity_weight*(scalar_type(1) - alignment));
        if(progress < scalar_type(0))
        {
            score += tangent_backward_penalty*progress*progress/tangent_norm_sq;
        }
        return score;
    }

    void stabilize_continuation_spectra(const bool use_tangent, vector_type& destination)
    {
        const scalar_type two_pi = scalar_type(2)*static_cast<scalar_type>(std::acos(static_cast<scalar_type>(-1)));
        scalar_type best_score = std::numeric_limits<scalar_type>::max();
        slice_data_type best_data;
        real_packed_fourier_1d_discrete_action best_action = real_packed_fourier_1d_discrete_action::identity;
        bool best_uses_lsq = false;
        bool have_best = false;

        for(std::size_t action_index = 0; action_index < discrete_action_count(); ++action_index)
        {
            const auto action = discrete_action_at(action_index);
            apply_discrete_action_spectrum(spectrum, action_spectrum, action);

            slice_data_type candidate_data;
            bool candidate_uses_lsq = false;
            if(stabilizer_policy == real_packed_fourier_1d_stabilizer_policy::lsq_multimode)
            {
                const scalar_type threshold =
                    std::max(active_mode_threshold(action_spectrum), active_mode_threshold(reference_spectrum));
                candidate_data = make_lsq_slice_data(action_spectrum, reference_spectrum, threshold);
                candidate_uses_lsq = candidate_data.active();
            }
            else
            {
                candidate_data = choose_continuation_slice_data(action_spectrum, last_data.mode);
            }

            if(!candidate_data.active())
            {
                const scalar_type score = continuation_candidate_score(action_spectrum, use_tangent);
                if(!have_best || score < best_score)
                {
                    best_score = score;
                    best_spectrum = action_spectrum;
                    best_data = candidate_data;
                    best_action = action;
                    best_uses_lsq = false;
                    have_best = true;
                }
                continue;
            }

            const scalar_type base_shift = candidate_data.shift;
            const std::size_t order =
                candidate_data.residual_group_order() == 0 ? std::size_t(1) : candidate_data.residual_group_order();

            for(std::size_t j = 0; j < order; ++j)
            {
                const scalar_type shift =
                    base_shift + two_pi*static_cast<scalar_type>(j)/static_cast<scalar_type>(order);
                slice.apply_shift(action_spectrum.data(), candidate_spectrum.data(), candidate_spectrum.size(), shift);
                const scalar_type score = continuation_candidate_score(candidate_spectrum, use_tangent);
                if(!have_best || score < best_score)
                {
                    best_score = score;
                    best_spectrum = candidate_spectrum;
                    best_data = candidate_data;
                    best_data.shift = shift;
                    best_data.set_shift(0, shift);
                    best_action = action;
                    best_uses_lsq = candidate_uses_lsq;
                    have_best = true;
                }
            }
        }

        last_data = best_data;
        last_action = best_action;
        last_data_uses_lsq = best_uses_lsq;
        stabilized_spectrum = best_spectrum;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    std::size_t discrete_action_count() const
    {
        return negative_reflection_symmetry_enabled ? std::size_t(2) : std::size_t(1);
    }

    real_packed_fourier_1d_discrete_action discrete_action_at(const std::size_t index) const
    {
        if(index == 0)
        {
            return real_packed_fourier_1d_discrete_action::identity;
        }
        return real_packed_fourier_1d_discrete_action::negative_reflection;
    }

    static complex_type apply_discrete_action_value(
        const complex_type& value,
        const real_packed_fourier_1d_discrete_action action)
    {
        switch(action)
        {
            case real_packed_fourier_1d_discrete_action::identity:
                return value;
            case real_packed_fourier_1d_discrete_action::negative_reflection:
                return complex_type(-value.real(), value.imag());
        }
        throw std::runtime_error("unknown Fourier discrete action");
    }

    void apply_discrete_action_spectrum(
        const std::vector<complex_type>& source,
        std::vector<complex_type>& destination,
        const real_packed_fourier_1d_discrete_action action) const
    {
        if(source.size() != destination.size())
        {
            throw std::runtime_error("discrete action spectrum sizes do not match");
        }
        for(std::size_t mode = 0; mode < source.size(); ++mode)
        {
            destination[mode] = apply_discrete_action_value(source[mode], action);
        }
    }

    void canonicalize_spectrum(
        const std::vector<complex_type>& source,
        std::vector<complex_type>& destination,
        slice_data_type& data)
    {
        const scalar_type two_pi = scalar_type(2)*static_cast<scalar_type>(std::acos(static_cast<scalar_type>(-1)));
        const scalar_type tolerance = canonical_zero_tolerance(source);
        bool have_best = false;
        slice_data_type best_data;
        real_packed_fourier_1d_discrete_action best_action = real_packed_fourier_1d_discrete_action::identity;

        for(std::size_t action_index = 0; action_index < discrete_action_count(); ++action_index)
        {
            const auto action = discrete_action_at(action_index);
            apply_discrete_action_spectrum(source, action_spectrum, action);
            slice_data_type candidate_data = choose_reliable_slice_data(action_spectrum, 0);
            if(!candidate_data.active())
            {
                candidate_spectrum = action_spectrum;
                zero_small_components(candidate_spectrum, tolerance);
                if(!have_best || lexicographically_greater(candidate_spectrum, best_spectrum, tolerance))
                {
                    best_spectrum = candidate_spectrum;
                    best_data = candidate_data;
                    best_action = action;
                    have_best = true;
                }
                continue;
            }

            const scalar_type base_shift = candidate_data.shift;
            const std::size_t order =
                candidate_data.residual_group_order() == 0 ? std::size_t(1) : candidate_data.residual_group_order();
            for(std::size_t j = 0; j < order; ++j)
            {
                const scalar_type shift =
                    base_shift + two_pi*static_cast<scalar_type>(j)/static_cast<scalar_type>(order);
                slice.apply_shift(action_spectrum.data(), candidate_spectrum.data(), candidate_spectrum.size(), shift);
                zero_small_components(candidate_spectrum, tolerance);
                if(!have_best || lexicographically_greater(candidate_spectrum, best_spectrum, tolerance))
                {
                    best_spectrum = candidate_spectrum;
                    best_data = candidate_data;
                    best_data.shift = shift;
                    best_data.set_shift(0, shift);
                    best_action = action;
                    have_best = true;
                }
            }
        }

        destination = best_spectrum;
        data = best_data;
        last_action = best_action;
    }

    scalar_type canonical_zero_tolerance(const std::vector<complex_type>& values) const
    {
        const scalar_type norm = std::sqrt(spectrum_distance_sq(values, zero_spectrum));
        return slice.active_mode_tolerance()*std::max<scalar_type>(scalar_type(1), norm);
    }

    static void zero_small_components(std::vector<complex_type>& values, const scalar_type tolerance)
    {
        for(auto& value: values)
        {
            const scalar_type real_part = std::abs(value.real()) <= tolerance ? scalar_type(0) : value.real();
            const scalar_type imag_part = std::abs(value.imag()) <= tolerance ? scalar_type(0) : value.imag();
            value = complex_type(real_part, imag_part);
        }
    }

    static bool lexicographically_greater(
        const std::vector<complex_type>& left,
        const std::vector<complex_type>& right,
        const scalar_type tolerance)
    {
        const std::size_t n = std::min(left.size(), right.size());
        for(std::size_t mode = 1; mode < n; ++mode)
        {
            const scalar_type real_delta = left[mode].real() - right[mode].real();
            if(real_delta > tolerance)
            {
                return true;
            }
            if(real_delta < -tolerance)
            {
                return false;
            }

            const scalar_type imag_delta = left[mode].imag() - right[mode].imag();
            if(imag_delta > tolerance)
            {
                return true;
            }
            if(imag_delta < -tolerance)
            {
                return false;
            }
        }
        return false;
    }

private:
    VectorOperations* vec_ops;
    std::size_t positive_modes_;
    real_packed_fourier_1d_layout layout;
    slice_type slice;
    lsq_solver_type lsq_solver;
    lsq_options_type lsq_options;
    slice_data_type last_data;
    real_packed_fourier_1d_discrete_action last_action = real_packed_fourier_1d_discrete_action::identity;
    real_packed_fourier_1d_stabilizer_policy stabilizer_policy =
        real_packed_fourier_1d_stabilizer_policy::single_mode;
    bool negative_reflection_symmetry_enabled = false;
    bool last_data_uses_lsq = false;
    scalar_type relative_active_mode_tolerance = scalar_type(0);
    scalar_type continuation_mode_switch_ratio = scalar_type(0.25);
    scalar_type tangent_continuity_weight = scalar_type(0.25);
    scalar_type tangent_backward_penalty = scalar_type(4);
    bool continuation_chart_active = false;
    std::vector<scalar_type> host_source;
    std::vector<scalar_type> host_destination;
    std::vector<complex_type> spectrum;
    std::vector<complex_type> reference_spectrum;
    std::vector<complex_type> stabilized_spectrum;
    std::vector<complex_type> zero_spectrum;
    std::vector<complex_type> action_spectrum;
    std::vector<complex_type> candidate_spectrum;
    std::vector<complex_type> best_spectrum;
    std::vector<complex_type> tangent_spectrum;
    std::vector<complex_type> vector_field_spectrum;
    std::vector<complex_type> vector_field_derivative_spectrum;
    std::vector<complex_type> gradient_spectrum;
    std::vector<complex_type> work_gradient_spectrum;
    std::vector<complex_type> source_gradient_spectrum;
    std::vector<complex_type> continuation_tangent_spectrum;
    std::vector<complex_type> physical_pullback_spectrum;
};

} // namespace fourier
} // namespace symmetry

#endif
