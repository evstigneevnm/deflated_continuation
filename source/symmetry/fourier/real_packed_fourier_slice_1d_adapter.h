#ifndef __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_ADAPTER_H__
#define __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_ADAPTER_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <symmetry/fourier/fourier_slice_differential_1d.h>
#include <symmetry/fourier/fourier_slice_1d.h>
#include <symmetry/fourier/fourier_isotropy_1d.h>
#include <symmetry/fourier/fourier_spectrum_ops.h>
#include <symmetry/fourier/frozen_fourier_chart_1d.h>
#include <symmetry/fourier/lsq_fourier_slice_1d_strategy.h>
#include <symmetry/fourier/real_packed_fourier_codec_1d.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_policy.h>
#include <symmetry/continuation/continuation_chart_state.h>
#include <symmetry/continuation/isotropy_transition.h>
#include <symmetry/continuation/local_representative_policy.h>

namespace symmetry
{
namespace fourier
{

template<class VectorOperations>
class real_packed_fourier_slice_1d_adapter
{
public:
    using vector_operations_type = VectorOperations;
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using complex_type = std::complex<scalar_type>;
    using codec_type = real_packed_fourier_codec_1d<VectorOperations, complex_type>;
    using slice_type = fourier_slice_1d<complex_type>;
    using slice_data_type = typename slice_type::slice_data;
    using lsq_strategy_type = lsq_fourier_slice_1d_strategy<complex_type>;
    using lsq_solver_type = typename lsq_strategy_type::solver_type;
    using lsq_options_type = typename lsq_strategy_type::options_type;
    using policy_type = real_packed_fourier_slice_1d_policy<scalar_type>;
    using frozen_chart_type = frozen_fourier_chart_1d<complex_type>;
    using frozen_chart_evaluation_type = typename frozen_chart_type::evaluation;
    using continuation_chart_state_type =
        symmetry::continuation::continuation_chart_state<slice_data_type>;

    real_packed_fourier_slice_1d_adapter(
        VectorOperations* vec_ops_,
        const std::size_t positive_modes_,
        const real_packed_fourier_1d_layout layout_ = real_packed_fourier_1d_layout::interleaved_real_imag,
        const scalar_type active_mode_tolerance = slice_type::default_active_mode_tolerance()):
        vec_ops(vec_ops_),
        positive_modes_(positive_modes_),
        codec(vec_ops_, positive_modes_, layout_),
        slice(active_mode_tolerance),
        frozen_linearization_chart(active_mode_tolerance),
        spectrum(positive_modes_ + 1, complex_type(0)),
        reference_spectrum(positive_modes_ + 1, complex_type(0)),
        stabilized_spectrum(positive_modes_ + 1, complex_type(0)),
        zero_spectrum(positive_modes_ + 1, complex_type(0)),
        candidate_spectrum(positive_modes_ + 1, complex_type(0)),
        best_spectrum(positive_modes_ + 1, complex_type(0)),
        tangent_spectrum(positive_modes_ + 1, complex_type(0)),
        vector_field_spectrum(positive_modes_ + 1, complex_type(0)),
        vector_field_derivative_spectrum(positive_modes_ + 1, complex_type(0)),
        gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        work_gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        source_gradient_spectrum(positive_modes_ + 1, complex_type(0)),
        continuation_tangent_spectrum(positive_modes_ + 1, complex_type(0)),
        isotropy_previous_spectrum(positive_modes_ + 1, complex_type(0)),
        isotropy_candidate_spectrum(positive_modes_ + 1, complex_type(0))
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
        return codec.expected_vector_size();
    }

    const slice_data_type& last_slice_data() const
    {
        return last_data;
    }

    const continuation_chart_state_type& continuation_chart_state() const
    {
        return continuation_state;
    }

    void configure(const policy_type& policy)
    {
        policy.validate();
        if(continuation_chart_active || continuation_chart_anchor_valid)
        {
            throw std::logic_error(
                "cannot reconfigure a Fourier slice adapter after a continuation chart has started");
        }

        configuration_ = policy;
        stabilizer_policy = policy.stabilizer;
        relative_active_mode_tolerance = policy.relative_active_mode_tolerance;
        continuation_mode_switch_ratio = policy.continuation_mode_switch_ratio;
        tangent_continuity_weight = policy.tangent_continuity_weight;
        tangent_backward_penalty = policy.tangent_backward_penalty;
        lsq_strategy.configure(policy.lsq);
        local_representative_options.relative_tie_tolerance =
            policy.local_representative_relative_tolerance;

        last_data = slice_data_type();
        last_data_uses_lsq = false;
        continuation_chart_anchor_data = slice_data_type();
        continuation_chart_anchor_uses_lsq = false;
        continuation_chart_anchor_valid = false;
        continuation_state = continuation_chart_state_type();
        linearization_chart_frozen = false;
    }

    void prepare_continuation_seed(const vector_type& source, vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);

        continuation_chart_active = false;
        continuation_chart_anchor_valid = false;
        continuation_state = continuation_chart_state_type();
        linearization_chart_frozen = false;
        last_data = slice_data_type();
        last_data_uses_lsq = false;

        vector_to_spectrum(source, spectrum);
        reference_spectrum = spectrum;
        std::fill(
            continuation_tangent_spectrum.begin(),
            continuation_tangent_spectrum.end(),
            complex_type(0));
        stabilize_continuation_spectra(false, destination);

        vector_to_spectrum(destination, reference_spectrum);
        continuation_state.data = last_data;
        continuation_state.uses_lsq = last_data_uses_lsq;
        continuation_state.initialized = true;
        ++continuation_state.generation;
    }

    void accept_continuation_step(vector_type& state, vector_type& tangent)
    {
        check_vector_size(state);
        check_vector_size(tangent);
        if(!continuation_state.initialized)
        {
            throw std::logic_error(
                "cannot accept a Fourier continuation step before preparing its seed");
        }

        vector_to_spectrum(state, spectrum);
        reference_spectrum = spectrum;
        vector_to_spectrum(tangent, continuation_tangent_spectrum);
        last_data = continuation_state.data;
        last_data_uses_lsq = continuation_state.uses_lsq;
        continuation_chart_active = false;
        continuation_chart_anchor_valid = false;

        stabilize_continuation_spectra(true, state);
        continuation_stabilizer_differential_from_last(tangent, tangent);

        continuation_state.data = last_data;
        continuation_state.uses_lsq = last_data_uses_lsq;
        continuation_state.initialized = true;
        ++continuation_state.generation;
        continuation_chart_active = false;
        continuation_chart_anchor_valid = false;
        linearization_chart_frozen = false;
    }

    const policy_type& configuration() const
    {
        return configuration_;
    }

    void set_relative_active_mode_tolerance(const scalar_type tolerance)
    {
        if(tolerance < scalar_type(0))
        {
            throw std::invalid_argument("relative active mode tolerance must be non-negative");
        }
        relative_active_mode_tolerance = tolerance;
        configuration_.relative_active_mode_tolerance = tolerance;
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
        configuration_.continuation_mode_switch_ratio = ratio;
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
        configuration_.tangent_continuity_weight = weight;
    }

    scalar_type get_tangent_continuity_weight() const
    {
        return tangent_continuity_weight;
    }

    void set_tangent_backward_penalty(const scalar_type penalty)
    {
        if(penalty < scalar_type(0))
        {
            throw std::invalid_argument("tangent backward penalty must be non-negative");
        }
        tangent_backward_penalty = penalty;
        configuration_.tangent_backward_penalty = penalty;
    }

    void set_stabilizer_policy(const real_packed_fourier_1d_stabilizer_policy policy)
    {
        stabilizer_policy = policy;
        configuration_.stabilizer = policy;
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
        configuration_.lsq.mode_min = mode_min;
        configuration_.lsq.mode_max = mode_max;
        lsq_strategy.configure(configuration_.lsq);
    }

    void set_lsq_max_active_modes(const std::size_t value)
    {
        if(value == 0)
        {
            throw std::invalid_argument("LSQ max_active_modes must be positive");
        }
        configuration_.lsq.max_active_modes = value;
        lsq_strategy.configure(configuration_.lsq);
    }

    void set_lsq_grid_points(const std::size_t value)
    {
        if(value < 8)
        {
            throw std::invalid_argument("LSQ grid_points must be at least 8");
        }
        configuration_.lsq.grid_points = value;
        lsq_strategy.configure(configuration_.lsq);
    }

    void set_lsq_newton_iterations(const std::size_t value)
    {
        configuration_.lsq.newton_iterations = value;
        lsq_strategy.configure(configuration_.lsq);
    }

    void set_lsq_prefer_trivial_residual_group(const bool value)
    {
        configuration_.lsq.prefer_trivial_residual_group = value;
        lsq_strategy.configure(configuration_.lsq);
    }

    void set_lsq_minimum_coprime_relative_score(const scalar_type value)
    {
        if(value < scalar_type(0) || value > scalar_type(1))
        {
            throw std::invalid_argument("LSQ minimum coprime relative score must be in [0,1]");
        }
        configuration_.lsq.minimum_coprime_relative_score = value;
        lsq_strategy.configure(configuration_.lsq);
    }

    void set_continuation_locality_relative_tolerance(const scalar_type value)
    {
        if(value < scalar_type(0))
        {
            throw std::invalid_argument("continuation locality relative tolerance must be non-negative");
        }
        local_representative_options.relative_tie_tolerance = value;
        configuration_.local_representative_relative_tolerance = value;
    }

    void stabilize(const vector_type& source, vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
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
        slice_data_type best_data = choose_reliable_slice_data(spectrum, last_data.mode);

        if(!best_data.active())
        {
            best_spectrum = spectrum;
        }
        else
        {
            const scalar_type base_shift = best_data.shift;
            const std::size_t order =
                best_data.residual_group_order() == 0 ? std::size_t(1) : best_data.residual_group_order();

            for(std::size_t j = 0; j < order; ++j)
            {
                const scalar_type shift =
                    base_shift + two_pi*static_cast<scalar_type>(j)/static_cast<scalar_type>(order);
                slice.apply_shift(spectrum.data(), candidate_spectrum.data(), candidate_spectrum.size(), shift);
                const scalar_type distance = fourier_spectrum_distance_sq(candidate_spectrum, reference_spectrum);
                if(j == 0 || distance < best_distance)
                {
                    best_distance = distance;
                    best_spectrum = candidate_spectrum;
                    best_data.shift = shift;
                    best_data.set_shift(0, shift);
                }
            }
        }

        last_data = best_data;
        last_data_uses_lsq = false;
        stabilized_spectrum = best_spectrum;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void freeze_linearization_chart(const vector_type& source, vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);
        vector_to_spectrum(source, spectrum);
        if(stabilizer_policy == real_packed_fourier_1d_stabilizer_policy::lsq_multimode)
        {
            frozen_linearization_chart.freeze_lsq(
                spectrum,
                lsq_strategy.options(),
                active_mode_threshold(spectrum));
        }
        else
        {
            frozen_linearization_chart.freeze_single_mode(
                spectrum,
                continuation_state.initialized
                    ? continuation_state.data.mode
                    : std::size_t(0));
        }
        frozen_linearization_evaluation = frozen_linearization_chart.evaluate(spectrum);
        stabilized_spectrum = frozen_linearization_evaluation.state_on_slice;
        last_data = frozen_linearization_evaluation.data;
        last_data_uses_lsq = frozen_linearization_evaluation.uses_lsq;
        linearization_chart_frozen = true;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    bool has_frozen_linearization_chart() const
    {
        return linearization_chart_frozen;
    }

    void evaluate_frozen_linearization_chart(
        const vector_type& source,
        vector_type& destination)
    {
        check_vector_size(source);
        check_vector_size(destination);
        if(!linearization_chart_frozen)
        {
            throw std::logic_error("cannot evaluate an unfrozen Fourier linearization chart");
        }

        vector_to_spectrum(source, spectrum);
        frozen_linearization_evaluation = frozen_linearization_chart.evaluate(spectrum);
        stabilized_spectrum = frozen_linearization_evaluation.state_on_slice;
        last_data = frozen_linearization_evaluation.data;
        last_data_uses_lsq = frozen_linearization_evaluation.uses_lsq;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void begin_continuation_chart(const vector_type& reference, const vector_type& reference_tangent)
    {
        check_vector_size(reference);
        check_vector_size(reference_tangent);
        vector_to_spectrum(reference, reference_spectrum);
        vector_to_spectrum(reference_tangent, tangent_spectrum);
        if(stabilizer_policy == real_packed_fourier_1d_stabilizer_policy::lsq_multimode)
        {
            last_data = make_lsq_slice_data(reference_spectrum, reference_spectrum, active_mode_threshold(reference_spectrum));
            last_data_uses_lsq = last_data.active();
        }
        else
        {
            const std::size_t preferred_mode = continuation_state.initialized
                ? continuation_state.data.mode
                : std::size_t(0);
            last_data = choose_continuation_slice_data(reference_spectrum, preferred_mode);
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
        slice.apply_shift(
            tangent_spectrum.data(),
            continuation_tangent_spectrum.data(),
            continuation_tangent_spectrum.size(),
            last_data.active() ? last_data.shift : scalar_type(0));
        reference_spectrum = stabilized_spectrum;
        continuation_chart_active = true;
        continuation_chart_anchor_data = last_data;
        continuation_chart_anchor_uses_lsq = last_data_uses_lsq;
        continuation_chart_anchor_valid = true;
    }

    void restore_continuation_chart()
    {
        if(!continuation_chart_anchor_valid)
        {
            throw std::runtime_error("cannot restore a Fourier continuation chart before it is initialized");
        }
        last_data = continuation_chart_anchor_data;
        last_data_uses_lsq = continuation_chart_anchor_uses_lsq;
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
        if(linearization_chart_frozen)
        {
            source_gradient_spectrum = frozen_linearization_chart.stabilizer_differential(
                frozen_linearization_evaluation,
                gradient_spectrum);
            spectrum_to_vector(source_gradient_spectrum, tangent_on_slice);
            return;
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

    void continuation_stabilizer_differential_from_last(
        const vector_type& source_tangent,
        vector_type& tangent_on_slice)
    {
        check_vector_size(source_tangent);
        check_vector_size(tangent_on_slice);

        vector_to_spectrum(source_tangent, gradient_spectrum);
        if(last_data_uses_lsq && last_data.active() && !last_data.active_modes.empty())
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

        vector_to_spectrum(vector_on_slice, gradient_spectrum);
        if(linearization_chart_frozen)
        {
            source_gradient_spectrum = frozen_linearization_chart.project_tangent(
                frozen_linearization_evaluation,
                gradient_spectrum);
            spectrum_to_vector(source_gradient_spectrum, projected);
            return;
        }

        vector_to_spectrum(state_on_slice, spectrum);
        const slice_data_type local_data = choose_reliable_slice_data(spectrum, last_data.mode);
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
        if(linearization_chart_frozen)
        {
            source_gradient_spectrum =
                frozen_linearization_chart.projected_vector_field_differential(
                    frozen_linearization_evaluation,
                    tangent_spectrum,
                    vector_field_spectrum,
                    vector_field_derivative_spectrum);
            spectrum_to_vector(source_gradient_spectrum, projected_derivative);
            return;
        }
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

    symmetry::continuation::isotropy_transition_result<scalar_type>
    detect_continuation_isotropy_transition(
        const vector_type& previous,
        const vector_type& candidate,
        const symmetry::continuation::isotropy_transition_policy<scalar_type>& policy)
    {
        check_vector_size(previous);
        check_vector_size(candidate);
        policy.validate();

        symmetry::continuation::isotropy_transition_result<scalar_type> result;
        result.supported = true;
        if(!policy.enabled)
        {
            return result;
        }

        vector_to_spectrum(previous, isotropy_previous_spectrum);
        vector_to_spectrum(candidate, isotropy_candidate_spectrum);
        result.previous_order = approximate_fourier_isotropy_order_1d(
            isotropy_previous_spectrum,
            policy.relative_mode_tolerance);
        result.candidate_order = approximate_fourier_isotropy_order_1d(
            isotropy_candidate_spectrum,
            policy.relative_mode_tolerance);
        result.transition_order = result.candidate_order;
        result.previous_orbit_type =
            symmetry::translation::orbit_type::cyclic_1d(result.previous_order);
        result.candidate_orbit_type =
            symmetry::translation::orbit_type::cyclic_1d(result.candidate_order);

        if(!symmetry::continuation::is_stabilizer_increase(
               result.previous_order,
               result.candidate_order,
               policy))
        {
            return result;
        }

        result.previous_transverse_ratio = relative_fourier_transverse_norm_1d(
            isotropy_previous_spectrum,
            result.transition_order);
        result.candidate_transverse_ratio = relative_fourier_transverse_norm_1d(
            isotropy_candidate_spectrum,
            result.transition_order);
        result.detected =
            result.previous_transverse_ratio > policy.relative_mode_tolerance &&
            result.candidate_transverse_ratio <= policy.relative_mode_tolerance;
        return result;
    }

private:
    struct continuation_representative_candidate
    {
        slice_data_type data;
        scalar_type shift = scalar_type(0);
        scalar_type distance_sq = std::numeric_limits<scalar_type>::max();
        scalar_type tangent_score = std::numeric_limits<scalar_type>::max();
        bool uses_lsq = false;
    };

    void check_vector_size(const vector_type& x) const
    {
        codec.check_vector_size(x);
    }

    void vector_to_spectrum(const vector_type& source, std::vector<complex_type>& destination)
    {
        codec.unpack(source, destination);
    }

    void spectrum_to_vector(const std::vector<complex_type>& source, vector_type& destination)
    {
        codec.pack(source, destination);
    }

    void project_lsq_slice_tangent(
        const std::vector<complex_type>& state_on_slice,
        const std::vector<complex_type>& tangent_on_slice,
        std::vector<complex_type>& projected)
    {
        lsq_strategy.project_tangent(
            last_data,
            state_on_slice,
            reference_spectrum,
            tangent_on_slice,
            projected);
    }

    void projected_lsq_vector_field_differential(
        const std::vector<complex_type>& tangent_on_slice,
        const std::vector<complex_type>& vector_field_on_slice,
        const std::vector<complex_type>& vector_field_derivative_on_slice,
        std::vector<complex_type>& projected_derivative)
    {
        lsq_strategy.projected_vector_field_differential(
            last_data,
            stabilized_spectrum,
            reference_spectrum,
            tangent_on_slice,
            vector_field_on_slice,
            vector_field_derivative_on_slice,
            work_gradient_spectrum,
            projected_derivative);
    }

    scalar_type active_mode_threshold(const std::vector<complex_type>& values) const
    {
        const scalar_type norm = std::sqrt(fourier_spectrum_distance_sq(values, zero_spectrum));
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
        return lsq_strategy.make_slice_data(source, reference, tolerance);
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
        const scalar_type distance_sq = fourier_spectrum_distance_sq(candidate, reference_spectrum);
        if(!use_tangent)
        {
            return distance_sq;
        }

        const scalar_type tangent_norm_sq = fourier_spectrum_real_inner(continuation_tangent_spectrum, continuation_tangent_spectrum);
        if(tangent_norm_sq <= scalar_type(0) || distance_sq <= scalar_type(0))
        {
            return distance_sq;
        }

        const scalar_type progress = fourier_spectrum_delta_inner(candidate, reference_spectrum, continuation_tangent_spectrum);
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
        std::vector<continuation_representative_candidate> candidates;

        const auto add_candidate = [&](const slice_data_type& data,
                                       const scalar_type shift,
                                       const bool uses_lsq,
                                       const std::vector<complex_type>& values)
        {
            continuation_representative_candidate candidate;
            candidate.data = data;
            candidate.data.shift = shift;
            candidate.data.set_shift(0, shift);
            candidate.shift = shift;
            candidate.distance_sq = fourier_spectrum_distance_sq(values, reference_spectrum);
            candidate.tangent_score = continuation_candidate_score(values, use_tangent);
            candidate.uses_lsq = uses_lsq;
            candidates.push_back(candidate);
        };

        slice_data_type candidate_data;
        bool candidate_uses_lsq = false;
        if(stabilizer_policy == real_packed_fourier_1d_stabilizer_policy::lsq_multimode)
        {
            const scalar_type threshold =
                std::max(active_mode_threshold(spectrum), active_mode_threshold(reference_spectrum));
            candidate_data = make_lsq_slice_data(spectrum, reference_spectrum, threshold);
            candidate_uses_lsq = candidate_data.active();
        }
        else
        {
            const std::size_t preferred_mode = continuation_chart_active &&
                                               continuation_chart_anchor_valid
                ? continuation_chart_anchor_data.mode
                : (continuation_state.initialized
                       ? continuation_state.data.mode
                       : last_data.mode);
            if(continuation_chart_active &&
               continuation_chart_anchor_valid &&
               continuation_chart_anchor_data.active())
            {
                const scalar_type threshold = active_mode_threshold(spectrum);
                if(preferred_mode < spectrum.size() &&
                   std::abs(spectrum[preferred_mode]) > threshold)
                {
                    candidate_data = make_slice_data(
                        spectrum,
                        preferred_mode,
                        threshold);
                }
            }
            else
            {
                candidate_data = choose_continuation_slice_data(
                    spectrum,
                    preferred_mode);
            }
        }

        if(!candidate_data.active())
        {
            add_candidate(candidate_data, scalar_type(0), false, spectrum);
        }
        else
        {
            const scalar_type base_shift = candidate_data.shift;
            const std::size_t order =
                candidate_data.residual_group_order() == 0 ? std::size_t(1) : candidate_data.residual_group_order();

            for(std::size_t j = 0; j < order; ++j)
            {
                const scalar_type shift =
                    base_shift + two_pi*static_cast<scalar_type>(j)/static_cast<scalar_type>(order);
                slice.apply_shift(spectrum.data(), candidate_spectrum.data(), candidate_spectrum.size(), shift);
                add_candidate(candidate_data, shift, candidate_uses_lsq, candidate_spectrum);
            }
        }

        if(candidates.empty())
        {
            throw std::runtime_error("continuation representative selection produced no candidates");
        }

        scalar_type minimum_distance_sq = candidates.front().distance_sq;
        for(const auto& candidate: candidates)
        {
            minimum_distance_sq = std::min(minimum_distance_sq, candidate.distance_sq);
        }
        const scalar_type state_scale_sq = std::max(
            fourier_spectrum_real_inner(reference_spectrum, reference_spectrum),
            fourier_spectrum_real_inner(spectrum, spectrum));

        const continuation_representative_candidate* best = nullptr;
        for(const auto& candidate: candidates)
        {
            if(!local_representative_options.admissible(
                   candidate.distance_sq,
                   minimum_distance_sq,
                   state_scale_sq))
            {
                continue;
            }
            if(best == nullptr || candidate.tangent_score < best->tangent_score)
            {
                best = &candidate;
            }
        }
        if(best == nullptr)
        {
            throw std::runtime_error("continuation representative selection rejected every local candidate");
        }

        if(best->data.active())
        {
            slice.apply_shift(
                spectrum.data(),
                best_spectrum.data(),
                best_spectrum.size(),
                best->shift);
        }
        else
        {
            best_spectrum = spectrum;
        }

        last_data = best->data;
        last_data_uses_lsq = best->uses_lsq;
        stabilized_spectrum = best_spectrum;
        spectrum_to_vector(stabilized_spectrum, destination);
    }

    void canonicalize_spectrum(
        const std::vector<complex_type>& source,
        std::vector<complex_type>& destination,
        slice_data_type& data)
    {
        const scalar_type two_pi = scalar_type(2)*static_cast<scalar_type>(std::acos(static_cast<scalar_type>(-1)));
        const scalar_type tolerance = canonical_zero_tolerance(source);
        bool have_best = false;
        slice_data_type best_data = choose_reliable_slice_data(source, 0);
        if(!best_data.active())
        {
            best_spectrum = source;
            fourier_zero_small_components(best_spectrum, tolerance);
            have_best = true;
        }
        else
        {
            const scalar_type base_shift = best_data.shift;
            const std::size_t order =
                best_data.residual_group_order() == 0 ? std::size_t(1) : best_data.residual_group_order();
            for(std::size_t j = 0; j < order; ++j)
            {
                const scalar_type shift =
                    base_shift + two_pi*static_cast<scalar_type>(j)/static_cast<scalar_type>(order);
                slice.apply_shift(source.data(), candidate_spectrum.data(), candidate_spectrum.size(), shift);
                fourier_zero_small_components(candidate_spectrum, tolerance);
                if(!have_best || fourier_spectrum_lexicographically_greater(candidate_spectrum, best_spectrum, tolerance))
                {
                    best_spectrum = candidate_spectrum;
                    best_data.shift = shift;
                    best_data.set_shift(0, shift);
                    have_best = true;
                }
            }
        }

        destination = best_spectrum;
        data = best_data;
    }

    scalar_type canonical_zero_tolerance(const std::vector<complex_type>& values) const
    {
        const scalar_type norm = std::sqrt(fourier_spectrum_distance_sq(values, zero_spectrum));
        return slice.active_mode_tolerance()*std::max<scalar_type>(scalar_type(1), norm);
    }

private:
    VectorOperations* vec_ops;
    std::size_t positive_modes_;
    codec_type codec;
    slice_type slice;
    frozen_chart_type frozen_linearization_chart;
    policy_type configuration_;
    lsq_strategy_type lsq_strategy;
    symmetry::continuation::local_representative_policy<scalar_type> local_representative_options;
    slice_data_type last_data;
    real_packed_fourier_1d_stabilizer_policy stabilizer_policy =
        real_packed_fourier_1d_stabilizer_policy::single_mode;
    bool last_data_uses_lsq = false;
    scalar_type relative_active_mode_tolerance = scalar_type(0);
    scalar_type continuation_mode_switch_ratio = scalar_type(0.25);
    scalar_type tangent_continuity_weight = scalar_type(0.25);
    scalar_type tangent_backward_penalty = scalar_type(4);
    bool continuation_chart_active = false;
    slice_data_type continuation_chart_anchor_data;
    bool continuation_chart_anchor_uses_lsq = false;
    bool continuation_chart_anchor_valid = false;
    continuation_chart_state_type continuation_state;
    frozen_chart_evaluation_type frozen_linearization_evaluation;
    bool linearization_chart_frozen = false;
    std::vector<complex_type> spectrum;
    std::vector<complex_type> reference_spectrum;
    std::vector<complex_type> stabilized_spectrum;
    std::vector<complex_type> zero_spectrum;
    std::vector<complex_type> candidate_spectrum;
    std::vector<complex_type> best_spectrum;
    std::vector<complex_type> tangent_spectrum;
    std::vector<complex_type> vector_field_spectrum;
    std::vector<complex_type> vector_field_derivative_spectrum;
    std::vector<complex_type> gradient_spectrum;
    std::vector<complex_type> work_gradient_spectrum;
    std::vector<complex_type> source_gradient_spectrum;
    std::vector<complex_type> continuation_tangent_spectrum;
    std::vector<complex_type> isotropy_previous_spectrum;
    std::vector<complex_type> isotropy_candidate_spectrum;
};

} // namespace fourier
} // namespace symmetry

#endif
