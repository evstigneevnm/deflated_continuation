#ifndef __SYMMETRY_FOURIER_FROZEN_FOURIER_CHART_1D_H__
#define __SYMMETRY_FOURIER_FROZEN_FOURIER_CHART_1D_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <symmetry/fourier/fourier_slice_1d.h>
#include <symmetry/fourier/fourier_slice_differential_1d.h>
#include <symmetry/fourier/lsq_phase_solver_1d.h>

namespace symmetry
{
namespace fourier
{

enum class frozen_fourier_chart_1d_kind
{
    single_mode,
    lsq_multimode
};

template<class Complex>
class frozen_fourier_chart_1d
{
public:
    using complex_type = Complex;
    using real_type = typename complex_type::value_type;
    using slice_type = fourier_slice_1d<complex_type>;
    using slice_data_type = typename slice_type::slice_data;
    using lsq_solver_type = lsq_phase_solver_1d<complex_type>;
    using lsq_options_type = typename lsq_solver_type::options;

    struct evaluation
    {
        slice_data_type data;
        std::vector<complex_type> state_on_slice;
        bool uses_lsq = false;
    };

    explicit frozen_fourier_chart_1d(
        const real_type active_mode_tolerance = slice_type::default_active_mode_tolerance()):
        slice_(active_mode_tolerance),
        active_mode_tolerance_(active_mode_tolerance)
    {
    }

    void freeze_single_mode(
        const std::vector<complex_type>& reference,
        const std::size_t preferred_mode = 0)
    {
        check_reference(reference);
        kind_ = frozen_fourier_chart_1d_kind::single_mode;
        const auto data = slice_.choose_slice_data(
            reference.data(), reference.size(), preferred_mode);
        active_modes_.clear();
        if(data.active())
        {
            active_modes_.push_back(data.mode);
        }
        reference_on_slice_ = slice_.apply_shift(reference, data.shift);
        residual_group_order_ = data.residual_group_order();
        frozen_ = true;
    }

    void freeze_lsq(
        const std::vector<complex_type>& reference,
        const lsq_options_type& options,
        const real_type active_tolerance)
    {
        check_reference(reference);
        kind_ = frozen_fourier_chart_1d_kind::lsq_multimode;
        lsq_options_ = options;
        const auto result = lsq_solver_.solve(reference, reference, active_tolerance, options);
        active_modes_ = result.active_modes;
        residual_group_order_ = result.active ? result.residual_group_order : std::size_t(1);
        reference_on_slice_ = result.active
            ? slice_.apply_shift(reference, result.shift)
            : reference;
        frozen_ = true;
    }

    bool frozen() const { return frozen_; }
    frozen_fourier_chart_1d_kind kind() const { return kind_; }
    const std::vector<std::size_t>& active_modes() const { return active_modes_; }
    const std::vector<complex_type>& reference_on_slice() const { return reference_on_slice_; }
    std::size_t residual_group_order() const { return residual_group_order_; }

    evaluation evaluate(const std::vector<complex_type>& source) const
    {
        check_source(source);
        if(kind_ == frozen_fourier_chart_1d_kind::lsq_multimode)
        {
            return evaluate_lsq(source);
        }
        return evaluate_single_mode(source);
    }

    std::vector<complex_type> stabilizer_differential(
        const evaluation& at,
        const std::vector<complex_type>& source_tangent) const
    {
        check_evaluation(at);
        check_source(source_tangent);
        std::vector<complex_type> shifted_tangent(source_tangent.size());
        std::vector<complex_type> result(source_tangent.size());
        apply_shift_1d(
            source_tangent.data(), shifted_tangent.data(), source_tangent.size(), at.data.shift);
        project_tangent_on_slice(at, shifted_tangent, result);
        return result;
    }

    std::vector<complex_type> project_tangent(
        const evaluation& at,
        const std::vector<complex_type>& tangent_on_slice) const
    {
        check_evaluation(at);
        check_source(tangent_on_slice);
        std::vector<complex_type> result(tangent_on_slice.size());
        project_tangent_on_slice(at, tangent_on_slice, result);
        return result;
    }

    std::vector<complex_type> projected_vector_field_differential(
        const evaluation& at,
        const std::vector<complex_type>& tangent_on_slice,
        const std::vector<complex_type>& vector_field_on_slice,
        const std::vector<complex_type>& vector_field_derivative_on_slice) const
    {
        check_evaluation(at);
        check_source(tangent_on_slice);
        check_source(vector_field_on_slice);
        check_source(vector_field_derivative_on_slice);

        if(!at.uses_lsq)
        {
            std::vector<complex_type> work(at.state_on_slice.size());
            std::vector<complex_type> result(at.state_on_slice.size());
            projected_vector_field_differential_1d(
                at.data,
                at.state_on_slice.data(),
                tangent_on_slice.data(),
                vector_field_on_slice.data(),
                vector_field_derivative_on_slice.data(),
                work.data(),
                result.data(),
                result.size());
            return result;
        }

        std::vector<complex_type> result = project_tangent(at, vector_field_derivative_on_slice);
        if(!at.data.active())
        {
            return result;
        }

        const real_type matrix = at.data.slice_matrix;
        require_nonsingular(matrix);
        const real_type phase_vector_field = lsq_solver_type::phase_value(
            vector_field_on_slice, reference_on_slice_, active_modes_);
        const real_type alpha = phase_vector_field/matrix;
        const real_type matrix_derivative = lsq_solver_type::slice_matrix(
            tangent_on_slice, reference_on_slice_, active_modes_);
        const real_type state_generator_scale =
            phase_vector_field*matrix_derivative/(matrix*matrix);

        for(std::size_t mode = 0; mode < result.size(); ++mode)
        {
            const complex_type state_generator =
                translation_generator_1d_mode(mode, at.state_on_slice[mode]);
            const complex_type tangent_generator =
                translation_generator_1d_mode(mode, tangent_on_slice[mode]);
            result[mode] += scale_complex(tangent_generator, -alpha);
            result[mode] += scale_complex(state_generator, state_generator_scale);
        }
        return result;
    }

private:
    evaluation evaluate_single_mode(const std::vector<complex_type>& source) const
    {
        evaluation out;
        out.state_on_slice = source;
        if(active_modes_.empty())
        {
            return out;
        }

        const std::size_t mode = active_modes_.front();
        const real_type magnitude = std::abs(source[mode]);
        if(magnitude <= active_mode_tolerance_)
        {
            return out;
        }
        const real_type base_shift =
            -std::atan2(source[mode].imag(), source[mode].real())/static_cast<real_type>(mode);
        const real_type shift = closest_residual_shift(source, base_shift, mode);

        out.data.group_dimension = 1;
        out.data.active_rank = 1;
        out.data.mode = mode;
        out.data.active_modes = {mode};
        out.data.residual_group_order_value = mode;
        out.data.selected_abs = magnitude;
        out.data.selected_real_on_slice = magnitude;
        out.data.shift = shift;
        out.data.set_shift(0, shift);
        out.data.slice_matrix = static_cast<real_type>(mode)*magnitude;
        out.state_on_slice = slice_.apply_shift(source, shift);
        return out;
    }

    evaluation evaluate_lsq(const std::vector<complex_type>& source) const
    {
        evaluation out;
        out.uses_lsq = !active_modes_.empty();
        out.state_on_slice = source;
        if(active_modes_.empty())
        {
            return out;
        }

        const auto result = lsq_solver_.solve_fixed_modes(
            source,
            reference_on_slice_,
            active_modes_,
            lsq_options_.grid_points,
            lsq_options_.newton_iterations);
        const real_type shift = closest_residual_shift(
            source, result.shift, result.residual_group_order);
        out.state_on_slice = slice_.apply_shift(source, shift);

        out.data.group_dimension = 1;
        out.data.active_rank = 1;
        out.data.active_modes = active_modes_;
        out.data.mode = active_modes_.front();
        out.data.residual_group_order_value = result.residual_group_order;
        out.data.shift = shift;
        out.data.set_shift(0, shift);
        out.data.lsq_objective = result.objective;
        out.data.slice_matrix = lsq_solver_type::slice_matrix(
            out.state_on_slice, reference_on_slice_, active_modes_);
        for(const auto mode: active_modes_)
        {
            out.data.selected_abs = std::max(out.data.selected_abs, std::abs(source[mode]));
        }
        return out;
    }

    real_type closest_residual_shift(
        const std::vector<complex_type>& source,
        const real_type base_shift,
        const std::size_t order) const
    {
        const std::size_t copies = std::max<std::size_t>(order, 1);
        const real_type two_pi = real_type(2)*static_cast<real_type>(std::acos(real_type(-1)));
        real_type best_shift = base_shift;
        real_type best_distance = std::numeric_limits<real_type>::max();
        for(std::size_t copy = 0; copy < copies; ++copy)
        {
            const real_type shift =
                base_shift + two_pi*static_cast<real_type>(copy)/static_cast<real_type>(copies);
            const auto candidate = slice_.apply_shift(source, shift);
            const real_type distance = distance_sq(candidate, reference_on_slice_);
            if(distance < best_distance)
            {
                best_distance = distance;
                best_shift = shift;
            }
        }
        return best_shift;
    }

    void project_tangent_on_slice(
        const evaluation& at,
        const std::vector<complex_type>& tangent_on_slice,
        std::vector<complex_type>& result) const
    {
        if(!at.data.active())
        {
            result = tangent_on_slice;
            return;
        }
        if(!at.uses_lsq)
        {
            project_slice_tangent_1d(
                at.data,
                at.state_on_slice.data(),
                tangent_on_slice.data(),
                result.data(),
                result.size());
            return;
        }

        const real_type matrix = at.data.slice_matrix;
        require_nonsingular(matrix);
        const real_type phase = lsq_solver_type::phase_value(
            tangent_on_slice, reference_on_slice_, active_modes_);
        const real_type alpha = phase/matrix;
        for(std::size_t mode = 0; mode < result.size(); ++mode)
        {
            const complex_type generator =
                translation_generator_1d_mode(mode, at.state_on_slice[mode]);
            result[mode] = tangent_on_slice[mode] + scale_complex(generator, -alpha);
        }
    }

    void check_reference(const std::vector<complex_type>& reference) const
    {
        if(reference.size() <= 1)
        {
            throw std::invalid_argument("a frozen Fourier chart needs positive Fourier modes");
        }
    }

    void check_source(const std::vector<complex_type>& source) const
    {
        if(!frozen_)
        {
            throw std::logic_error("Fourier chart is not frozen");
        }
        if(source.size() != reference_on_slice_.size())
        {
            throw std::invalid_argument("frozen Fourier chart spectrum size mismatch");
        }
    }

    void check_evaluation(const evaluation& at) const
    {
        check_source(at.state_on_slice);
    }

    void require_nonsingular(const real_type matrix) const
    {
        const real_type scale = std::max<real_type>(real_type(1), std::abs(matrix));
        if(std::abs(matrix) <= std::numeric_limits<real_type>::epsilon()*scale)
        {
            throw std::runtime_error("frozen Fourier chart has a singular slice matrix");
        }
    }

    static real_type distance_sq(
        const std::vector<complex_type>& left,
        const std::vector<complex_type>& right)
    {
        real_type result = real_type(0);
        for(std::size_t i = 0; i < left.size(); ++i)
        {
            result += std::norm(left[i]-right[i]);
        }
        return result;
    }

    slice_type slice_;
    lsq_solver_type lsq_solver_;
    lsq_options_type lsq_options_;
    frozen_fourier_chart_1d_kind kind_ = frozen_fourier_chart_1d_kind::single_mode;
    real_type active_mode_tolerance_;
    std::vector<std::size_t> active_modes_;
    std::vector<complex_type> reference_on_slice_;
    std::size_t residual_group_order_ = 1;
    bool frozen_ = false;
};

} // namespace fourier
} // namespace symmetry

#endif
