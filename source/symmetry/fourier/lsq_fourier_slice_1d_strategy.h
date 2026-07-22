#ifndef __SYMMETRY_FOURIER_LSQ_FOURIER_SLICE_1D_STRATEGY_H__
#define __SYMMETRY_FOURIER_LSQ_FOURIER_SLICE_1D_STRATEGY_H__

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <symmetry/fourier/fourier_slice_1d.h>
#include <symmetry/fourier/lsq_fourier_slice_1d_policy.h>
#include <symmetry/fourier/lsq_phase_solver_1d.h>
#include <symmetry/fourier/translation_generators.h>

namespace symmetry
{
namespace fourier
{

template<class Complex>
class lsq_fourier_slice_1d_strategy
{
public:
    using complex_type = Complex;
    using scalar_type = typename complex_type::value_type;
    using slice_data_type = typename fourier_slice_1d<complex_type>::slice_data;
    using solver_type = lsq_phase_solver_1d<complex_type>;
    using options_type = typename solver_type::options;
    using policy_type = lsq_fourier_slice_1d_policy<scalar_type>;

    void configure(const policy_type& policy)
    {
        policy.validate();
        options_.mode_min = policy.mode_min;
        options_.mode_max = policy.mode_max;
        options_.max_active_modes = policy.max_active_modes;
        options_.grid_points = policy.grid_points;
        options_.newton_iterations = policy.newton_iterations;
        options_.prefer_trivial_residual_group = policy.prefer_trivial_residual_group;
        options_.minimum_coprime_relative_score = policy.minimum_coprime_relative_score;
    }

    const options_type& options() const
    {
        return options_;
    }

    slice_data_type make_slice_data(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const scalar_type tolerance) const
    {
        slice_data_type data;
        data.group_dimension = 1;
        data.tolerance = tolerance;

        const auto result = solver_.solve(source, reference, tolerance, options_);
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
        for(const auto mode: result.active_modes)
        {
            data.selected_abs = std::max<scalar_type>(data.selected_abs, std::abs(source[mode]));
        }
        return data;
    }

    void project_tangent(
        const slice_data_type& data,
        const std::vector<complex_type>& state_on_slice,
        const std::vector<complex_type>& reference,
        const std::vector<complex_type>& tangent_on_slice,
        std::vector<complex_type>& projected) const
    {
        if(projected.size() != tangent_on_slice.size() ||
           state_on_slice.size() != tangent_on_slice.size() ||
           reference.size() != tangent_on_slice.size())
        {
            throw std::runtime_error("LSQ Fourier projection spectrum sizes do not match");
        }
        if(!data.active() || data.active_modes.empty())
        {
            projected = tangent_on_slice;
            return;
        }

        const scalar_type matrix = solver_type::slice_matrix(
            state_on_slice,
            reference,
            data.active_modes);
        if(std::abs(matrix) <= std::numeric_limits<scalar_type>::epsilon())
        {
            throw std::runtime_error("LSQ Fourier projection got a singular slice matrix");
        }

        const scalar_type phase = solver_type::phase_value(
            tangent_on_slice,
            reference,
            data.active_modes);
        const scalar_type alpha = phase/matrix;
        for(std::size_t mode = 0; mode < tangent_on_slice.size(); ++mode)
        {
            const complex_type generator = translation_generator_1d_mode(mode, state_on_slice[mode]);
            projected[mode] = tangent_on_slice[mode] + scale_complex(generator, -alpha);
        }
    }

    void projected_vector_field_differential(
        const slice_data_type& data,
        const std::vector<complex_type>& state_on_slice,
        const std::vector<complex_type>& reference,
        const std::vector<complex_type>& tangent_on_slice,
        const std::vector<complex_type>& vector_field_on_slice,
        const std::vector<complex_type>& vector_field_derivative_on_slice,
        std::vector<complex_type>& projection_work,
        std::vector<complex_type>& projected_derivative) const
    {
        project_tangent(
            data,
            state_on_slice,
            reference,
            vector_field_derivative_on_slice,
            projection_work);

        if(!data.active() || data.active_modes.empty())
        {
            projected_derivative = projection_work;
            return;
        }

        const scalar_type matrix = solver_type::slice_matrix(
            state_on_slice,
            reference,
            data.active_modes);
        if(std::abs(matrix) <= std::numeric_limits<scalar_type>::epsilon())
        {
            throw std::runtime_error(
                "LSQ Fourier vector-field differential got a singular slice matrix");
        }

        const scalar_type phase_vector_field = solver_type::phase_value(
            vector_field_on_slice,
            reference,
            data.active_modes);
        const scalar_type alpha = phase_vector_field/matrix;
        const scalar_type matrix_derivative = solver_type::slice_matrix(
            tangent_on_slice,
            reference,
            data.active_modes);
        const scalar_type state_generator_scale =
            phase_vector_field*matrix_derivative/(matrix*matrix);

        for(std::size_t mode = 0; mode < projected_derivative.size(); ++mode)
        {
            const complex_type state_generator =
                translation_generator_1d_mode(mode, state_on_slice[mode]);
            const complex_type tangent_generator =
                translation_generator_1d_mode(mode, tangent_on_slice[mode]);
            complex_type value = projection_work[mode];
            value += scale_complex(tangent_generator, -alpha);
            value += scale_complex(state_generator, state_generator_scale);
            projected_derivative[mode] = value;
        }
    }

private:
    solver_type solver_;
    options_type options_;
};

} // namespace fourier
} // namespace symmetry

#endif
