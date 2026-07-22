#ifndef __SYMMETRY_FOURIER_LSQ_PHASE_SOLVER_1D_H__
#define __SYMMETRY_FOURIER_LSQ_PHASE_SOLVER_1D_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace symmetry
{
namespace fourier
{

template<class Complex>
class lsq_phase_solver_1d
{
public:
    using complex_type = Complex;
    using real_type = typename complex_type::value_type;

    struct options
    {
        std::size_t mode_min = 1;
        std::size_t mode_max = 0;
        std::size_t max_active_modes = 8;
        std::size_t grid_points = 64;
        std::size_t newton_iterations = 8;
        bool prefer_trivial_residual_group = true;
        real_type minimum_coprime_relative_score = real_type(0.05);
    };

    struct result
    {
        bool active = false;
        real_type shift = real_type(0);
        real_type objective = real_type(0);
        real_type slice_matrix = real_type(0);
        std::size_t residual_group_order = 1;
        std::vector<std::size_t> active_modes;
    };

    result solve(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const real_type active_tolerance,
        const options& opts) const
    {
        if(source.size() != reference.size())
        {
            throw std::runtime_error("lsq_phase_solver_1d spectrum sizes do not match");
        }

        result out;
        if(source.size() <= 1)
        {
            return out;
        }

        out.active_modes = select_active_modes(source, reference, active_tolerance, opts);
        if(out.active_modes.empty())
        {
            return out;
        }

        out.active = true;
        out.residual_group_order = residual_group_order(out.active_modes);

        const real_type two_pi = real_type(2)*acos_minus_one();
        const std::size_t grid_points = std::max<std::size_t>(opts.grid_points, 8);
        real_type best_shift = real_type(0);
        real_type best_value = objective(source, reference, out.active_modes, real_type(0));

        for(std::size_t i = 1; i < grid_points; ++i)
        {
            const real_type theta = two_pi*static_cast<real_type>(i)/static_cast<real_type>(grid_points);
            const real_type value = objective(source, reference, out.active_modes, theta);
            if(value < best_value)
            {
                best_value = value;
                best_shift = theta;
            }
        }

        real_type theta = best_shift;
        for(std::size_t it = 0; it < opts.newton_iterations; ++it)
        {
            const auto deriv = objective_derivatives(source, reference, out.active_modes, theta);
            if(std::abs(deriv.second) <= std::numeric_limits<real_type>::epsilon())
            {
                break;
            }
            theta -= deriv.first/deriv.second;
            theta = normalize_angle(theta);
        }

        out.shift = theta;
        out.objective = objective(source, reference, out.active_modes, out.shift);
        out.slice_matrix = slice_matrix(source, reference, out.active_modes, out.shift);
        return out;
    }

    result solve_fixed_modes(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const std::vector<std::size_t>& active_modes,
        const std::size_t grid_points = 64,
        const std::size_t newton_iterations = 8) const
    {
        if(source.size() != reference.size())
        {
            throw std::runtime_error("lsq_phase_solver_1d spectrum sizes do not match");
        }

        result out;
        out.active_modes = active_modes;
        for(const auto mode: out.active_modes)
        {
            if(mode == 0 || mode >= source.size())
            {
                throw std::out_of_range("lsq_phase_solver_1d fixed active mode exceeds spectrum");
            }
        }
        if(out.active_modes.empty())
        {
            return out;
        }

        std::sort(out.active_modes.begin(), out.active_modes.end());
        out.active_modes.erase(
            std::unique(out.active_modes.begin(), out.active_modes.end()),
            out.active_modes.end());
        out.active = true;
        out.residual_group_order = residual_group_order(out.active_modes);

        const real_type two_pi = real_type(2)*acos_minus_one();
        const std::size_t grid_size = std::max<std::size_t>(grid_points, 8);
        real_type best_shift = real_type(0);
        real_type best_value = objective(source, reference, out.active_modes, real_type(0));
        for(std::size_t i = 1; i < grid_size; ++i)
        {
            const real_type theta = two_pi*static_cast<real_type>(i)/static_cast<real_type>(grid_size);
            const real_type value = objective(source, reference, out.active_modes, theta);
            if(value < best_value)
            {
                best_value = value;
                best_shift = theta;
            }
        }

        real_type theta = best_shift;
        for(std::size_t it = 0; it < newton_iterations; ++it)
        {
            const auto deriv = objective_derivatives(source, reference, out.active_modes, theta);
            if(std::abs(deriv.second) <= std::numeric_limits<real_type>::epsilon())
            {
                break;
            }
            theta = normalize_angle(theta-deriv.first/deriv.second);
        }

        out.shift = theta;
        out.objective = objective(source, reference, out.active_modes, theta);
        out.slice_matrix = slice_matrix(source, reference, out.active_modes, theta);
        return out;
    }

    static real_type phase_value(
        const std::vector<complex_type>& vector_on_slice,
        const std::vector<complex_type>& reference,
        const std::vector<std::size_t>& active_modes)
    {
        if(vector_on_slice.size() != reference.size())
        {
            throw std::runtime_error("lsq_phase_solver_1d phase spectrum sizes do not match");
        }
        real_type value = real_type(0);
        for(const auto mode: active_modes)
        {
            const complex_type product = vector_on_slice[mode]*std::conj(reference[mode]);
            value += static_cast<real_type>(mode)*product.imag();
        }
        return value;
    }

    static real_type slice_matrix(
        const std::vector<complex_type>& state_on_slice,
        const std::vector<complex_type>& reference,
        const std::vector<std::size_t>& active_modes)
    {
        if(state_on_slice.size() != reference.size())
        {
            throw std::runtime_error("lsq_phase_solver_1d slice spectrum sizes do not match");
        }
        real_type value = real_type(0);
        for(const auto mode: active_modes)
        {
            const complex_type product = state_on_slice[mode]*std::conj(reference[mode]);
            const real_type k = static_cast<real_type>(mode);
            value += k*k*product.real();
        }
        return value;
    }

private:
    static real_type acos_minus_one()
    {
        return static_cast<real_type>(std::acos(static_cast<real_type>(-1)));
    }

    static real_type normalize_angle(real_type theta)
    {
        const real_type two_pi = real_type(2)*acos_minus_one();
        theta = std::fmod(theta, two_pi);
        if(theta < real_type(0))
        {
            theta += two_pi;
        }
        return theta;
    }

    static real_type objective(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const std::vector<std::size_t>& active_modes,
        const real_type theta)
    {
        real_type value = real_type(0);
        for(const auto mode: active_modes)
        {
            const real_type phase = static_cast<real_type>(mode)*theta;
            const complex_type shifted = std::polar(real_type(1), phase)*source[mode];
            const complex_type delta = shifted - reference[mode];
            value += std::norm(delta);
        }
        return value;
    }

    static std::pair<real_type, real_type> objective_derivatives(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const std::vector<std::size_t>& active_modes,
        const real_type theta)
    {
        real_type first = real_type(0);
        real_type second = real_type(0);
        for(const auto mode: active_modes)
        {
            const real_type k = static_cast<real_type>(mode);
            const complex_type shifted = std::polar(real_type(1), k*theta)*source[mode];
            const complex_type product = shifted*std::conj(reference[mode]);
            first += real_type(2)*k*product.imag();
            second += real_type(2)*k*k*product.real();
        }
        return {first, second};
    }

    static real_type slice_matrix(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const std::vector<std::size_t>& active_modes,
        const real_type theta)
    {
        real_type value = real_type(0);
        for(const auto mode: active_modes)
        {
            const real_type k = static_cast<real_type>(mode);
            const complex_type shifted = std::polar(real_type(1), k*theta)*source[mode];
            const complex_type product = shifted*std::conj(reference[mode]);
            value += k*k*product.real();
        }
        return value;
    }

    static std::vector<std::size_t> select_active_modes(
        const std::vector<complex_type>& source,
        const std::vector<complex_type>& reference,
        const real_type active_tolerance,
        const options& opts)
    {
        const std::size_t mode_min = std::max<std::size_t>(opts.mode_min, 1);
        const std::size_t mode_max =
            opts.mode_max == 0 ? source.size() - 1 : std::min<std::size_t>(opts.mode_max, source.size() - 1);

        std::vector<std::pair<real_type, std::size_t>> candidates;
        for(std::size_t mode = mode_min; mode <= mode_max; ++mode)
        {
            const real_type source_abs = std::abs(source[mode]);
            const real_type reference_abs = std::abs(reference[mode]);
            if(source_abs <= active_tolerance || reference_abs <= active_tolerance)
            {
                continue;
            }
            const real_type score = source_abs*reference_abs*static_cast<real_type>(mode);
            candidates.push_back({score, mode});
        }

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const auto& left, const auto& right)
            {
                if(left.first == right.first)
                {
                    return left.second < right.second;
                }
                return left.first > right.first;
            });

        const std::size_t max_active =
            opts.max_active_modes == 0 ? candidates.size() : std::min<std::size_t>(opts.max_active_modes, candidates.size());
        std::vector<std::size_t> modes;
        modes.reserve(max_active);
        for(std::size_t i = 0; i < max_active; ++i)
        {
            modes.push_back(candidates[i].second);
        }

        if(opts.prefer_trivial_residual_group && !modes.empty() && !candidates.empty())
        {
            const real_type reliability_floor =
                opts.minimum_coprime_relative_score*candidates.front().first;
            std::size_t current_order = residual_group_order(modes);
            while(current_order > 1)
            {
                bool found_replacement = false;
                std::size_t best_position = 0;
                std::size_t best_mode = 0;
                std::size_t best_order = current_order;
                real_type best_retained_score = real_type(-1);

                for(const auto& candidate: candidates)
                {
                    if(candidate.first < reliability_floor ||
                       std::find(modes.begin(), modes.end(), candidate.second) != modes.end())
                    {
                        continue;
                    }

                    for(std::size_t position = 0; position < modes.size(); ++position)
                    {
                        std::vector<std::size_t> trial = modes;
                        trial[position] = candidate.second;
                        const std::size_t trial_order = residual_group_order(trial);
                        if(trial_order >= current_order)
                        {
                            continue;
                        }

                        real_type retained_score = candidate.first;
                        for(std::size_t i = 0; i < modes.size(); ++i)
                        {
                            if(i == position)
                            {
                                continue;
                            }
                            const auto selected = std::find_if(
                                candidates.begin(),
                                candidates.end(),
                                [&](const auto& entry) { return entry.second == modes[i]; });
                            if(selected != candidates.end())
                            {
                                retained_score += selected->first;
                            }
                        }

                        if(!found_replacement ||
                           trial_order < best_order ||
                           (trial_order == best_order && retained_score > best_retained_score))
                        {
                            found_replacement = true;
                            best_position = position;
                            best_mode = candidate.second;
                            best_order = trial_order;
                            best_retained_score = retained_score;
                        }
                    }
                }

                if(!found_replacement)
                {
                    break;
                }
                modes[best_position] = best_mode;
                current_order = best_order;
            }
        }

        std::sort(modes.begin(), modes.end());
        return modes;
    }

    static std::size_t residual_group_order(const std::vector<std::size_t>& modes)
    {
        if(modes.empty())
        {
            return 1;
        }
        std::size_t result = modes.front();
        for(const auto mode: modes)
        {
            result = std::gcd(result, mode);
        }
        return result == 0 ? std::size_t(1) : result;
    }
};

} // namespace fourier
} // namespace symmetry

#endif
