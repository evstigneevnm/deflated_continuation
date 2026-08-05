#ifndef __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_GROUP_2D_H__
#define __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_GROUP_2D_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <symmetry/fourier/mode_descriptor.h>

namespace symmetry
{
namespace fourier
{

template<class Real>
struct residual_translation_policy_2d
{
    Real absolute_active_tolerance = Real(0);
    Real relative_active_tolerance =
        Real(4096)*std::numeric_limits<Real>::epsilon();
    std::size_t basis_search_modes = 64;
    std::int64_t maximum_pair_determinant = 4096;
    bool factor_coordinate_half_shifts = true;
};

struct residual_translation_character_2d
{
    std::int64_t numerator_x = 0;
    std::int64_t numerator_y = 0;
    std::int64_t denominator = 1;

    bool is_identity() const
    {
        return numerator_x == 0 && numerator_y == 0;
    }

    bool compatible(const mode_index<2>& mode) const
    {
        return dot_numerator(mode)%denominator == 0;
    }

    int sign(const mode_index<2>& mode) const
    {
        const std::int64_t dot = dot_numerator(mode);
        if(dot%denominator != 0)
        {
            throw std::invalid_argument(
                "residual translation character is incompatible with a mode");
        }
        const std::int64_t quotient = dot/denominator;
        return quotient%2 == 0 ? 1 : -1;
    }

private:
    std::int64_t dot_numerator(const mode_index<2>& mode) const
    {
        return numerator_x*static_cast<std::int64_t>(mode[0]) +
            numerator_y*static_cast<std::int64_t>(mode[1]);
    }
};

template<class Real>
struct residual_translation_group_result_2d
{
    std::size_t active_rank = 0;
    Real maximum_amplitude = Real(0);
    Real active_tolerance = Real(0);
    std::vector<residual_translation_character_2d> characters;
};

namespace detail
{

inline std::int64_t positive_modulo(
    const std::int64_t value,
    const std::int64_t modulus)
{
    const std::int64_t result = value%modulus;
    return result < 0 ? result + modulus : result;
}

inline std::int64_t determinant(
    const mode_index<2>& left,
    const mode_index<2>& right)
{
    return static_cast<std::int64_t>(left[0])*right[1] -
        static_cast<std::int64_t>(left[1])*right[0];
}

inline std::int64_t extended_gcd_positive(
    const std::int64_t a,
    const std::int64_t b,
    std::int64_t& x,
    std::int64_t& y)
{
    if(b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    std::int64_t next_x = 0;
    std::int64_t next_y = 0;
    const std::int64_t gcd = extended_gcd_positive(
        b,
        a%b,
        next_x,
        next_y);
    x = next_y;
    y = next_x - (a/b)*next_y;
    return gcd;
}

inline std::array<std::int64_t, 3> extended_gcd(
    const std::int64_t a,
    const std::int64_t b)
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    const std::int64_t gcd = extended_gcd_positive(
        std::abs(a),
        std::abs(b),
        x,
        y);
    if(a < 0)
    {
        x = -x;
    }
    if(b < 0)
    {
        y = -y;
    }
    return {gcd, x, y};
}

template<class Real>
struct active_mode_2d
{
    mode_index<2> mode;
    Real amplitude = Real(0);
};

template<class Real>
std::vector<active_mode_2d<Real>> collect_active_modes(
    const std::vector<mode_index<2>>& modes,
    const std::vector<Real>& amplitudes,
    const residual_translation_policy_2d<Real>& policy,
    residual_translation_group_result_2d<Real>& result)
{
    if(modes.size() != amplitudes.size())
    {
        throw std::invalid_argument(
            "residual translation mode and amplitude sizes differ");
    }
    for(const Real value: amplitudes)
    {
        result.maximum_amplitude = std::max(
            result.maximum_amplitude,
            static_cast<Real>(std::abs(value)));
    }
    result.active_tolerance = std::max(
        policy.absolute_active_tolerance,
        policy.relative_active_tolerance*result.maximum_amplitude);

    std::vector<active_mode_2d<Real>> active;
    active.reserve(modes.size());
    for(std::size_t index = 0; index < modes.size(); ++index)
    {
        const Real amplitude = static_cast<Real>(std::abs(amplitudes[index]));
        if(amplitude > result.active_tolerance &&
           (modes[index][0] != 0 || modes[index][1] != 0))
        {
            active.push_back({modes[index], amplitude});
        }
    }
    std::stable_sort(
        active.begin(),
        active.end(),
        [](const auto& left, const auto& right)
        {
            const std::int64_t left_norm =
                static_cast<std::int64_t>(left.mode[0])*left.mode[0] +
                static_cast<std::int64_t>(left.mode[1])*left.mode[1];
            const std::int64_t right_norm =
                static_cast<std::int64_t>(right.mode[0])*right.mode[0] +
                static_cast<std::int64_t>(right.mode[1])*right.mode[1];
            if(left_norm != right_norm)
            {
                return left_norm < right_norm;
            }
            if(left.mode[0] != right.mode[0])
            {
                return left.mode[0] < right.mode[0];
            }
            return left.mode[1] < right.mode[1];
        });
    return active;
}

template<class Real>
std::vector<std::int8_t> signature(
    const residual_translation_character_2d& character,
    const std::vector<active_mode_2d<Real>>& active)
{
    std::vector<std::int8_t> result;
    result.reserve(active.size());
    for(const auto& observation: active)
    {
        if(!character.compatible(observation.mode))
        {
            return {};
        }
        result.push_back(static_cast<std::int8_t>(
            character.sign(observation.mode)));
    }
    return result;
}

inline bool signature_product_is_in_subgroup(
    const std::vector<std::int8_t>& left,
    const std::vector<std::int8_t>& right,
    const std::vector<std::vector<std::int8_t>>& subgroup)
{
    for(const auto& element: subgroup)
    {
        bool matches = element.size() == left.size() &&
            right.size() == left.size();
        for(std::size_t index = 0; matches && index < left.size(); ++index)
        {
            matches = left[index]*right[index] == element[index];
        }
        if(matches)
        {
            return true;
        }
    }
    return false;
}

template<class Real>
std::vector<residual_translation_character_2d> factor_half_shifts(
    const std::vector<residual_translation_character_2d>& input,
    const std::vector<active_mode_2d<Real>>& active,
    const bool enabled)
{
    if(!enabled || input.size() <= 1)
    {
        return input;
    }

    std::vector<std::vector<std::int8_t>> half_shift_signatures;
    for(const std::int64_t shift_x: {std::int64_t(0), std::int64_t(1)})
    {
        for(const std::int64_t shift_y: {std::int64_t(0), std::int64_t(1)})
        {
            const residual_translation_character_2d character{
                shift_x,
                shift_y,
                1};
            auto signs = signature(character, active);
            if(std::find(
                   half_shift_signatures.begin(),
                   half_shift_signatures.end(),
                   signs) == half_shift_signatures.end())
            {
                half_shift_signatures.push_back(std::move(signs));
            }
        }
    }

    struct candidate_type
    {
        residual_translation_character_2d character;
        std::vector<std::int8_t> signs;
    };
    std::vector<candidate_type> selected;
    for(const auto& character: input)
    {
        const auto signs = signature(character, active);
        bool represented = false;
        for(const auto& representative: selected)
        {
            if(signature_product_is_in_subgroup(
                   signs,
                   representative.signs,
                   half_shift_signatures))
            {
                represented = true;
                break;
            }
        }
        if(!represented)
        {
            selected.push_back({character, signs});
        }
    }

    std::vector<residual_translation_character_2d> result;
    result.reserve(selected.size());
    for(auto& candidate: selected)
    {
        result.push_back(candidate.character);
    }
    return result;
}

template<class Real>
std::vector<residual_translation_character_2d> rank_one_characters(
    const std::vector<active_mode_2d<Real>>& active)
{
    const auto first = active.front().mode;
    const std::int64_t first_gcd = std::gcd(
        std::abs(static_cast<std::int64_t>(first[0])),
        std::abs(static_cast<std::int64_t>(first[1])));
    std::int64_t primitive_x = first[0]/first_gcd;
    std::int64_t primitive_y = first[1]/first_gcd;
    if(primitive_x < 0 || (primitive_x == 0 && primitive_y < 0))
    {
        primitive_x = -primitive_x;
        primitive_y = -primitive_y;
    }

    std::int64_t coefficient_gcd = 0;
    for(const auto& observation: active)
    {
        const auto mode = observation.mode;
        const std::int64_t coefficient = primitive_x != 0
            ? mode[0]/primitive_x
            : mode[1]/primitive_y;
        if(mode[0] != coefficient*primitive_x ||
           mode[1] != coefficient*primitive_y)
        {
            throw std::logic_error(
                "rank-one residual translation modes are not collinear");
        }
        coefficient_gcd = std::gcd(
            coefficient_gcd,
            std::abs(coefficient));
    }
    const std::int64_t basis_x = coefficient_gcd*primitive_x;
    const std::int64_t basis_y = coefficient_gcd*primitive_y;
    const auto bezout = extended_gcd(basis_x, basis_y);
    if(bezout[0] == 0)
    {
        return {{{0, 0, 1}}};
    }
    residual_translation_character_2d nontrivial{
        positive_modulo(bezout[1], 2*bezout[0]),
        positive_modulo(bezout[2], 2*bezout[0]),
        bezout[0]};
    return {{0, 0, 1}, nontrivial};
}

template<class Real>
std::vector<residual_translation_character_2d> rank_two_characters(
    const std::vector<active_mode_2d<Real>>& active,
    const residual_translation_policy_2d<Real>& policy)
{
    const std::size_t search_count = std::min(
        active.size(),
        std::max<std::size_t>(policy.basis_search_modes, 1));
    std::int64_t best_determinant = 0;
    mode_index<2> first;
    mode_index<2> second;
    for(std::size_t left = 0; left < search_count; ++left)
    {
        for(std::size_t right = left + 1; right < active.size(); ++right)
        {
            const std::int64_t determinant_value =
                determinant(active[left].mode, active[right].mode);
            const std::int64_t magnitude = std::abs(determinant_value);
            if(magnitude != 0 &&
               (best_determinant == 0 || magnitude < std::abs(best_determinant)))
            {
                best_determinant = determinant_value;
                first = active[left].mode;
                second = active[right].mode;
                if(magnitude == 1)
                {
                    break;
                }
            }
        }
        if(std::abs(best_determinant) == 1)
        {
            break;
        }
    }
    if(best_determinant == 0)
    {
        throw std::logic_error(
            "rank-two residual translation basis was not found");
    }

    const std::int64_t determinant_magnitude = std::abs(best_determinant);
    if(determinant_magnitude > policy.maximum_pair_determinant)
    {
        throw std::runtime_error(
            "residual translation lattice determinant exceeds the configured limit");
    }
    const std::int64_t signed_scale = best_determinant < 0 ? -1 : 1;
    const std::int64_t modulus = 2*determinant_magnitude;
    std::set<std::pair<std::int64_t, std::int64_t>> numerators;
    for(std::int64_t first_integer = 0;
        first_integer < modulus;
        ++first_integer)
    {
        for(std::int64_t second_integer = 0;
            second_integer < modulus;
            ++second_integer)
        {
            const std::int64_t numerator_x = positive_modulo(
                signed_scale*(
                    static_cast<std::int64_t>(second[1])*first_integer -
                    static_cast<std::int64_t>(first[1])*second_integer),
                modulus);
            const std::int64_t numerator_y = positive_modulo(
                signed_scale*(
                    -static_cast<std::int64_t>(second[0])*first_integer +
                    static_cast<std::int64_t>(first[0])*second_integer),
                modulus);
            numerators.emplace(numerator_x, numerator_y);
        }
    }

    struct signed_character
    {
        residual_translation_character_2d character;
        std::vector<std::int8_t> signs;
    };
    std::vector<signed_character> unique;
    for(const auto& numerator: numerators)
    {
        residual_translation_character_2d character{
            numerator.first,
            numerator.second,
            determinant_magnitude};
        auto signs = signature(character, active);
        if(signs.empty())
        {
            continue;
        }
        const auto duplicate = std::find_if(
            unique.begin(),
            unique.end(),
            [&signs](const auto& value)
            {
                return value.signs == signs;
            });
        if(duplicate == unique.end())
        {
            unique.push_back({character, std::move(signs)});
        }
    }
    std::stable_sort(
        unique.begin(),
        unique.end(),
        [](const auto& left, const auto& right)
        {
            if(left.character.is_identity() != right.character.is_identity())
            {
                return left.character.is_identity();
            }
            if(left.character.denominator != right.character.denominator)
            {
                return left.character.denominator < right.character.denominator;
            }
            if(left.character.numerator_x != right.character.numerator_x)
            {
                return left.character.numerator_x < right.character.numerator_x;
            }
            return left.character.numerator_y < right.character.numerator_y;
        });

    std::vector<residual_translation_character_2d> result;
    result.reserve(unique.size());
    for(auto& value: unique)
    {
        result.push_back(value.character);
    }
    return result;
}

} // namespace detail

template<class Real>
residual_translation_group_result_2d<Real> build_residual_translation_group_2d(
    const std::vector<mode_index<2>>& modes,
    const std::vector<Real>& amplitudes,
    const residual_translation_policy_2d<Real>& policy = {})
{
    if(policy.absolute_active_tolerance < Real(0) ||
       policy.relative_active_tolerance < Real(0) ||
       policy.basis_search_modes == 0 ||
       policy.maximum_pair_determinant <= 0)
    {
        throw std::invalid_argument(
            "residual translation policy is invalid");
    }

    residual_translation_group_result_2d<Real> result;
    const auto active = detail::collect_active_modes(
        modes,
        amplitudes,
        policy,
        result);
    if(active.empty())
    {
        result.characters.push_back({0, 0, 1});
        return result;
    }

    bool rank_two = false;
    for(std::size_t index = 1; index < active.size(); ++index)
    {
        if(detail::determinant(active.front().mode, active[index].mode) != 0)
        {
            rank_two = true;
            break;
        }
    }
    result.active_rank = rank_two ? 2 : 1;
    auto characters = rank_two
        ? detail::rank_two_characters(active, policy)
        : detail::rank_one_characters(active);
    result.characters = detail::factor_half_shifts(
        characters,
        active,
        policy.factor_coordinate_half_shifts);
    if(result.characters.empty())
    {
        throw std::logic_error(
            "residual translation quotient has no identity representative");
    }
    return result;
}

} // namespace fourier
} // namespace symmetry

#endif
