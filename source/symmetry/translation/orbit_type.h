#ifndef __SYMMETRY_TRANSLATION_ORBIT_TYPE_H__
#define __SYMMETRY_TRANSLATION_ORBIT_TYPE_H__

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace symmetry
{
namespace translation
{

struct orbit_type
{
    std::size_t group_dimension = 0;
    std::size_t active_rank = 0;
    std::size_t continuous_isotropy_dimension = 0;
    std::vector<std::uint64_t> finite_invariants;

    std::uint64_t finite_isotropy_order() const
    {
        return std::accumulate(
            finite_invariants.begin(),
            finite_invariants.end(),
            std::uint64_t(1),
            [](const std::uint64_t left, const std::uint64_t right)
            {
                return left*right;
            });
    }

    static orbit_type cyclic_1d(const std::size_t order)
    {
        orbit_type result;
        result.group_dimension = 1;
        result.active_rank = 1;
        result.continuous_isotropy_dimension = 0;
        result.finite_invariants = {
            static_cast<std::uint64_t>(order == 0 ? std::size_t(1) : order)};
        return result;
    }
};

inline bool operator==(const orbit_type& left, const orbit_type& right)
{
    return left.group_dimension == right.group_dimension &&
           left.active_rank == right.active_rank &&
           left.continuous_isotropy_dimension ==
               right.continuous_isotropy_dimension &&
           left.finite_invariants == right.finite_invariants;
}

inline bool operator!=(const orbit_type& left, const orbit_type& right)
{
    return !(left == right);
}

} // namespace translation
} // namespace symmetry

#endif
