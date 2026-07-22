#ifndef __SYMMETRY_CONTINUATION_LOCAL_REPRESENTATIVE_POLICY_H__
#define __SYMMETRY_CONTINUATION_LOCAL_REPRESENTATIVE_POLICY_H__

#include <algorithm>
#include <limits>

namespace symmetry
{
namespace continuation
{

template<class T>
struct local_representative_policy
{
    T relative_tie_tolerance = T(0.05);
    T roundoff_multiplier = T(64);

    bool admissible(
        const T candidate_distance_sq,
        const T minimum_distance_sq,
        const T state_scale_sq) const
    {
        const T roundoff =
            roundoff_multiplier*std::numeric_limits<T>::epsilon()*std::max(T(1), state_scale_sq);
        const T relative = relative_tie_tolerance*std::max(minimum_distance_sq, roundoff);
        return candidate_distance_sq <= minimum_distance_sq + std::max(roundoff, relative);
    }
};

} // namespace continuation
} // namespace symmetry

#endif
