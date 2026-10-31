#ifndef __SYMMETRY_CONTINUATION_ISOTROPY_TRANSITION_DETECTOR_H__
#define __SYMMETRY_CONTINUATION_ISOTROPY_TRANSITION_DETECTOR_H__

#include <symmetry/continuation/isotropy_transition.h>

namespace symmetry
{
namespace continuation
{

// Geometry owns representation-specific classification and backend workspaces.
template <class Geometry, class Vector, class T>
isotropy_transition_result<T> detect_isotropy_transition(
    Geometry &geometry, const Vector &previous, const Vector &candidate, const isotropy_transition_policy<T> &policy
)
{
    policy.validate();
    isotropy_transition_result<T> result;
    result.supported = true;
    if ( !policy.enabled )
        return result;
    const auto before           = geometry.classify( previous, policy.relative_mode_tolerance );
    const auto after            = geometry.classify( candidate, policy.relative_mode_tolerance );
    result.previous_orbit_type  = geometry.orbit_type( before );
    result.candidate_orbit_type = geometry.orbit_type( after );
    result.previous_stabilizer  = geometry.describe( before );
    result.candidate_stabilizer = geometry.describe( after );
    result.previous_order       = result.previous_orbit_type.finite_isotropy_order();
    result.candidate_order      = result.candidate_orbit_type.finite_isotropy_order();
    result.transition_order     = result.candidate_order;
    if ( geometry.is_zero( before ) || geometry.is_zero( after ) ||
         !geometry.proper_stabilizer_increase( before, after ) || result.candidate_order > policy.maximum_order )
        return result;
    result.previous_transverse_ratio  = geometry.transverse_ratio( previous, after );
    result.candidate_transverse_ratio = geometry.transverse_ratio( candidate, after );
    result.detected                   = result.previous_transverse_ratio > policy.relative_mode_tolerance &&
                      result.candidate_transverse_ratio <= policy.relative_mode_tolerance;
    return result;
}

}
}

#endif
