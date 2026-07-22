#ifndef __SYMMETRY_CONTINUATION_ISOTROPY_TRANSITION_H__
#define __SYMMETRY_CONTINUATION_ISOTROPY_TRANSITION_H__

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <symmetry/translation/orbit_type.h>

namespace symmetry
{
namespace continuation
{

template<class T>
struct isotropy_transition_policy
{
    bool enabled = false;
    T relative_mode_tolerance = T(1.0e-10);
    std::size_t maximum_order = 16;
    unsigned int maximum_refinements = 6;
    // Interior fraction used to split the non-event/event step bracket.
    T refinement_step_factor = T(0.5);
    T registry_lambda_tolerance = T(1.0e-7);
    T registry_state_tolerance = T(1.0e-7);
    bool verbose = false;

    void validate() const
    {
        using std::isfinite;
        if(!isfinite(relative_mode_tolerance) || relative_mode_tolerance <= T(0))
        {
            throw std::invalid_argument(
                "isotropy transition relative mode tolerance must be finite and positive");
        }
        if(maximum_order < 2)
        {
            throw std::invalid_argument(
                "isotropy transition maximum order must be at least two");
        }
        if(maximum_refinements == 0)
        {
            throw std::invalid_argument(
                "isotropy transition maximum refinements must be positive");
        }
        if(!isfinite(refinement_step_factor) ||
           refinement_step_factor <= T(0) || refinement_step_factor >= T(1))
        {
            throw std::invalid_argument(
                "isotropy transition refinement step factor must be in (0,1)");
        }
        if(!isfinite(registry_lambda_tolerance) || registry_lambda_tolerance <= T(0))
        {
            throw std::invalid_argument(
                "isotropy event registry lambda tolerance must be finite and positive");
        }
        if(!isfinite(registry_state_tolerance) || registry_state_tolerance <= T(0))
        {
            throw std::invalid_argument(
                "isotropy event registry state tolerance must be finite and positive");
        }
    }
};

template<class T>
struct isotropy_transition_result
{
    bool supported = false;
    bool detected = false;
    std::size_t previous_order = 1;
    std::size_t candidate_order = 1;
    std::size_t transition_order = 1;
    symmetry::translation::orbit_type previous_orbit_type;
    symmetry::translation::orbit_type candidate_orbit_type;
    T previous_transverse_ratio = T(0);
    T candidate_transverse_ratio = T(0);
    unsigned int refinements = 0;
};

template<class T>
bool is_stabilizer_increase(
    const std::size_t previous_order,
    const std::size_t candidate_order,
    const isotropy_transition_policy<T>& policy)
{
    if(!policy.enabled || previous_order == 0 || candidate_order <= previous_order)
    {
        return false;
    }
    if(candidate_order > policy.maximum_order)
    {
        return false;
    }
    return candidate_order%previous_order == 0;
}

} // namespace continuation
} // namespace symmetry

#endif
