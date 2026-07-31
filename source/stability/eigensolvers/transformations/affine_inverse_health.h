#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_AFFINE_INVERSE_HEALTH_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_AFFINE_INVERSE_HEALTH_H__

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class Real>
struct affine_inverse_health
{
    bool available = false;
    Real minimum_abs_denominator =
        std::numeric_limits<Real>::infinity();
    Real maximum_abs_denominator = Real{};
    Real minimum_relative_denominator =
        std::numeric_limits<Real>::infinity();
    bool relative_denominator_available = false;
    std::string diagnostic;

    bool valid() const
    {
        using std::isfinite;
        return
            available &&
            isfinite(minimum_abs_denominator) &&
            isfinite(maximum_abs_denominator) &&
            minimum_abs_denominator >= Real{} &&
            maximum_abs_denominator >= minimum_abs_denominator;
    }

    bool near_pole(
        Real absolute_tolerance,
        Real relative_tolerance) const
    {
        if(!available)
            return false;
        if(!valid())
            return true;
        return
            minimum_abs_denominator <= absolute_tolerance ||
            (
                relative_denominator_available &&
                (
                    !std::isfinite(
                        minimum_relative_denominator) ||
                    minimum_relative_denominator <=
                        relative_tolerance
                )
            );
    }
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
