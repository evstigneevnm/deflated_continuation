#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_STABILITY_GAUGE_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_STABILITY_GAUGE_H__

#include <stdexcept>

namespace symmetry
{
namespace linearization
{

template<class Scalar>
struct projected_stability_gauge
{
    Scalar operator_completion = Scalar{};
    Scalar scaled_eigenvalue = Scalar{};
};

template<class Scalar>
projected_stability_gauge<Scalar>
make_projected_stability_gauge(
    Scalar linearization_scale,
    bool left_halfplane_is_stable,
    Scalar magnitude = Scalar(1))
{
    if(linearization_scale == Scalar(0))
        throw std::invalid_argument(
            "projected stability gauge requires a nonzero "
            "linearization scale");
    if(!(magnitude > Scalar(0)))
        throw std::invalid_argument(
            "projected stability gauge magnitude must be positive");

    const Scalar scaled_eigenvalue =
        left_halfplane_is_stable ? -magnitude : magnitude;
    return {
        scaled_eigenvalue/linearization_scale,
        scaled_eigenvalue
    };
}

} // namespace linearization
} // namespace symmetry

#endif
