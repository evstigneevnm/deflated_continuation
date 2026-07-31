#ifndef __CONTINUATION_OBSERVATIONAL_KNOT_SAMPLE_H__
#define __CONTINUATION_OBSERVATIONAL_KNOT_SAMPLE_H__

namespace continuation
{

enum class observational_knot_sample_status
{
    not_crossed,
    sampled_requested,
    sampled_relocated,
    failed
};

template<class Scalar>
struct observational_knot_sample_result
{
    observational_knot_sample_status status =
        observational_knot_sample_status::not_crossed;
    Scalar requested_parameter = Scalar(0);
    Scalar attempted_parameter = Scalar(0);
    Scalar sampled_parameter = Scalar(0);

    bool crossed() const
    {
        return status != observational_knot_sample_status::not_crossed;
    }

    bool sampled() const
    {
        return status ==
                   observational_knot_sample_status::sampled_requested ||
               status ==
                   observational_knot_sample_status::sampled_relocated;
    }

    bool relocated() const
    {
        return status ==
               observational_knot_sample_status::sampled_relocated;
    }
};

template<class Scalar>
bool parameter_is_bracketed(
    const Scalar& value,
    const Scalar& left,
    const Scalar& right)
{
    return (value - left)*(value - right) <= Scalar(0);
}

template<class Scalar, class Vector, class Interpolate, class Relocate>
observational_knot_sample_result<Scalar> sample_knot_observationally(
    const Scalar& requested_parameter,
    const Scalar& attempted_parameter,
    const Scalar& parameter_left,
    const Vector& value_left,
    const Scalar& parameter_right,
    const Vector& value_right,
    Vector& sample,
    Interpolate&& interpolate,
    Relocate&& relocate)
{
    observational_knot_sample_result<Scalar> result;
    result.requested_parameter = requested_parameter;
    result.attempted_parameter = attempted_parameter;
    result.sampled_parameter = attempted_parameter;

    if(!parameter_is_bracketed(
           attempted_parameter,
           parameter_left,
           parameter_right))
    {
        return result;
    }

    result.status = observational_knot_sample_status::failed;
    if(interpolate(
           attempted_parameter,
           parameter_left,
           value_left,
           parameter_right,
           value_right,
           sample))
    {
        result.status =
            observational_knot_sample_status::sampled_requested;
        return result;
    }

    Scalar relocated_parameter = attempted_parameter;
    if(relocate(
           requested_parameter,
           parameter_left,
           value_left,
           parameter_right,
           value_right,
           relocated_parameter,
           sample))
    {
        result.sampled_parameter = relocated_parameter;
        result.status =
            observational_knot_sample_status::sampled_relocated;
    }
    return result;
}

} // namespace continuation

#endif // __CONTINUATION_OBSERVATIONAL_KNOT_SAMPLE_H__
