#ifndef NMFD_TIME_STEPPERS_ADAPTATION_MATLAB_H
#define NMFD_TIME_STEPPERS_ADAPTATION_MATLAB_H
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <time_stepper/detail/scaled_error_mapping.h>
#include <time_stepper/detail/status.h>

namespace nmfd
{
namespace time_steppers
{
namespace integration
{
template<class VectorOperations>
class time_step_adaptation_matlab
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    struct params
    {
        scalar_type relative_tolerance = scalar_type(1e-8);
        scalar_type absolute_tolerance = scalar_type(1e-10);
        scalar_type initial_step = scalar_type(.01);
        scalar_type minimum_step = scalar_type(1e-14);
        scalar_type maximum_step = scalar_type(1);
    };

    time_step_adaptation_matlab(VectorOperations& operations, const params& p = {}):
        operations_(operations), params_(p)
    {
        const scalar_type values[] = {p.relative_tolerance, p.absolute_tolerance,
            p.initial_step, p.minimum_step, p.maximum_step};
        for (const auto value : values)
            if (!std::isfinite(value) || value <= 0)
                throw std::invalid_argument("Adaptive-step parameters must be positive and finite");
        if (p.minimum_step > p.maximum_step)
            throw std::invalid_argument("Inverted adaptive-step bounds");
        reset();
    }
    scalar_type initialize(scalar_type, const vector_type&, scalar_type dt = -1)
    {
        reset();
        if (dt > 0) dt_ = std::clamp(dt, params_.minimum_step, params_.maximum_step);
        return dt_;
    }
    void reset()
    {
        dt_ = std::clamp(params_.initial_step, params_.minimum_step, params_.maximum_step);
        rejected_ = false;
    }
    scalar_type get_dt() const { return dt_; }
    bool requires_error_estimate() const { return true; }

    adaptation_status assess(scalar_type, scalar_type dt_used, const vector_type& in,
        const vector_type& out, scalar_type& dt_next, unsigned int error_order = 0,
        const vector_type* error = nullptr)
    {
        if (!error || !error_order) return adaptation_status::failed;
        const detail::scaled_error_mapping<scalar_type> mapping{params_.absolute_tolerance / params_.relative_tolerance};
        const auto err = operations_.transform_reduce_max(mapping, *error, in, out) / params_.relative_tolerance;
        const auto h = std::abs(dt_used);
        if (!std::isfinite(err))
        {
            const auto decision = reject_step(dt_used);
            dt_next = dt_;
            return decision;
        }
        // Error is already a STATE error; do not multiply by h again.
        const auto factor = err == 0 ? scalar_type(5) :
            scalar_type(.8) * std::pow(scalar_type(1) / err, scalar_type(1) / error_order);
        if (err > 1)
        {
            dt_next = h * (rejected_ ? scalar_type(.5) : std::max(scalar_type(.1), factor));
            rejected_ = true;
            if (h <= params_.minimum_step) return adaptation_status::failed;
            dt_ = std::max(params_.minimum_step, dt_next);
            dt_next = dt_;
            return adaptation_status::rejected;
        }
        dt_next = rejected_ ? h : h * std::min(scalar_type(5), factor);
        dt_ = std::clamp(dt_next, params_.minimum_step, params_.maximum_step);
        dt_next = dt_;
        return adaptation_status::accepted;
    }
    void update(adaptation_status outcome, scalar_type, scalar_type h, const vector_type&)
    {
        if (outcome == adaptation_status::accepted_modified)
            dt_ = std::clamp(std::min(dt_, std::abs(h)), params_.minimum_step, params_.maximum_step);
        if (outcome != adaptation_status::rejected) rejected_ = false;
    }
    adaptation_status reject_step(scalar_type dt_used)
    {
        const auto h = std::abs(dt_used);
        if (h <= params_.minimum_step) return adaptation_status::failed;
        dt_ = std::max(params_.minimum_step, scalar_type(.5) * h);
        rejected_ = true;
        return adaptation_status::rejected;
    }

private:
    VectorOperations& operations_;
    params params_;
    scalar_type dt_;
    bool rejected_;
};
}
}
}
#endif
