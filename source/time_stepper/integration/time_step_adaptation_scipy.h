#ifndef NMFD_TIME_STEPPERS_ADAPTATION_SCIPY_H
#define NMFD_TIME_STEPPERS_ADAPTATION_SCIPY_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <time_stepper/detail/scipy_error_mapping.h>
#include <time_stepper/detail/status.h>

namespace nmfd
{
namespace time_steppers
{
namespace integration
{
template<class VectorOperations>
class time_step_adaptation_scipy
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    static_assert(std::is_floating_point_v<scalar_type>, "Time adaptation requires a real floating-point scalar");

    struct params
    {
        scalar_type relative_tolerance = scalar_type(1e-8);
        scalar_type absolute_tolerance = scalar_type(1e-10);
        scalar_type initial_step = scalar_type(.01);
        scalar_type minimum_step = scalar_type(1e-14);
        scalar_type maximum_step = scalar_type(1);
    };

    time_step_adaptation_scipy(VectorOperations& operations, const params& p = {}) : operations_(operations), params_(p)
    {
        const scalar_type values[] = {
            p.relative_tolerance, p.absolute_tolerance, p.initial_step, p.minimum_step, p.maximum_step};
        for (const auto value : values)
        {
            if (!std::isfinite(value) || value <= 0)
            {
                throw std::invalid_argument("Adaptive-step parameters must be positive and finite");
            }
        }
        if (p.minimum_step > p.maximum_step || operations_.get_vector_size() == 0)
        {
            throw std::invalid_argument("Inverted adaptive-step bounds or empty vector space");
        }
        params_.relative_tolerance =
            std::max(p.relative_tolerance, scalar_type(100) * std::numeric_limits<scalar_type>::epsilon());
        reset();
    }

    scalar_type initialize(scalar_type time, const vector_type&, scalar_type dt = scalar_type(-1))
    {
        if (!std::isfinite(time) || !std::isfinite(dt) || (dt != scalar_type(-1) && dt <= 0))
        {
            throw std::invalid_argument("Invalid adaptation initialization time or step");
        }
        reset();
        if (dt > 0)
        {
            dt_ = std::clamp(dt, params_.minimum_step, params_.maximum_step);
        }
        return dt_;
    }

    void reset()
    {
        dt_ = std::clamp(params_.initial_step, params_.minimum_step, params_.maximum_step);
        rejected_ = false;
    }

    scalar_type get_dt() const
    {
        return dt_;
    }

    scalar_type relative_tolerance() const
    {
        return params_.relative_tolerance;
    }

    bool requires_error_estimate() const
    {
        return true;
    }

    adaptation_status assess(scalar_type time, scalar_type dt_used, const vector_type& in, const vector_type& out,
        scalar_type& dt_next, unsigned int error_order = 0, const vector_type* error = nullptr,
        const vector_type* error3 = nullptr)
    {
        dt_next = dt_;
        if (!std::isfinite(time) || !std::isfinite(dt_used) || dt_used == 0 || !error || !error_order ||
            (error3 && error_order != 8))
        {
            return adaptation_status::failed;
        }
        const auto count = static_cast<scalar_type>(operations_.get_vector_size());
        const detail::scipy_error_square_mapping<scalar_type> mapping{
            params_.absolute_tolerance, params_.relative_tolerance};
        const auto square = operations_.transform_reduce_sum(mapping, *error, in, out);
        auto ratio = std::sqrt(square / count);
        if (error3)
        {
            const auto square3 = operations_.transform_reduce_sum(mapping, *error3, in, out);
            if (!std::isfinite(square) || !std::isfinite(square3))
            {
                const auto decision = reject_step(dt_used);
                dt_next = dt_;
                return decision;
            }
            const auto rms3 = std::sqrt(square3 / count);
            // Both inputs are state errors: their step-size factor is already included.
            const auto denominator = std::hypot(ratio, scalar_type(.1) * rms3);
            ratio = denominator == 0 ? scalar_type(0) : ratio * (ratio / denominator);
        }
        return assess_error_ratio(dt_used, ratio, error_order, dt_next);
    }

    void update(adaptation_status outcome, scalar_type, scalar_type h, const vector_type&)
    {
        if (outcome == adaptation_status::accepted_modified && std::isfinite(h) && h != 0)
        {
            dt_ = std::clamp(std::min(dt_, std::abs(h)), params_.minimum_step, params_.maximum_step);
        }
        if (outcome != adaptation_status::rejected)
        {
            rejected_ = false;
        }
    }

    adaptation_status reject_step(scalar_type dt_used)
    {
        return reject_with_step(std::abs(dt_used) * scalar_type(.2));
    }

private:
    VectorOperations& operations_;
    params params_;
    scalar_type dt_;
    bool rejected_;

    adaptation_status reject_with_step(scalar_type next)
    {
        rejected_ = true;
        if (!std::isfinite(next) || next < params_.minimum_step)
        {
            return adaptation_status::failed;
        }
        dt_ = std::min(next, params_.maximum_step);
        return adaptation_status::rejected;
    }

    adaptation_status assess_error_ratio(
        scalar_type dt_used, scalar_type ratio, unsigned int error_order, scalar_type& dt_next)
    {
        if (!std::isfinite(ratio))
        {
            const auto decision = reject_step(dt_used);
            dt_next = dt_;
            return decision;
        }
        const auto h = std::abs(dt_used);
        const auto factor =
            ratio == 0 ? scalar_type(10) : scalar_type(.9) * std::pow(ratio, -scalar_type(1) / error_order);
        if (ratio < 1)
        {
            auto growth = std::min(scalar_type(10), factor);
            if (rejected_)
            {
                growth = std::min(scalar_type(1), growth);
            }
            dt_ = std::clamp(h * growth, params_.minimum_step, params_.maximum_step);
            dt_next = dt_;
            return adaptation_status::accepted;
        }
        const auto decision = reject_with_step(h * std::max(scalar_type(.2), factor));
        dt_next = dt_;
        return decision;
    }
};
}
}
}

#endif
