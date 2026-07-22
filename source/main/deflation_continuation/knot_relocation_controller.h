#ifndef __MAIN_DEFLATION_CONTINUATION_KNOT_RELOCATION_CONTROLLER_H__
#define __MAIN_DEFLATION_CONTINUATION_KNOT_RELOCATION_CONTROLLER_H__

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace main_classes
{
namespace deflation_continuation_detail
{

template<class Scalar, class IntersectionStatus, class Settings, class Log>
class knot_relocation_controller
{
public:
    template<class KnotValues>
    knot_relocation_controller(
        const Settings& settings,
        const KnotValues& knot_values,
        std::string project_directory,
        Log* log):
        settings_(settings),
        project_directory_(std::move(project_directory)),
        log_(log)
    {
        knots_.reserve(knot_values.size());
        for(const auto& value: knot_values)
        {
            knots_.push_back(static_cast<Scalar>(value));
        }
        std::sort(knots_.begin(), knots_.end());
        knots_.erase(std::unique(knots_.begin(), knots_.end()), knots_.end());
    }

    bool enabled() const
    {
        return settings_.enabled;
    }

    std::string registry_file_name() const
    {
        if(settings_.registry_file.empty())
        {
            return {};
        }
        if(settings_.registry_file.front() == '/')
        {
            return settings_.registry_file;
        }
        return project_directory_ + settings_.registry_file;
    }

    std::pair<Scalar, Scalar> bounds(const Scalar& requested_parameter) const
    {
        if(knots_.empty())
        {
            return {requested_parameter, requested_parameter};
        }
        if(knots_.size() == 1)
        {
            return {knots_.front(), knots_.front()};
        }

        for(std::size_t index = 0; index < knots_.size(); ++index)
        {
            if(same_parameter(knots_[index], requested_parameter))
            {
                const Scalar lower = index == 0
                    ? knots_[index]
                    : knots_[index - 1];
                const Scalar upper = index + 1 >= knots_.size()
                    ? knots_[index]
                    : knots_[index + 1];
                return {lower, upper};
            }
        }

        const auto upper = std::upper_bound(
            knots_.begin(),
            knots_.end(),
            requested_parameter);
        if(upper == knots_.begin())
        {
            return {knots_.front(), knots_.front()};
        }
        if(upper == knots_.end())
        {
            return {knots_.back(), knots_.back()};
        }
        return {*(upper - 1), *upper};
    }

    Scalar candidate(
        const Scalar& requested_parameter,
        const unsigned int candidate_index) const
    {
        const unsigned int radius_count = std::max(
            1u,
            (settings_.candidate_count + 1u)/2u);
        const unsigned int radius_index = candidate_index/2u;
        const Scalar alpha = radius_count == 1u
            ? Scalar(0)
            : static_cast<Scalar>(radius_index)/
                static_cast<Scalar>(radius_count - 1u);
        const Scalar radius = settings_.min_shift_abs +
            alpha*(settings_.max_shift_abs - settings_.min_shift_abs);
        const bool positive_slot = (candidate_index%2u) == 0u;
        const Scalar sign = positive_slot == settings_.prefer_positive_shift
            ? Scalar(1)
            : Scalar(-1);
        return requested_parameter + sign*radius;
    }

    template<class Registry, class RebuildIntersections>
    bool relocate_restart_intersection(
        const Scalar& requested_parameter,
        const IntersectionStatus& failed_status,
        Registry& registry,
        Scalar& effective_parameter,
        IntersectionStatus& effective_status,
        RebuildIntersections&& rebuild_intersections) const
    {
        if(!enabled() || settings_.candidate_count == 0)
        {
            return false;
        }

        const unsigned int required_intersections =
            failed_status.added +
            failed_status.failed +
            failed_status.missing_data +
            failed_status.skipped_incomplete;
        const auto candidate_bounds = bounds(requested_parameter);
        for(unsigned int index = 0; index < settings_.candidate_count; ++index)
        {
            const Scalar candidate_parameter = candidate(requested_parameter, index);
            if(!inside(candidate_parameter, candidate_bounds))
            {
                continue;
            }
            const auto status = rebuild_intersections(candidate_parameter);
            if(!acceptable(status, required_intersections))
            {
                continue;
            }

            effective_parameter = candidate_parameter;
            effective_status = status;
            registry.set(
                requested_parameter,
                effective_parameter,
                "intersection_newton_failed_at_requested_knot",
                effective_status);
            if(settings_.save_registry)
            {
                registry.save();
            }
            if(log_ != nullptr)
            {
                log_->warning_f(
                    "MAIN:deflation_continuation: relocated requested knot %le to non-singular knot %le after restart intersection failure; added = %u, skipped_discontinuous = %u, skipped_incomplete = %u.",
                    double(requested_parameter),
                    double(effective_parameter),
                    effective_status.added,
                    effective_status.skipped_discontinuous,
                    effective_status.skipped_incomplete);
            }
            return true;
        }
        return false;
    }

    template<class Vector, class Registry, class Interpolate>
    bool relocate_active_intersection(
        const Scalar& requested_parameter,
        const Scalar& parameter_left,
        const Vector& value_left,
        const Scalar& parameter_right,
        const Vector& value_right,
        Registry& registry,
        Scalar& effective_parameter,
        Vector& effective_value,
        Interpolate&& interpolate) const
    {
        if(!enabled() ||
           settings_.candidate_count == 0 ||
           same_parameter(parameter_left, parameter_right))
        {
            return false;
        }

        const auto candidate_bounds = bounds(requested_parameter);
        for(unsigned int index = 0; index < settings_.candidate_count; ++index)
        {
            const Scalar candidate_parameter = candidate(requested_parameter, index);
            if(!inside(candidate_parameter, candidate_bounds) ||
               !inside_active_interval(
                   candidate_parameter,
                   parameter_left,
                   parameter_right))
            {
                continue;
            }

            if(!interpolate(
                   candidate_parameter,
                   parameter_left,
                   value_left,
                   parameter_right,
                   value_right,
                   effective_value))
            {
                if(log_ != nullptr)
                {
                    log_->info_f(
                        "MAIN:deflation_continuation: active knot relocation candidate %le for requested knot %le failed Newton interpolation.",
                        double(candidate_parameter),
                        double(requested_parameter));
                }
                continue;
            }

            effective_parameter = candidate_parameter;
            IntersectionStatus status;
            status.added = 1;
            registry.set(
                requested_parameter,
                effective_parameter,
                "active_continuation_interpolation_failed_at_requested_knot",
                status);
            if(settings_.save_registry)
            {
                registry.save();
            }
            if(log_ != nullptr)
            {
                log_->warning_f(
                    "MAIN:deflation_continuation: shifted active continuation knot %le to validated non-singular knot %le after interpolation Newton failure.",
                    double(requested_parameter),
                    double(effective_parameter));
            }
            return true;
        }

        if(log_ != nullptr)
        {
            log_->warning_f(
                "MAIN:deflation_continuation: active continuation knot relocation failed for requested knot %le inside step [%le, %le].",
                double(requested_parameter),
                double(parameter_left),
                double(parameter_right));
        }
        return false;
    }

private:
    static Scalar abs_value(const Scalar& value)
    {
        return value < Scalar(0) ? -value : value;
    }

    static bool same_parameter(const Scalar& left, const Scalar& right)
    {
        const Scalar scale = std::max<Scalar>(
            Scalar(1),
            std::max<Scalar>(abs_value(left), abs_value(right)));
        return abs_value(left - right) <=
            Scalar(64)*std::numeric_limits<Scalar>::epsilon()*scale;
    }

    static bool inside(
        const Scalar& value,
        const std::pair<Scalar, Scalar>& interval)
    {
        return value > interval.first && value < interval.second;
    }

    static bool inside_active_interval(
        const Scalar& value,
        const Scalar& left,
        const Scalar& right)
    {
        return value > std::min(left, right) && value < std::max(left, right);
    }

    bool acceptable(
        const IntersectionStatus& status,
        const unsigned int required_intersections) const
    {
        if(!status.ok())
        {
            return false;
        }
        return !settings_.require_all_intersections ||
               status.added >= required_intersections;
    }

    Settings settings_;
    std::vector<Scalar> knots_;
    std::string project_directory_;
    Log* log_;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_KNOT_RELOCATION_CONTROLLER_H__
