#ifndef __MAIN_DEFLATION_CONTINUATION_KNOT_EXECUTOR_H__
#define __MAIN_DEFLATION_CONTINUATION_KNOT_EXECUTOR_H__

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>

#include <continuation/continuation_result.h>
#include <main/deflation_continuation/rejected_candidate_cache.h>

namespace main_classes
{
namespace deflation_continuation_detail
{

template<class Scalar>
struct knot_execution_policy
{
    bool relocation_enabled = false;
    bool allow_incomplete_restart_intersections = false;
    bool allow_failed_continuation_curve_save = false;
    bool preserve_partial_curves = true;
    unsigned int max_failed_continuations_per_knot = 0;
    Scalar failed_continuation_rejection_tolerance = Scalar(1.0e-8);
    bool check_duplicate_after_deflation = true;
    Scalar duplicate_after_deflation_tolerance = Scalar(1.0e-8);
    unsigned int duplicate_after_deflation_retries = 0;
};

struct knot_execution_result
{
    unsigned int knots_visited = 0;
    unsigned int solutions_found = 0;
    unsigned int accepted_curves = 0;
    unsigned int discarded_curves = 0;
    unsigned int partial_curves = 0;
    unsigned int skipped_incomplete_knots = 0;
};

template<class Scalar, class Vector, class IntersectionStatus>
struct knot_executor_callbacks
{
    std::function<bool()> load_archive;
    std::function<void(bool)> build_analytical_branches;
    std::function<void()> restore_output_settings;
    std::function<int()> current_curve_count;
    std::function<bool(const Scalar&, Scalar&)> resolve_parameter;
    std::function<std::string()> relocation_registry_file;
    std::function<IntersectionStatus(const Scalar&)> rebuild_intersections;
    std::function<bool(
        const Scalar&,
        const IntersectionStatus&,
        Scalar&,
        IntersectionStatus&)> relocate_intersection;
    std::function<bool(const Scalar&)> find_deflated_solution;
    std::function<void(Vector&)> get_deflated_solution;
    std::function<void(Vector&)> stabilize;
    std::function<bool(Vector&, Scalar&)> nearest_known_distance;
    std::function<bool(
        const Scalar&,
        const Vector&,
        Scalar&,
        std::uint64_t&)> nearest_persistent_rejection;
    std::function<void(std::uint64_t)> mark_persistent_rejection_seen;
    std::function<void(
        const Scalar&,
        const Scalar&,
        const Vector&,
        const continuation::continuation_curve_result<Scalar>&)>
        record_persistent_rejection;
    std::function<continuation::continuation_curve_result<Scalar>(
        Vector&,
        const Scalar&)> continue_candidate;
    std::function<void(
        const Scalar&,
        const continuation::continuation_curve_result<Scalar>&)>
        accept_candidate;
    std::function<void()> discard_candidate;
    std::function<void()> save_archive;
};

template<class VectorOperations, class Knots, class Log, class IntersectionStatus>
class knot_executor
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using callbacks_type = knot_executor_callbacks<
        scalar_type,
        vector_type,
        IntersectionStatus>;

    knot_executor(
        VectorOperations* vector_operations,
        Knots* knots,
        Log* log,
        knot_execution_policy<scalar_type> policy,
        callbacks_type callbacks):
        vector_operations_(vector_operations),
        knots_(knots),
        log_(log),
        policy_(std::move(policy)),
        callbacks_(std::move(callbacks))
    {
    }

    knot_execution_result execute()
    {
        knot_execution_result result;
        const bool archive_exists = callbacks_.load_archive();
        callbacks_.restore_output_settings();

        vector_type deflated_solution;
        bool has_next_knot = knots_->next();
        if(has_next_knot)
        {
            callbacks_.build_analytical_branches(archive_exists);
        }

        rejected_candidate_cache<VectorOperations> rejected_candidates(
            vector_operations_);
        scalar_type rejected_cache_parameter = scalar_type(0);
        bool rejected_cache_active = false;
        unsigned int failed_continuations_at_knot = 0;

        while(has_next_knot)
        {
            ++result.knots_visited;
            const scalar_type requested_parameter = knots_->get_value();
            scalar_type effective_parameter = requested_parameter;
            if(policy_.relocation_enabled &&
               callbacks_.resolve_parameter(
                   requested_parameter,
                   effective_parameter))
            {
                log_->warning_f(
                    "MAIN:deflation_continuation: requested knot %le is mapped to validated non-singular knot %le from %s.",
                    double(requested_parameter),
                    double(effective_parameter),
                    callbacks_.relocation_registry_file().c_str());
            }

            log_->info_f(
                "MAIN:deflation_continuation: currently having %i curves.",
                callbacks_.current_curve_count());
            auto intersection_status =
                callbacks_.rebuild_intersections(effective_parameter);
            if(!intersection_status.ok() && policy_.relocation_enabled)
            {
                scalar_type relocated_parameter = effective_parameter;
                IntersectionStatus relocated_status;
                if(callbacks_.relocate_intersection(
                       requested_parameter,
                       intersection_status,
                       relocated_parameter,
                       relocated_status))
                {
                    effective_parameter = relocated_parameter;
                    intersection_status = relocated_status;
                }
                else
                {
                    intersection_status =
                        callbacks_.rebuild_intersections(effective_parameter);
                }
            }

            const bool intersections_incomplete = !intersection_status.ok();
            if(intersections_incomplete &&
               !policy_.allow_incomplete_restart_intersections)
            {
                log_incomplete_intersections(
                    "skipping deflation",
                    requested_parameter,
                    effective_parameter,
                    intersection_status);
                ++result.skipped_incomplete_knots;
                has_next_knot = knots_->next();
                continue;
            }
            if(intersections_incomplete)
            {
                log_incomplete_intersections(
                    "continuing with incomplete restart intersections",
                    requested_parameter,
                    effective_parameter,
                    intersection_status);
            }

            if(!rejected_cache_active ||
               !same_parameter(
                   effective_parameter,
                   rejected_cache_parameter))
            {
                rejected_candidates.clear();
                rejected_cache_parameter = effective_parameter;
                rejected_cache_active = true;
                failed_continuations_at_knot = 0;
            }

            const unsigned int maximum_failed =
                policy_.max_failed_continuations_per_knot;
            if(maximum_failed > 0 &&
               failed_continuations_at_knot >= maximum_failed)
            {
                log_->warning_f(
                    "MAIN:deflation_continuation: skipping requested lambda = %lf, effective lambda = %lf after %u failed continuation attempts at this knot.",
                    double(requested_parameter),
                    double(effective_parameter),
                    failed_continuations_at_knot);
                has_next_knot = knots_->next();
                continue;
            }

            bool is_new_solution = false;
            bool candidate_duplicate = false;
            unsigned int duplicate_retry = 0;
            do
            {
                candidate_duplicate = false;
                is_new_solution =
                    callbacks_.find_deflated_solution(effective_parameter);
                if(is_new_solution)
                {
                    callbacks_.get_deflated_solution(deflated_solution);
                    callbacks_.stabilize(deflated_solution);
                    scalar_type nearest_rejected_distance =
                        std::numeric_limits<scalar_type>::infinity();
                    const bool rejected_distance_available =
                        rejected_candidates.nearest_distance(
                            effective_parameter,
                            deflated_solution,
                            nearest_rejected_distance);
                    std::uint64_t persistent_rejection_id = 0;
                    scalar_type persistent_rejection_distance =
                        std::numeric_limits<scalar_type>::infinity();
                    const bool persistent_rejection_available =
                        callbacks_.nearest_persistent_rejection &&
                        callbacks_.nearest_persistent_rejection(
                            effective_parameter,
                            deflated_solution,
                            persistent_rejection_distance,
                            persistent_rejection_id);
                    const bool rejected_by_session =
                        rejected_distance_available &&
                        nearest_rejected_distance <=
                            policy_.failed_continuation_rejection_tolerance;
                    const bool rejected_by_persistence =
                        persistent_rejection_available &&
                        persistent_rejection_distance <=
                            policy_.failed_continuation_rejection_tolerance;
                    if(rejected_by_session || rejected_by_persistence)
                    {
                        const scalar_type reported_distance =
                            std::min(
                                nearest_rejected_distance,
                                persistent_rejection_distance);
                        if(rejected_by_persistence &&
                           callbacks_.mark_persistent_rejection_seen)
                        {
                            callbacks_.mark_persistent_rejection_seen(
                                persistent_rejection_id);
                        }
                        candidate_duplicate = true;
                        is_new_solution = false;
                        log_->warning_f(
                            "MAIN:deflation_continuation: deflated Newton returned a candidate rejected after failed continuation at lambda = %lf with stabilized distance = %le and tolerance = %le%s.",
                            double(effective_parameter),
                            double(reported_distance),
                            double(policy_.failed_continuation_rejection_tolerance),
                            rejected_by_persistence
                                ? " (persistent registry)"
                                : "");
                    }
                    if(!candidate_duplicate &&
                       policy_.check_duplicate_after_deflation)
                    {
                        scalar_type nearest_distance =
                            std::numeric_limits<scalar_type>::infinity();
                        if(callbacks_.nearest_known_distance(
                               deflated_solution,
                               nearest_distance) &&
                           nearest_distance <=
                               policy_.duplicate_after_deflation_tolerance)
                        {
                            candidate_duplicate = true;
                            is_new_solution = false;
                            log_->warning_f(
                                "MAIN:deflation_continuation: deflated Newton returned a duplicate solution at lambda = %lf with stabilized distance = %le and tolerance = %le.",
                                double(effective_parameter),
                                double(nearest_distance),
                                double(policy_.duplicate_after_deflation_tolerance));
                        }
                    }
                }
                ++duplicate_retry;
            }
            while(candidate_duplicate &&
                  duplicate_retry <=
                      policy_.duplicate_after_deflation_retries);

            if(!is_new_solution)
            {
                has_next_knot = knots_->next();
                continue;
            }

            ++result.solutions_found;
            log_->info_f(
                "MAIN:deflation_continuation: found %i solutions for lambda = %lf.",
                static_cast<int>(result.solutions_found),
                double(effective_parameter));
            const auto continuation_result = callbacks_.continue_candidate(
                deflated_solution,
                effective_parameter);
            const bool preserve_partial =
                policy_.preserve_partial_curves &&
                continuation_result.has_valid_progress();
            if(continuation_result.complete() || preserve_partial ||
               policy_.allow_failed_continuation_curve_save)
            {
                callbacks_.accept_candidate(
                    effective_parameter,
                    continuation_result);
                ++result.accepted_curves;
                if(!continuation_result.complete())
                {
                    ++result.partial_curves;
                    log_->warning_f(
                        "MAIN:deflation_continuation: preserved a recoverable partial curve at lambda = %lf with %u started semicurves.",
                        double(effective_parameter),
                        continuation_result.semicurves_started);
                }
                continue;
            }

            log_->warning_f(
                "MAIN:deflation_continuation: continuation of a new curve at lambda = %lf failed; discarding curve according to restart policy.",
                double(effective_parameter));
            callbacks_.discard_candidate();
            rejected_candidates.add(
                effective_parameter,
                deflated_solution);
            if(callbacks_.record_persistent_rejection)
            {
                callbacks_.record_persistent_rejection(
                    requested_parameter,
                    effective_parameter,
                    deflated_solution,
                    continuation_result);
            }
            ++failed_continuations_at_knot;
            ++result.discarded_curves;
            log_->warning_f(
                "MAIN:deflation_continuation: stored failed-continuation candidate rejection %u/%u at lambda = %lf.",
                failed_continuations_at_knot,
                maximum_failed,
                double(effective_parameter));
            callbacks_.save_archive();
            if(maximum_failed > 0 &&
               failed_continuations_at_knot >= maximum_failed)
            {
                log_->warning_f(
                    "MAIN:deflation_continuation: reached max_failed_continuations_per_knot = %u at requested lambda = %lf, effective lambda = %lf; advancing to the next knot.",
                    maximum_failed,
                    double(requested_parameter),
                    double(effective_parameter));
                has_next_knot = knots_->next();
            }
        }
        return result;
    }

private:
    static scalar_type abs_value(const scalar_type& value)
    {
        return value < scalar_type(0) ? -value : value;
    }

    static bool same_parameter(
        const scalar_type& left,
        const scalar_type& right)
    {
        const scalar_type scale = std::max<scalar_type>(
            scalar_type(1),
            std::max<scalar_type>(abs_value(left), abs_value(right)));
        return abs_value(left - right) <=
            scalar_type(64)*std::numeric_limits<scalar_type>::epsilon()*scale;
    }

    void log_incomplete_intersections(
        const char* action,
        const scalar_type& requested_parameter,
        const scalar_type& effective_parameter,
        const IntersectionStatus& status) const
    {
        log_->warning_f(
            "MAIN:deflation_continuation: %s at requested lambda = %lf, effective lambda = %lf because restart intersections are incomplete: added = %u, failed = %u, missing_data = %u, skipped_discontinuous = %u, skipped_incomplete = %u.",
            action,
            double(requested_parameter),
            double(effective_parameter),
            status.added,
            status.failed,
            status.missing_data,
            status.skipped_discontinuous,
            status.skipped_incomplete);
    }

    VectorOperations* vector_operations_;
    Knots* knots_;
    Log* log_;
    knot_execution_policy<scalar_type> policy_;
    callbacks_type callbacks_;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_KNOT_EXECUTOR_H__
