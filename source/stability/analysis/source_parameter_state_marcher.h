#ifndef __STABILITY_ANALYSIS_SOURCE_PARAMETER_STATE_MARCHER_H__
#define __STABILITY_ANALYSIS_SOURCE_PARAMETER_STATE_MARCHER_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "detail/vector_workspace.h"
#include "source_parameter_path.h"
#include "transition_state_alignment.h"

namespace stability
{
namespace analysis
{

template<class Real>
struct source_parameter_state_march_result
{
    bool succeeded = false;
    std::size_t completed_steps = 0;
    std::size_t failed_position = 0;
    std::uint64_t failed_source_index = 0;
    Real failed_parameter = Real{};
    unsigned int fallback_recoveries = 0;
    bool terminal_predecessor_available = false;
    bool last_completed_state_available = false;
    std::size_t last_completed_position = 0;
    std::uint64_t last_completed_source_index = 0;
    Real last_completed_parameter = Real{};
    bool bidirectional_recovery_attempted = false;
    bool used_bidirectional_recovery = false;
    bool topology_split_detected = false;
    std::size_t topology_split_left_position = 0;
    std::uint64_t topology_split_left_source_index = 0;
    Real topology_split_left_parameter = Real{};
    std::size_t topology_split_right_position = 0;
    std::uint64_t topology_split_right_source_index = 0;
    Real topology_split_right_parameter = Real{};
    std::size_t bidirectional_join_position = 0;
    std::uint64_t bidirectional_join_source_index = 0;
    Real bidirectional_join_relative_distance = Real{};
    std::size_t reverse_completed_steps = 0;
    std::string diagnostic;
};

template<class VectorOperations>
class source_parameter_state_marcher
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using sample_type = source_parameter_sample<scalar_type>;
    using result_type = source_parameter_state_march_result<scalar_type>;

    struct correction_result
    {
        bool succeeded = false;
        bool used_fallback = false;
        std::string diagnostic;
    };

    using correction_function = std::function<correction_result(
        vector_type&,
        scalar_type)>;

    explicit source_parameter_state_marcher(
        VectorOperations* vector_operations)
        : vector_operations_(vector_operations),
          alignment_(vector_operations)
    {
    }

    template<class Aligner>
    void set_state_aligner(Aligner* aligner)
    {
        alignment_.set(aligner);
    }

    void reset_state_aligner()
    {
        alignment_.reset();
    }

    result_type march(
        const vector_type& anchor_state,
        const std::vector<sample_type>& samples,
        std::size_t anchor_position,
        std::size_t target_position,
        const correction_function& correction,
        vector_type& destination,
        vector_type* terminal_predecessor = nullptr,
        vector_type* last_completed_state = nullptr)
    {
        result_type result;
        if(samples.empty())
        {
            result.diagnostic = "state march requires path samples";
            return result;
        }
        if(
            anchor_position >= samples.size() ||
            target_position >= samples.size())
        {
            result.diagnostic = "state march position is out of range";
            return result;
        }
        if(!correction)
        {
            result.diagnostic = "state march correction is not configured";
            return result;
        }

        detail::vector_workspace<VectorOperations> previous(
            vector_operations_);
        detail::vector_workspace<VectorOperations> current(
            vector_operations_);
        detail::vector_workspace<VectorOperations> trial(
            vector_operations_);
        vector_operations_->assign(anchor_state, current.get());
        result.last_completed_position = anchor_position;
        result.last_completed_source_index =
            samples[anchor_position].source_index;
        result.last_completed_parameter =
            samples[anchor_position].parameter;
        result.last_completed_state_available = true;
        if(anchor_position == target_position)
        {
            vector_operations_->assign(current.get(), destination);
            if(last_completed_state != nullptr)
            {
                vector_operations_->assign(
                    current.get(),
                    *last_completed_state);
            }
            result.succeeded = true;
            result.diagnostic = "state march anchor is the target";
            return result;
        }

        const int direction =
            target_position > anchor_position ? 1 : -1;
        std::size_t position = anchor_position;
        while(position != target_position)
        {
            position = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(position) + direction);
            vector_operations_->assign(current.get(), trial.get());
            const correction_result corrected = correction(
                trial.get(),
                samples[position].parameter);
            if(!corrected.succeeded)
            {
                result.failed_position = position;
                result.failed_source_index =
                    samples[position].source_index;
                result.failed_parameter = samples[position].parameter;
                std::ostringstream message;
                message
                    << "state march failed at path position "
                    << position
                    << ", source point "
                    << samples[position].source_index
                    << ", parameter "
                    << samples[position].parameter;
                if(!corrected.diagnostic.empty())
                    message << ": " << corrected.diagnostic;
                result.diagnostic = message.str();
                if(last_completed_state != nullptr)
                {
                    vector_operations_->assign(
                        current.get(),
                        *last_completed_state);
                }
                return result;
            }
            if(corrected.used_fallback)
                ++result.fallback_recoveries;
            alignment_.align(
                current.get(),
                trial.get(),
                trial.get());
            vector_operations_->assign(
                current.get(),
                previous.get());
            vector_operations_->assign(trial.get(), current.get());
            ++result.completed_steps;
            result.last_completed_position = position;
            result.last_completed_source_index =
                samples[position].source_index;
            result.last_completed_parameter =
                samples[position].parameter;
        }

        vector_operations_->assign(current.get(), destination);
        if(terminal_predecessor != nullptr)
        {
            vector_operations_->assign(
                previous.get(),
                *terminal_predecessor);
            result.terminal_predecessor_available = true;
        }
        if(last_completed_state != nullptr)
        {
            vector_operations_->assign(
                current.get(),
                *last_completed_state);
        }
        result.succeeded = true;
        std::ostringstream message;
        message
            << "state march completed "
            << result.completed_steps
            << " source step(s)";
        if(result.fallback_recoveries != 0)
        {
            message
                << " with "
                << result.fallback_recoveries
                << " fallback recovery/recoveries";
        }
        result.diagnostic = message.str();
        return result;
    }

    result_type advance_with_secant_predictor(
        const vector_type& previous_state,
        const vector_type& current_state,
        const sample_type& target_sample,
        const correction_function& correction,
        vector_type& destination)
    {
        result_type result;
        if(!correction)
        {
            result.diagnostic =
                "secant advance correction is not configured";
            return result;
        }

        detail::vector_workspace<VectorOperations> trial(
            vector_operations_);
        vector_operations_->assign_mul(
            scalar_type(2),
            current_state,
            scalar_type(-1),
            previous_state,
            trial.get());
        const correction_result corrected = correction(
            trial.get(),
            target_sample.parameter);
        if(!corrected.succeeded)
        {
            result.failed_source_index = target_sample.source_index;
            result.failed_parameter = target_sample.parameter;
            result.diagnostic =
                "secant-predicted source-path advance failed at source "
                "point " + std::to_string(target_sample.source_index);
            if(!corrected.diagnostic.empty())
                result.diagnostic += ": " + corrected.diagnostic;
            return result;
        }
        alignment_.align(
            current_state,
            trial.get(),
            trial.get());
        vector_operations_->assign(trial.get(), destination);
        result.succeeded = true;
        result.completed_steps = 1;
        result.fallback_recoveries = corrected.used_fallback ? 1u : 0u;
        result.diagnostic =
            "secant-predicted source-path advance succeeded";
        return result;
    }

    template<class TurningCallback>
    result_type march_through_turning_points(
        const vector_type& anchor_state,
        const std::vector<sample_type>& samples,
        std::size_t anchor_position,
        std::size_t target_position,
        const std::vector<std::size_t>& turning_positions,
        const correction_function& correction,
        TurningCallback&& on_turning_point,
        vector_type& destination,
        const vector_type* target_anchor_state = nullptr,
        scalar_type relative_matching_tolerance = scalar_type(0),
        vector_type* topology_left_state = nullptr,
        vector_type* topology_right_state = nullptr)
    {
        result_type result;
        if(samples.empty())
        {
            result.diagnostic =
                "turning-path march requires path samples";
            return result;
        }
        if(
            anchor_position >= samples.size() ||
            target_position >= samples.size() ||
            anchor_position >= target_position)
        {
            result.diagnostic =
                "turning-path march requires an increasing valid "
                "source interval";
            return result;
        }
        if(!correction)
        {
            result.diagnostic =
                "turning-path march correction is not configured";
            return result;
        }
        if(
            target_anchor_state != nullptr &&
            relative_matching_tolerance <= scalar_type(0))
        {
            result.diagnostic =
                "bidirectional turning-path recovery requires a positive "
                "matching tolerance";
            return result;
        }
        if(
            (topology_left_state == nullptr) !=
            (topology_right_state == nullptr))
        {
            result.diagnostic =
                "turning-path topology recovery requires both state "
                "outputs or neither";
            return result;
        }

        detail::vector_workspace<VectorOperations> current(
            vector_operations_);
        detail::vector_workspace<VectorOperations> approach(
            vector_operations_);
        detail::vector_workspace<VectorOperations> departure(
            vector_operations_);
        detail::vector_workspace<VectorOperations> predecessor(
            vector_operations_);
        detail::vector_workspace<VectorOperations> forward_terminal(
            vector_operations_);
        vector_operations_->assign(anchor_state, current.get());
        std::size_t current_position = anchor_position;
        std::size_t previous_turning_position = 0;
        bool first_turn = true;

        for(
            std::size_t turning_index = 0;
            turning_index < turning_positions.size();
            ++turning_index)
        {
            const std::size_t turning_position =
                turning_positions[turning_index];
            if(
                (!first_turn &&
                    turning_position <= previous_turning_position) ||
                turning_position <= anchor_position ||
                turning_position >= target_position)
            {
                result.diagnostic =
                    "turning-path positions must increase strictly "
                    "inside the source interval";
                return result;
            }
            first_turn = false;
            previous_turning_position = turning_position;

            const auto turning =
                estimate_source_parameter_turning_point(
                    samples,
                    turning_position);
            if(!turning.valid)
            {
                result.diagnostic =
                    "turning-path estimate failed at turning point " +
                    std::to_string(turning_index) + ": " +
                    turning.diagnostic;
                return result;
            }
            if(
                turning.left_position < current_position ||
                turning.right_position <= turning.left_position ||
                turning.right_position > target_position)
            {
                result.diagnostic =
                    "turning-path guard positions are inconsistent "
                    "with source order";
                return result;
            }

            const result_type approach_result = march(
                current.get(),
                samples,
                current_position,
                turning.left_position,
                correction,
                approach.get(),
                &predecessor.get());
            result.completed_steps +=
                approach_result.completed_steps;
            result.fallback_recoveries +=
                approach_result.fallback_recoveries;
            if(!approach_result.succeeded)
            {
                result.failed_position =
                    approach_result.failed_position;
                result.failed_source_index =
                    approach_result.failed_source_index;
                result.failed_parameter =
                    approach_result.failed_parameter;
                result.diagnostic =
                    "turning-path approach failed at turning point " +
                    std::to_string(turning_index) + ": " +
                    approach_result.diagnostic;
                return result;
            }
            if(!approach_result.terminal_predecessor_available)
            {
                result.diagnostic =
                    "turning-path secant crossing has no predecessor "
                    "at turning point " +
                    std::to_string(turning_index);
                return result;
            }

            const result_type crossing_result =
                advance_with_secant_predictor(
                    predecessor.get(),
                    approach.get(),
                    samples[turning.right_position],
                    correction,
                    departure.get());
            result.completed_steps +=
                crossing_result.completed_steps;
            result.fallback_recoveries +=
                crossing_result.fallback_recoveries;
            if(!crossing_result.succeeded)
            {
                result.failed_position = turning.right_position;
                result.failed_source_index =
                    crossing_result.failed_source_index;
                result.failed_parameter =
                    crossing_result.failed_parameter;
                result.diagnostic =
                    "turning-path secant crossing failed at turning "
                    "point " + std::to_string(turning_index) + ": " +
                    crossing_result.diagnostic;
                return result;
            }

            on_turning_point(
                turning_index,
                turning,
                approach.get(),
                departure.get(),
                approach_result,
                crossing_result);
            vector_operations_->assign(
                departure.get(),
                current.get());
            current_position = turning.right_position;
        }

        const result_type terminal_result = march(
            current.get(),
            samples,
            current_position,
            target_position,
            correction,
            destination,
            nullptr,
            &forward_terminal.get());
        result.completed_steps += terminal_result.completed_steps;
        result.fallback_recoveries +=
            terminal_result.fallback_recoveries;
        if(!terminal_result.succeeded)
        {
            if(target_anchor_state != nullptr)
            {
                result.bidirectional_recovery_attempted = true;
                detail::vector_workspace<VectorOperations> reverse_join(
                    vector_operations_);
                detail::vector_workspace<VectorOperations> aligned_reverse(
                    vector_operations_);
                detail::vector_workspace<VectorOperations> difference(
                    vector_operations_);
                const result_type reverse_result = march(
                    *target_anchor_state,
                    samples,
                    target_position,
                    terminal_result.last_completed_position,
                    correction,
                    reverse_join.get());
                result.reverse_completed_steps =
                    reverse_result.completed_steps;
                result.completed_steps +=
                    reverse_result.completed_steps;
                result.fallback_recoveries +=
                    reverse_result.fallback_recoveries;
                if(reverse_result.succeeded)
                {
                    alignment_.align(
                        forward_terminal.get(),
                        reverse_join.get(),
                        aligned_reverse.get());
                    vector_operations_->assign_mul(
                        scalar_type(1),
                        forward_terminal.get(),
                        scalar_type(-1),
                        aligned_reverse.get(),
                        difference.get());
                    const scalar_type scale = scalar_type(1) + std::max(
                        vector_operations_->norm(forward_terminal.get()),
                        vector_operations_->norm(aligned_reverse.get()));
                    result.bidirectional_join_relative_distance =
                        vector_operations_->norm(difference.get())/scale;
                    result.bidirectional_join_position =
                        terminal_result.last_completed_position;
                    result.bidirectional_join_source_index =
                        samples[result.bidirectional_join_position].
                            source_index;
                    if(
                        result.bidirectional_join_relative_distance <=
                            relative_matching_tolerance)
                    {
                        vector_operations_->assign(
                            *target_anchor_state,
                            destination);
                        result.succeeded = true;
                        result.used_bidirectional_recovery = true;
                        result.last_completed_state_available = true;
                        result.last_completed_position = target_position;
                        result.last_completed_source_index =
                            samples[target_position].source_index;
                        result.last_completed_parameter =
                            samples[target_position].parameter;
                        std::ostringstream message;
                        message
                            << "turning-path terminal march joined a "
                            << "backward reconstruction at source point "
                            << result.bidirectional_join_source_index
                            << " with relative aligned distance "
                            << result.bidirectional_join_relative_distance
                            << " after "
                            << terminal_result.completed_steps
                            << " forward and "
                            << reverse_result.completed_steps
                            << " reverse source step(s)";
                        result.diagnostic = message.str();
                        return result;
                    }

                    if(
                        topology_left_state != nullptr &&
                        topology_right_state != nullptr)
                    {
                        vector_operations_->assign(
                            forward_terminal.get(),
                            *topology_left_state);
                        vector_operations_->assign(
                            aligned_reverse.get(),
                            *topology_right_state);
                        vector_operations_->assign(
                            *target_anchor_state,
                            destination);
                        result.succeeded = true;
                        result.topology_split_detected = true;
                        result.last_completed_state_available = true;
                        result.last_completed_position = target_position;
                        result.last_completed_source_index =
                            samples[target_position].source_index;
                        result.last_completed_parameter =
                            samples[target_position].parameter;
                        result.topology_split_right_position =
                            result.bidirectional_join_position;
                        result.topology_split_right_source_index =
                            result.bidirectional_join_source_index;
                        result.topology_split_right_parameter =
                            samples[result.bidirectional_join_position].
                                parameter;
                        result.topology_split_left_position =
                            result.bidirectional_join_position == 0
                            ? std::size_t(0)
                            : result.bidirectional_join_position - 1;
                        result.topology_split_left_source_index =
                            samples[result.topology_split_left_position].
                                source_index;
                        result.topology_split_left_parameter =
                            samples[result.topology_split_left_position].
                                parameter;
                        std::ostringstream message;
                        message
                            << "turning-path terminal march detected a "
                            << "source-path topology split across source "
                            << "points ["
                            << result.topology_split_left_source_index
                            << ','
                            << result.topology_split_right_source_index
                            << "]: forward and backward reconstructions "
                            << "at parameter "
                            << result.topology_split_right_parameter
                            << " have relative aligned distance "
                            << result.bidirectional_join_relative_distance
                            << ", tolerance = "
                            << relative_matching_tolerance;
                        result.diagnostic = message.str();
                        return result;
                    }
                }

                result.failed_position =
                    terminal_result.failed_position;
                result.failed_source_index =
                    terminal_result.failed_source_index;
                result.failed_parameter =
                    terminal_result.failed_parameter;
                std::ostringstream message;
                message
                    << "turning-path terminal march failed: "
                    << terminal_result.diagnostic
                    << "; bidirectional recovery ";
                if(!reverse_result.succeeded)
                {
                    message
                        << "failed before reaching the last forward source "
                        << "point: " << reverse_result.diagnostic;
                }
                else
                {
                    message
                        << "reached source point "
                        << result.bidirectional_join_source_index
                        << " on a different aligned state, relative "
                        << "distance = "
                        << result.bidirectional_join_relative_distance
                        << ", tolerance = "
                        << relative_matching_tolerance;
                }
                result.diagnostic = message.str();
                return result;
            }
            result.failed_position = terminal_result.failed_position;
            result.failed_source_index =
                terminal_result.failed_source_index;
            result.failed_parameter = terminal_result.failed_parameter;
            result.diagnostic =
                "turning-path terminal march failed: " +
                terminal_result.diagnostic;
            return result;
        }

        result.succeeded = true;
        result.last_completed_state_available = true;
        result.last_completed_position = target_position;
        result.last_completed_source_index =
            samples[target_position].source_index;
        result.last_completed_parameter =
            samples[target_position].parameter;
        std::ostringstream message;
        message
            << "turning-path march completed "
            << result.completed_steps
            << " source step(s) across "
            << turning_positions.size()
            << " turning point(s)";
        if(result.fallback_recoveries != 0)
        {
            message
                << " with "
                << result.fallback_recoveries
                << " fallback recovery/recoveries";
        }
        result.diagnostic = message.str();
        return result;
    }

private:
    VectorOperations* vector_operations_;
    transition_state_alignment<VectorOperations> alignment_;
};

} // namespace analysis
} // namespace stability

#endif // __STABILITY_ANALYSIS_SOURCE_PARAMETER_STATE_MARCHER_H__
