#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <stability/analysis/source_parameter_state_marcher.h>

namespace
{

struct vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void init_vector(vector_type& vector) const
    {
        vector.assign(1, 0.0);
    }

    void start_use_vector(vector_type&) const {}
    void stop_use_vector(vector_type&) const {}
    void free_vector(vector_type& vector) const { vector.clear(); }

    void assign(const vector_type& source, vector_type& destination) const
    {
        destination = source;
    }

    void assign_mul(
        double first_scale,
        const vector_type& first,
        double second_scale,
        const vector_type& second,
        vector_type& destination) const
    {
        destination[0] =
            first_scale*first[0] + second_scale*second[0];
    }

    double norm(const vector_type& vector) const
    {
        return std::abs(vector[0]);
    }
};

struct sign_aligner
{
    void stabilize_closest_to_reference(
        const std::vector<double>& reference,
        const std::vector<double>& source,
        std::vector<double>& destination)
    {
        destination = source;
        if(
            !reference.empty() &&
            !source.empty() &&
            reference[0]*source[0] < 0.0)
        {
            destination[0] = -destination[0];
        }
    }
};

int checks = 0;
int failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void require_close(double value, double expected, const std::string& message)
{
    require(std::abs(value - expected) <= 1.0e-14, message);
}

void test_forward_and_backward_march()
{
    vector_operations operations;
    using marcher_type =
        stability::analysis::source_parameter_state_marcher<
            vector_operations>;
    marcher_type marcher(&operations);
    const std::vector<marcher_type::sample_type> samples{
        {10, 1.0}, {11, 2.0}, {12, 3.0}, {13, 4.0}};
    const auto correction = [](
        std::vector<double>& state,
        double parameter)
    {
        state[0] = parameter*parameter;
        return marcher_type::correction_result{true, false, {}};
    };

    std::vector<double> destination{99.0};
    std::vector<double> predecessor{99.0};
    const auto forward = marcher.march(
        std::vector<double>{1.0},
        samples,
        0,
        3,
        correction,
        destination,
        &predecessor);
    require(forward.succeeded, "forward march succeeds");
    require(forward.completed_steps == 3, "forward step count");
    require_close(destination[0], 16.0, "forward destination");
    require(
        forward.terminal_predecessor_available,
        "forward predecessor is available");
    require_close(predecessor[0], 9.0, "forward predecessor");

    const auto backward = marcher.march(
        std::vector<double>{16.0},
        samples,
        3,
        1,
        correction,
        destination);
    require(backward.succeeded, "backward march succeeds");
    require(backward.completed_steps == 2, "backward step count");
    require_close(destination[0], 4.0, "backward destination");
}

void test_secant_predictor_crosses_fold()
{
    vector_operations operations;
    using marcher_type =
        stability::analysis::source_parameter_state_marcher<
            vector_operations>;
    marcher_type marcher(&operations);
    const auto signed_root_correction = [](
        std::vector<double>& state,
        double parameter)
    {
        const double sign = state[0] < 0.0 ? -1.0 : 1.0;
        state[0] = sign*std::sqrt(parameter);
        return marcher_type::correction_result{true, false, {}};
    };

    std::vector<double> crossed{77.0};
    const auto result = marcher.advance_with_secant_predictor(
        std::vector<double>{-1.0},
        std::vector<double>{-0.2},
        marcher_type::sample_type{3, 0.09},
        signed_root_correction,
        crossed);
    require(result.succeeded, "secant fold crossing succeeds");
    require_close(crossed[0], 0.3, "secant predictor selects other fold side");

    std::vector<double> copied{-0.2};
    signed_root_correction(copied, 0.09);
    require_close(copied[0], -0.3, "copy predictor stays on old fold side");

    std::vector<double> preserved{19.0};
    const auto failed = marcher.advance_with_secant_predictor(
        std::vector<double>{-1.0},
        std::vector<double>{-0.2},
        marcher_type::sample_type{3, 0.09},
        [](std::vector<double>& state, double)
        {
            state[0] = -500.0;
            return marcher_type::correction_result{
                false,
                false,
                "injected secant failure"};
        },
        preserved);
    require(!failed.succeeded, "secant correction failure is reported");
    require_close(
        preserved[0],
        19.0,
        "failed secant correction preserves destination");
}

void test_transactional_failure()
{
    vector_operations operations;
    using marcher_type =
        stability::analysis::source_parameter_state_marcher<
            vector_operations>;
    marcher_type marcher(&operations);
    const std::vector<marcher_type::sample_type> samples{
        {0, 1.0}, {1, 2.0}, {2, 3.0}};
    std::vector<double> destination{42.0};
    const auto result = marcher.march(
        std::vector<double>{1.0},
        samples,
        0,
        2,
        [](std::vector<double>& state, double parameter)
        {
            if(parameter == 3.0)
            {
                state[0] = -100.0;
                return marcher_type::correction_result{
                    false,
                    false,
                    "injected failure"};
            }
            state[0] = parameter;
            return marcher_type::correction_result{true, true, {}};
        },
        destination);
    require(!result.succeeded, "injected failure is reported");
    require(result.completed_steps == 1, "completed steps before failure");
    require(result.failed_source_index == 2, "failed source index");
    require(result.fallback_recoveries == 1, "fallback recovery count");
    require_close(destination[0], 42.0, "failed march preserves destination");
}

void test_alignment()
{
    vector_operations operations;
    using marcher_type =
        stability::analysis::source_parameter_state_marcher<
            vector_operations>;
    marcher_type marcher(&operations);
    sign_aligner aligner;
    marcher.set_state_aligner(&aligner);
    const std::vector<marcher_type::sample_type> samples{
        {0, 1.0}, {1, 2.0}, {2, 3.0}};
    std::vector<double> destination{0.0};
    const auto result = marcher.march(
        std::vector<double>{1.0},
        samples,
        0,
        2,
        [](std::vector<double>& state, double parameter)
        {
            state[0] = -parameter;
            return marcher_type::correction_result{true, false, {}};
        },
        destination);
    require(result.succeeded, "aligned march succeeds");
    require_close(destination[0], 3.0, "alignment preserves local representative");
}

void test_multiple_turn_march()
{
    vector_operations operations;
    using marcher_type =
        stability::analysis::source_parameter_state_marcher<
            vector_operations>;
    marcher_type marcher(&operations);
    const std::vector<double> exact_states{
        -2.0, -1.75, -1.5, -1.25, -1.1, -0.9, -0.5, 0.0,
        0.5, 0.9, 1.1, 1.25, 1.5, 1.75, 2.0};
    std::vector<marcher_type::sample_type> samples;
    for(std::size_t index = 0; index < exact_states.size(); ++index)
    {
        const double state = exact_states[index];
        samples.push_back({
            static_cast<std::uint64_t>(100 + index),
            state*state*state - 3.0*state});
    }
    const auto path =
        stability::analysis::analyze_source_parameter_path(samples);
    require(path.valid, "cubic source path is valid");
    require(path.turning_positions.size() == 2, "cubic path has two turns");

    const auto cubic_correction = [](
        std::vector<double>& state,
        double parameter)
    {
        double value = state[0];
        for(unsigned int iteration = 0; iteration < 60; ++iteration)
        {
            const double residual =
                value*value*value - 3.0*value - parameter;
            if(std::abs(residual) <= 1.0e-13)
            {
                state[0] = value;
                return marcher_type::correction_result{true, false, {}};
            }
            const double derivative = 3.0*value*value - 3.0;
            if(std::abs(derivative) <= 1.0e-10)
            {
                return marcher_type::correction_result{
                    false,
                    false,
                    "singular cubic derivative"};
            }
            const double step = std::max(
                -0.35,
                std::min(0.35, residual/derivative));
            value -= step;
        }
        return marcher_type::correction_result{
            false,
            false,
            "cubic Newton iteration limit"};
    };

    std::vector<double> destination{77.0};
    std::size_t observed_turns = 0;
    const auto result = marcher.march_through_turning_points(
        std::vector<double>{exact_states.front()},
        samples,
        0,
        samples.size() - 1,
        path.turning_positions,
        cubic_correction,
        [&observed_turns](
            std::size_t turning_index,
            const auto&,
            const std::vector<double>& left_state,
            const std::vector<double>& right_state,
            const auto&,
            const auto&)
        {
            ++observed_turns;
            if(turning_index == 0)
            {
                require(left_state[0] < -1.0, "maximum left guard");
                require(right_state[0] > -1.0, "maximum right guard");
            }
            else
            {
                require(left_state[0] < 1.0, "minimum left guard");
                require(right_state[0] > 1.0, "minimum right guard");
            }
        },
        destination);
    require(result.succeeded, "multiple-turn state march succeeds");
    require(observed_turns == 2, "multiple-turn callback count");
    require_close(destination[0], 2.0, "multiple-turn terminal state");

    std::vector<double> preserved{91.0};
    std::size_t callbacks_before_failure = 0;
    const double rejected_parameter = samples[10].parameter;
    const auto failed = marcher.march_through_turning_points(
        std::vector<double>{exact_states.front()},
        samples,
        0,
        samples.size() - 1,
        path.turning_positions,
        [cubic_correction, rejected_parameter](
            std::vector<double>& state,
            double parameter)
        {
            if(
                state[0] > 0.0 &&
                std::abs(parameter - rejected_parameter) < 1.0e-14)
            {
                return marcher_type::correction_result{
                    false,
                    false,
                    "injected second-turn failure"};
            }
            return cubic_correction(state, parameter);
        },
        [&callbacks_before_failure](
            std::size_t,
            const auto&,
            const std::vector<double>&,
            const std::vector<double>&,
            const auto&,
            const auto&)
        {
            ++callbacks_before_failure;
        },
        preserved);
    require(!failed.succeeded, "multiple-turn failure is reported");
    require(
        callbacks_before_failure == 1,
        "only completed turns invoke the callback");
    require_close(
        preserved[0],
        91.0,
        "failed multiple-turn march preserves destination");
}

void test_bidirectional_terminal_recovery()
{
    vector_operations operations;
    using marcher_type =
        stability::analysis::source_parameter_state_marcher<
            vector_operations>;
    marcher_type marcher(&operations);
    const std::vector<marcher_type::sample_type> samples{
        {10, 1.0}, {11, 2.0}, {12, 3.0}, {13, 4.0}, {14, 5.0}};
    const auto correction = [](
        std::vector<double>& state,
        double parameter)
    {
        if(parameter == 3.0 && state[0] < 3.0)
        {
            return marcher_type::correction_result{
                false,
                false,
                "injected one-sided singularity"};
        }
        state[0] = parameter;
        return marcher_type::correction_result{true, false, {}};
    };
    const auto no_turn = [](
        std::size_t,
        const auto&,
        const auto&,
        const auto&,
        const auto&,
        const auto&) {};

    const std::vector<double> upper_anchor{5.0};
    std::vector<double> destination{77.0};
    const auto recovered = marcher.march_through_turning_points(
        std::vector<double>{1.0},
        samples,
        0,
        samples.size() - 1,
        std::vector<std::size_t>{},
        correction,
        no_turn,
        destination,
        &upper_anchor,
        1.0e-13);
    require(recovered.succeeded, "bidirectional terminal recovery succeeds");
    require(
        recovered.bidirectional_recovery_attempted,
        "bidirectional terminal recovery is attempted");
    require(
        recovered.used_bidirectional_recovery,
        "bidirectional terminal recovery is selected");
    require(
        recovered.bidirectional_join_source_index == 11,
        "bidirectional recovery joins at last forward source");
    require(
        recovered.reverse_completed_steps == 3,
        "bidirectional recovery reverse step count");
    require_close(
        recovered.bidirectional_join_relative_distance,
        0.0,
        "bidirectional recovery join distance");
    require_close(destination[0], 5.0, "bidirectional recovery destination");

    const auto mismatched_correction = [](
            std::vector<double>& state,
            double parameter)
        {
            if(parameter == 3.0 && state[0] < 3.0)
            {
                return marcher_type::correction_result{
                    false,
                    false,
                    "injected one-sided singularity"};
            }
            if(parameter == 2.0 && state[0] > 2.5)
                state[0] = 20.0;
            else
                state[0] = parameter;
            return marcher_type::correction_result{true, false, {}};
        };
    std::vector<double> preserved{91.0};
    const auto mismatched = marcher.march_through_turning_points(
        std::vector<double>{1.0},
        samples,
        0,
        samples.size() - 1,
        std::vector<std::size_t>{},
        mismatched_correction,
        no_turn,
        preserved,
        &upper_anchor,
        1.0e-13);
    require(!mismatched.succeeded, "different bidirectional branches fail");
    require(
        mismatched.bidirectional_recovery_attempted,
        "different branches retain recovery diagnostic");
    require(
        !mismatched.used_bidirectional_recovery,
        "different branches are not joined");
    require_close(
        preserved[0],
        91.0,
        "different bidirectional branches preserve destination");

    std::vector<double> split_destination{91.0};
    std::vector<double> split_left{0.0};
    std::vector<double> split_right{0.0};
    const auto split = marcher.march_through_turning_points(
        std::vector<double>{1.0},
        samples,
        0,
        samples.size() - 1,
        std::vector<std::size_t>{},
        mismatched_correction,
        no_turn,
        split_destination,
        &upper_anchor,
        1.0e-13,
        &split_left,
        &split_right);
    require(
        split.succeeded && split.topology_split_detected,
        "different branches can be preserved as a topology split");
    require(
        !split.used_bidirectional_recovery &&
            split.topology_split_left_source_index == 10 &&
            split.topology_split_right_source_index == 11,
        "topology split records the discontinuous source edge");
    require_close(split_left[0], 2.0, "topology split left state");
    require_close(split_right[0], 20.0, "topology split right state");
    require_close(
        split_destination[0],
        5.0,
        "topology split preserves the saved target anchor");
}

void test_bidirectional_recovery_alignment()
{
    vector_operations operations;
    using marcher_type =
        stability::analysis::source_parameter_state_marcher<
            vector_operations>;
    marcher_type marcher(&operations);
    sign_aligner aligner;
    marcher.set_state_aligner(&aligner);
    const std::vector<marcher_type::sample_type> samples{
        {20, 1.0}, {21, 2.0}, {22, 3.0}, {23, 4.0}, {24, 5.0}};
    const auto correction = [](
        std::vector<double>& state,
        double parameter)
    {
        const double sign = state[0] < 0.0 ? -1.0 : 1.0;
        if(parameter == 3.0 && sign > 0.0)
        {
            return marcher_type::correction_result{
                false,
                false,
                "injected positive-representative singularity"};
        }
        state[0] = sign*parameter;
        return marcher_type::correction_result{true, false, {}};
    };
    const auto no_turn = [](
        std::size_t,
        const auto&,
        const auto&,
        const auto&,
        const auto&,
        const auto&) {};

    const std::vector<double> upper_anchor{-5.0};
    std::vector<double> destination{0.0};
    const auto recovered = marcher.march_through_turning_points(
        std::vector<double>{1.0},
        samples,
        0,
        samples.size() - 1,
        std::vector<std::size_t>{},
        correction,
        no_turn,
        destination,
        &upper_anchor,
        1.0e-13);
    require(
        recovered.succeeded && recovered.used_bidirectional_recovery,
        "aligned quotient representatives join bidirectionally");
    require_close(
        recovered.bidirectional_join_relative_distance,
        0.0,
        "aligned quotient join distance");
    require_close(
        destination[0],
        -5.0,
        "aligned recovery preserves saved upper representative");
}

} // namespace

int main()
{
    test_forward_and_backward_march();
    test_transactional_failure();
    test_alignment();
    test_secant_predictor_crosses_fold();
    test_multiple_turn_march();
    test_bidirectional_terminal_recovery();
    test_bidirectional_recovery_alignment();
    std::cout << "Checks: " << checks << ", failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
