#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <stability/analysis/source_parameter_path.h>

namespace
{

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

using sample = stability::analysis::source_parameter_sample<double>;

void test_monotone_paths()
{
    const auto increasing =
        stability::analysis::analyze_source_parameter_path<double>(
            {{10, 1.0}, {11, 1.5}, {12, 2.0}});
    require(increasing.valid, "increasing path is valid");
    require(increasing.monotone, "increasing path is monotone");
    require(increasing.monotone_spans.size() == 1, "one increasing span");
    require(increasing.monotone_spans[0].direction == 1, "increasing direction");

    const auto decreasing =
        stability::analysis::analyze_source_parameter_path<double>(
            {{20, 3.0}, {21, 2.0}, {22, 1.0}});
    require(decreasing.valid, "decreasing path is valid");
    require(decreasing.monotone, "decreasing path is monotone");
    require(decreasing.monotone_spans[0].direction == -1, "decreasing direction");
}

void test_single_fold()
{
    const std::vector<sample> values{
        {50, 17.90555708252424},
        {60, 17.76357569176771},
        {82, 17.59781476644556},
        {83, 17.59742706637848},
        {84, 17.59756950409960},
        {100, 17.65050011673659}};
    const auto result =
        stability::analysis::analyze_source_parameter_path(values);
    require(result.valid, "folded path is valid");
    require(!result.monotone, "folded path is non-monotone");
    require(result.turning_positions.size() == 1, "one turning point");
    require(result.turning_positions[0] == 3, "turning position is the minimum");
    require(result.monotone_spans.size() == 2, "fold splits into two spans");
    require(result.monotone_spans[0].direction == -1, "left fold direction");
    require(result.monotone_spans[1].direction == 1, "right fold direction");

    const auto turning =
        stability::analysis::estimate_source_parameter_turning_point(
            values,
            result.turning_positions[0]);
    require(turning.valid, "fold turning estimate is valid");
    require(turning.left_position == 3, "fold left guard");
    require(turning.right_position == 4, "fold right guard");
    require(
        std::abs(turning.right_fraction - 0.23131943375747557) <
            1.0e-10,
        "fold source interpolation fraction");
    require(
        std::abs(turning.parameter - 17.59741288289173) < 1.0e-12,
        "fold parameter estimate");
}

void test_plateau_and_noise()
{
    const double epsilon = 1.0e-15;
    const auto result =
        stability::analysis::analyze_source_parameter_path<double>(
            {{0, 2.0}, {1, 2.0 + epsilon}, {2, 1.0}, {3, 1.0}, {4, 1.5}},
            1.0e-12);
    require(result.valid, "plateau path is valid");
    require(!result.monotone, "plateau path retains true reversal");
    require(result.turning_positions.size() == 1, "plateau has one reversal");
    require(result.turning_positions[0] == 3, "plateau reversal uses last flat point");
}

void test_maximum_with_left_turning_offset()
{
    const std::vector<sample> values{
        {20, 1.4375}, {21, 1.9375}, {22, 0.4375}};
    const auto path =
        stability::analysis::analyze_source_parameter_path(values);
    require(path.valid, "maximum path is valid");
    require(path.turning_positions.size() == 1, "maximum has one turn");
    const auto turning =
        stability::analysis::estimate_source_parameter_turning_point(
            values,
            path.turning_positions[0]);
    require(turning.valid, "maximum turning estimate is valid");
    require(turning.left_position == 0, "maximum left guard");
    require(turning.right_position == 1, "maximum right guard");
    require(
        std::abs(turning.right_fraction - 0.75) < 1.0e-14,
        "maximum interpolation fraction");
    require(
        std::abs(turning.parameter - 2.0) < 1.0e-14,
        "maximum parameter estimate");
}

void test_multiple_turns()
{
    const auto result =
        stability::analysis::analyze_source_parameter_path<double>({
            {100, 0.0},
            {101, 1.0},
            {102, 2.0},
            {103, 1.0},
            {104, 0.0},
            {105, -1.0},
            {106, -2.0},
            {107, -1.0},
            {108, 0.0}});
    require(result.valid, "multiple-turn path is valid");
    require(!result.monotone, "multiple-turn path is non-monotone");
    require(result.turning_positions.size() == 2, "two turning points");
    require(result.turning_positions[0] == 2, "maximum position");
    require(result.turning_positions[1] == 6, "minimum position");
    require(result.monotone_spans.size() == 3, "three monotone spans");
    require(result.monotone_spans[0].direction == 1, "first span direction");
    require(result.monotone_spans[1].direction == -1, "middle span direction");
    require(result.monotone_spans[2].direction == 1, "last span direction");
}

void test_terminal_parameter_discontinuity()
{
    const auto curve14 =
        stability::analysis::analyze_source_parameter_terminal_step<double>({
            {100, 16.18665853687033},
            {120, 16.04104272271114},
            {130, 16.00805937005701},
            {137, 16.00012351257957},
            {138, 16.00000001929521},
            {139, 30.0}});
    require(curve14.valid, "curve-14 terminal step is analyzable");
    require(
        curve14.discontinuous,
        "curve-14 analytical-branch boundary jump is discontinuous");
    require(
        curve14.preceding_direction == -1 &&
            curve14.terminal_direction == 1,
        "curve-14 terminal step reverses parameter direction");
    require(
        curve14.step_ratio > 90.0,
        "curve-14 terminal jump dominates preceding source steps");

    const auto ordinary_fold =
        stability::analysis::analyze_source_parameter_terminal_step<double>({
            {0, 16.2}, {1, 16.1}, {2, 16.0}, {3, 16.05}});
    require(ordinary_fold.valid, "ordinary fold terminal step is analyzable");
    require(
        !ordinary_fold.discontinuous,
        "ordinary local fold is not a terminal discontinuity");

    const auto monotone_boundary =
        stability::analysis::analyze_source_parameter_terminal_step<double>({
            {0, 1.0}, {1, 1.1}, {2, 4.0}});
    require(
        monotone_boundary.valid && !monotone_boundary.discontinuous,
        "large monotone boundary step is not a reversal discontinuity");
}

void test_turning_guard_join_policy()
{
    using status =
        stability::analysis::source_path_turning_join_status;
    const double tolerance = 1.0e-6;

    const auto forward_join =
        stability::analysis::decide_source_path_turning_join(
            433,
            434,
            2.0e-3,
            5.0e-7,
            3.0e-2,
            tolerance,
            false);
    require(
        forward_join.accepted() &&
            forward_join.status == status::joined &&
            forward_join.forward_matches &&
            !forward_join.reverse_matches,
        "one matching crossing joins turning guards");

    const auto rejected =
        stability::analysis::decide_source_path_turning_join(
            433,
            434,
            2.0e-3,
            2.0e-2,
            3.0e-2,
            tolerance,
            false);
    require(
        !rejected.accepted() && rejected.status == status::rejected,
        "non-joining guards remain a hard failure by default");

    const auto topology =
        stability::analysis::decide_source_path_turning_join(
            433,
            434,
            2.0e-3,
            2.0e-2,
            3.0e-2,
            tolerance,
            true);
    require(
        topology.accepted() &&
            topology.status == status::topology_split &&
            !topology.forward_matches &&
            !topology.reverse_matches,
        "opt-in policy preserves non-joining guards as topology");
    require(
        topology.diagnostic.find("[433,434]") != std::string::npos &&
            topology.diagnostic.find("topology split") !=
                std::string::npos,
        "topology decision retains source bracket diagnostics");
}

void test_invalid_paths()
{
    const auto too_short =
        stability::analysis::analyze_source_parameter_path<double>({{0, 1.0}});
    require(!too_short.valid, "single sample is rejected");

    const auto repeated_index =
        stability::analysis::analyze_source_parameter_path<double>(
            {{1, 1.0}, {1, 2.0}});
    require(!repeated_index.valid, "repeated source index is rejected");

    const auto non_finite =
        stability::analysis::analyze_source_parameter_path<double>(
            {{0, 1.0}, {1, std::numeric_limits<double>::infinity()}});
    require(!non_finite.valid, "non-finite parameter is rejected");

    const auto boundary_turn =
        stability::analysis::estimate_source_parameter_turning_point<double>(
            {{0, 1.0}, {1, 2.0}},
            0);
    require(!boundary_turn.valid, "boundary turning estimate is rejected");
}

} // namespace

int main()
{
    test_monotone_paths();
    test_single_fold();
    test_plateau_and_noise();
    test_maximum_with_left_turning_offset();
    test_multiple_turns();
    test_terminal_parameter_discontinuity();
    test_turning_guard_join_policy();
    test_invalid_paths();
    std::cout << "Checks: " << checks << ", failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
