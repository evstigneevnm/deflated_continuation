#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <containers/knots.hpp>
#include <main/deflation_continuation/knot_executor.h>

namespace
{

struct fake_vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void init_vector(vector_type&) {}
    void start_use_vector(vector_type&) { ++active_vectors; }
    void stop_use_vector(vector_type&) { --active_vectors; }
    void free_vector(vector_type& value) { value.clear(); }
    void assign(const vector_type& source, vector_type& destination) { destination = source; }
    void assign_mul(
        double left_scale,
        const vector_type& left,
        double right_scale,
        const vector_type& right,
        vector_type& output)
    {
        output.resize(left.size());
        for(std::size_t index = 0; index < left.size(); ++index)
        {
            output[index] = left_scale*left[index] + right_scale*right[index];
        }
    }
    double norm_l2(const vector_type& value)
    {
        double sum = 0.0;
        for(const double entry: value)
        {
            sum += entry*entry;
        }
        return std::sqrt(sum);
    }

    int active_vectors = 0;
};

struct fake_knots
{
    explicit fake_knots(std::vector<double> values_): values(std::move(values_)) {}
    bool next()
    {
        ++index;
        return index < static_cast<int>(values.size());
    }
    double get_value() const { return values[static_cast<std::size_t>(index)]; }

    std::vector<double> values;
    int index = -1;
};

struct fake_log
{
    template<class... Args> void warning_f(const char*, Args&&...) {}
    template<class... Args> void info_f(const char*, Args&&...) {}
};

struct intersection_status
{
    unsigned int added = 0;
    unsigned int failed = 0;
    unsigned int missing_data = 0;
    unsigned int skipped_discontinuous = 0;
    unsigned int skipped_incomplete = 0;
    bool ok() const { return failed == 0 && missing_data == 0 && skipped_incomplete == 0; }
};

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

continuation::continuation_curve_result<double> complete_continuation()
{
    continuation::continuation_curve_result<double> result;
    result.semicurves_started = 2;
    for(auto& semicurve: result.semicurves)
    {
        semicurve.status = continuation::semicurve_status::complete;
        semicurve.accepted_points = 2;
    }
    return result;
}

continuation::continuation_curve_result<double> failed_without_progress()
{
    continuation::continuation_curve_result<double> result;
    result.semicurves_started = 1;
    result.semicurves[0].status =
        continuation::semicurve_status::open_recoverable;
    result.semicurves[0].failure =
        continuation::continuation_failure_kind::minimum_step;
    result.semicurves[0].accepted_points = 1;
    return result;
}

continuation::continuation_curve_result<double> failed_after_progress()
{
    auto result = failed_without_progress();
    auto& semicurve = result.semicurves[0];
    semicurve.accepted_points = 7;
    semicurve.segment_id = 3;
    semicurve.first_point_index = 0;
    semicurve.last_point_index = 6;
    semicurve.start_parameter = 8.0;
    semicurve.last_parameter = 8.75;
    return result;
}

}

int main()
{
    using callbacks_type =
        main_classes::deflation_continuation_detail::knot_executor_callbacks<
            double,
            std::vector<double>,
            intersection_status>;
    using executor_type =
        main_classes::deflation_continuation_detail::knot_executor<
            fake_vector_operations,
            fake_knots,
            fake_log,
            intersection_status>;

    try
    {
        container::knots<double> empty_knots;
        require(!empty_knots.next(), "empty knot schedule");
        empty_knots.add_element(std::vector<double>{1.0});
        require(!empty_knots.next(), "single-boundary knot schedule");

        container::knots<double> boundary_knots;
        boundary_knots.add_element(std::vector<double>{1.0, 80.0});
        require(!boundary_knots.next(), "boundary-only knot schedule");

        fake_vector_operations vector_operations;
        fake_knots knots({4.0, 6.0});
        fake_log log;
        callbacks_type callbacks;
        int analytical_calls = 0;
        int deflation_calls_at_four = 0;
        int accepted = 0;
        callbacks.load_archive = []() { return false; };
        callbacks.build_analytical_branches = [&analytical_calls](bool) { ++analytical_calls; };
        callbacks.restore_output_settings = []() {};
        callbacks.current_curve_count = []() { return 0; };
        callbacks.resolve_parameter = [](const double& requested, double& effective)
        {
            effective = requested;
            return false;
        };
        callbacks.relocation_registry_file = []() { return std::string{}; };
        callbacks.rebuild_intersections = [](const double& parameter)
        {
            intersection_status status;
            if(parameter == 6.0)
            {
                status.failed = 1;
            }
            return status;
        };
        callbacks.relocate_intersection = [](const double&, const intersection_status&, double&, intersection_status&)
        {
            return false;
        };
        callbacks.find_deflated_solution = [&deflation_calls_at_four](const double& parameter)
        {
            if(parameter != 4.0)
            {
                return false;
            }
            return deflation_calls_at_four++ == 0;
        };
        callbacks.get_deflated_solution = [](std::vector<double>& output) { output = {1.0, 2.0}; };
        callbacks.stabilize = [](std::vector<double>&) {};
        callbacks.nearest_known_distance = [](std::vector<double>&, double&) { return false; };
        callbacks.continue_candidate = [](std::vector<double>&, const double&)
        {
            return complete_continuation();
        };
        callbacks.accept_candidate = [&accepted](
            const double&,
            const continuation::continuation_curve_result<double>&)
        {
            ++accepted;
        };
        callbacks.discard_candidate = []() {};
        callbacks.save_archive = []() {};

        main_classes::deflation_continuation_detail::knot_execution_policy<double> policy;
        policy.allow_incomplete_restart_intersections = false;
        executor_type executor(&vector_operations, &knots, &log, policy, callbacks);
        const auto result = executor.execute();
        require(analytical_calls == 1, "analytical initialization");
        require(result.solutions_found == 1, "solution count");
        require(result.accepted_curves == 1 && accepted == 1, "accepted curve");
        require(result.skipped_incomplete_knots == 1, "incomplete knot skip");
        require(vector_operations.active_vectors == 0, "rejection cache cleanup");

        fake_knots failure_knots({8.0});
        callbacks.rebuild_intersections = [](const double&) { return intersection_status{}; };
        callbacks.find_deflated_solution = [](const double&) { return true; };
        callbacks.continue_candidate = [](std::vector<double>&, const double&)
        {
            return failed_without_progress();
        };
        int discarded = 0;
        int saves = 0;
        callbacks.discard_candidate = [&discarded]() { ++discarded; };
        callbacks.save_archive = [&saves]() { ++saves; };
        policy.max_failed_continuations_per_knot = 1;
        executor_type failure_executor(
            &vector_operations,
            &failure_knots,
            &log,
            policy,
            callbacks);
        const auto failure_result = failure_executor.execute();
        require(failure_result.discarded_curves == 1 && discarded == 1, "failed curve discard");
        require(saves == 1, "failed curve archive save");

        fake_knots partial_knots({9.0});
        int partial_deflation_calls = 0;
        callbacks.find_deflated_solution =
            [&partial_deflation_calls](const double&)
            {
                return partial_deflation_calls++ == 0;
            };
        callbacks.continue_candidate = [](std::vector<double>&, const double&)
        {
            return failed_after_progress();
        };
        int partial_accepts = 0;
        callbacks.accept_candidate = [&partial_accepts](
            const double&,
            const continuation::continuation_curve_result<double>& result_)
        {
            require(result_.has_recoverable_segment(),
                    "accepted partial exposes recoverable progress");
            ++partial_accepts;
        };
        callbacks.discard_candidate = [&discarded]() { ++discarded; };
        callbacks.save_archive = [&saves]() { ++saves; };
        policy.preserve_partial_curves = true;
        executor_type partial_executor(
            &vector_operations,
            &partial_knots,
            &log,
            policy,
            callbacks);
        const auto partial_result = partial_executor.execute();
        require(partial_result.accepted_curves == 1,
                "partial curve is committed");
        require(partial_result.partial_curves == 1 && partial_accepts == 1,
                "partial curve is classified separately");
        require(partial_result.discarded_curves == 0,
                "partial curve is not discarded");
        require(discarded == 1,
                "partial preservation does not invoke discard callback");
        require(vector_operations.active_vectors == 0,
                "partial path releases rejection-cache vectors");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
