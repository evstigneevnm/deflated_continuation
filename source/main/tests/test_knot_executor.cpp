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
        callbacks.continue_candidate = [](std::vector<double>&, const double&) { return true; };
        callbacks.accept_candidate = [&accepted](const double&) { ++accepted; };
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
        callbacks.continue_candidate = [](std::vector<double>&, const double&) { return false; };
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
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
