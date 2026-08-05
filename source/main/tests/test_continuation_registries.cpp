#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <containers/failed_continuation_registry.h>
#include <main/deflation_continuation/continuation_recovery_registry.h>

namespace
{

struct vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void init_vector(vector_type&) {}
    void start_use_vector(vector_type&) { ++active_vectors; }
    void stop_use_vector(vector_type&) { --active_vectors; }
    void free_vector(vector_type& value) { value.clear(); }
    void assign(const vector_type& source, vector_type& destination) const
    {
        destination = source;
    }
    void assign_mul(
        const double left_scale,
        const vector_type& left,
        const double right_scale,
        const vector_type& right,
        vector_type& destination) const
    {
        destination.resize(left.size());
        for(std::size_t index = 0; index < left.size(); ++index)
        {
            destination[index] =
                left_scale*left[index] + right_scale*right[index];
        }
    }
    double norm_l2(const vector_type& value) const
    {
        double sum = 0.0;
        for(const double item: value)
        {
            sum += item*item;
        }
        return std::sqrt(sum);
    }

    int active_vectors = 0;
};

struct vector_file_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void write_vector(
        const std::string& file_name,
        const vector_type& value) const
    {
        std::ofstream stream(file_name, std::ofstream::trunc);
        if(!stream)
        {
            throw std::runtime_error("failed to write vector fixture");
        }
        stream << value.size() << '\n';
        for(const double item: value)
        {
            stream << item << '\n';
        }
    }

    void read_vector(
        const std::string& file_name,
        vector_type& value) const
    {
        std::ifstream stream(file_name);
        if(!stream)
        {
            throw std::runtime_error("failed to read vector fixture");
        }
        std::size_t size = 0;
        stream >> size;
        value.resize(size);
        for(double& item: value)
        {
            stream >> item;
        }
    }
};

void require(const bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

continuation::continuation_curve_result<double> failed_result()
{
    continuation::continuation_curve_result<double> result;
    result.semicurves_started = 1;
    auto& semicurve = result.semicurves[0];
    semicurve.status = continuation::semicurve_status::open_recoverable;
    semicurve.failure = continuation::continuation_failure_kind::minimum_step;
    semicurve.endpoint_reason =
        container::curve_endpoint_reason::hard_failure;
    semicurve.direction = 1;
    semicurve.accepted_points = 1;
    semicurve.start_parameter = 19.0;
    semicurve.last_parameter = 21.6063;
    semicurve.attempted_step = 1.0e-4;
    semicurve.retry_count = 3;
    semicurve.message = "minimum continuation step";
    return result;
}

bool contains_temporary_file(const std::filesystem::path& directory)
{
    for(const auto& item:
        std::filesystem::recursive_directory_iterator(directory))
    {
        if(item.path().extension() == ".tmp")
        {
            return true;
        }
    }
    return false;
}

} // namespace

int main()
{
    const std::filesystem::path directory =
        "continuation_registry_test_data";
    try
    {
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);

        vector_operations operations;
        vector_file_operations file_operations;
        using failed_registry = container::failed_continuation_registry<
            vector_operations,
            vector_file_operations>;
        failed_registry::settings failed_settings;
        failed_settings.enabled = true;
        failed_settings.project_directory = directory;
        failed_settings.policy_generation = 7;
        failed_settings.symmetry_fingerprint = "symmetry-v1";
        failed_settings.continuation_fingerprint = "continuation-v1";
        auto distance = [](const std::vector<double>& left,
                           const std::vector<double>& right)
        {
            double value = 0.0;
            for(std::size_t index = 0; index < left.size(); ++index)
            {
                const double difference = left[index] - right[index];
                value += difference*difference;
            }
            return std::sqrt(value);
        };

        {
            failed_registry registry(
                &operations,
                &file_operations,
                failed_settings,
                distance);
            const auto id = registry.record(
                19.0,
                19.0,
                std::vector<double>{1.0, 2.0},
                failed_result());
            require(id == 0, "first failed-continuation ID");
            double nearest = 0.0;
            std::uint64_t nearest_id = 99;
            require(
                registry.nearest_active(
                    19.0,
                    std::vector<double>{1.0, 2.1},
                    nearest,
                    nearest_id),
                "recorded rejection is searchable");
            require(nearest_id == id, "nearest rejection ID");
            require(std::abs(nearest - 0.1) < 1.0e-12,
                    "nearest rejection distance");
            registry.mark_seen(id);
        }

        {
            failed_registry registry(
                &operations,
                &file_operations,
                failed_settings,
                distance);
            require(registry.size() == 1, "failed registry reload");
            require(registry.entries()[0].occurrences == 2,
                    "failed registry occurrence persistence");
            registry.mark_resolved(0);
        }

        auto incompatible_settings = failed_settings;
        incompatible_settings.policy_generation = 8;
        {
            failed_registry registry(
                &operations,
                &file_operations,
                incompatible_settings,
                distance);
            require(registry.size() == 0,
                    "policy generation invalidates old failures");
        }

        using recovery_registry =
            main_classes::deflation_continuation_detail::
                continuation_recovery_registry<vector_file_operations>;
        recovery_registry::settings recovery_settings;
        recovery_settings.enabled = true;
        recovery_settings.project_directory = directory;
        recovery_settings.policy_generation = 3;
        std::uint64_t recovery_id = 0;
        {
            recovery_registry registry(
                &file_operations,
                recovery_settings);
            auto semicurve = failed_result().semicurves[0];
            semicurve.segment_id = 2;
            semicurve.last_point_index = 850;
            semicurve.accepted_points = 850;
            recovery_id = registry.record(
                38,
                19.0,
                semicurve,
                std::vector<double>{3.0, 4.0},
                std::vector<double>{0.5, -0.5},
                21.6063,
                0.0);
            require(recovery_id != 0,
                    "zero is reserved for no recovery task");
            require(registry.pending().size() == 1,
                    "new recovery task is pending");
        }
        {
            recovery_registry registry(
                &file_operations,
                recovery_settings);
            require(registry.entries().size() == 1,
                    "recovery registry reload");
            require(registry.entries()[0].endpoint_point_index == 850,
                    "recovery endpoint persistence");
            std::vector<double> checkpoint_state;
            std::vector<double> checkpoint_tangent;
            double checkpoint_parameter = 0.0;
            double checkpoint_parameter_tangent = 0.0;
            require(
                registry.read_checkpoint(
                    recovery_id,
                    checkpoint_state,
                    checkpoint_tangent,
                    checkpoint_parameter,
                    checkpoint_parameter_tangent),
                "recovery checkpoint reload");
            require(
                checkpoint_state == std::vector<double>({3.0, 4.0}) &&
                checkpoint_tangent == std::vector<double>({0.5, -0.5}),
                "recovery checkpoint values");
            require(registry.begin_attempt(recovery_id, "pseudo_arclength"),
                    "recovery attempt begins transactionally");
            require(registry.mark_pending(
                        recovery_id,
                        "corrector retry limit"),
                    "failed recovery returns to pending");
            require(registry.entries()[0].recovery_attempts == 1,
                    "recovery attempt count");
            require(registry.entries()[0].last_strategy ==
                        "pseudo_arclength",
                    "recovery strategy provenance");
            require(registry.resolve_by_connection(recovery_id, 12),
                    "recovery resolved by topology connection");
            require(registry.pending().empty(),
                    "resolved recovery is not pending");
        }

        require(!contains_temporary_file(directory),
                "registry transactions leave no temporary files");
        require(operations.active_vectors == 0,
                "registry releases all device-vector handles");
        std::filesystem::remove_all(directory);
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        std::filesystem::remove_all(directory);
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
