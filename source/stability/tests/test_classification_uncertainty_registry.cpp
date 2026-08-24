#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stability/persistence/classification_uncertainty_registry.h>

namespace
{

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(condition)
        return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

using registry_type =
    stability::persistence::classification_uncertainty_registry<double>;
using stage_type =
    stability::persistence::classification_uncertainty_stage;
using observation_type =
    stability::analysis::unstable_dimension_observation;

std::vector<observation_type> observations()
{
    return {
        {{6, 0}, 3},
        {{7, 0}, 1}
    };
}

void test_restart_and_resolution(const std::filesystem::path& directory)
{
    const auto file_name = directory/"uncertainties.json";
    registry_type::options options;
    options.file_name = file_name;
    options.maximum_diagnostic_length = 32;

    {
        registry_type registry(options);
        require(registry.all().empty(), "new registry is empty");
        require(
            registry.record_failure(
                5,
                14,
                19.98295790554961,
                15,
                20.45964272926033,
                20.45219452888985,
                stage_type::transition_refinement,
                observations(),
                "classification disagreement with a deliberately long "
                "diagnostic"),
            "failure is saved atomically");
        require(registry.unresolved_count() == 1,
            "failure remains unresolved");
        require(registry.all().front().diagnostic.size() == 32,
            "diagnostic is bounded");
        require(
            !std::filesystem::exists(file_name.string() + ".tmp"),
            "atomic save leaves no temporary file");
    }

    {
        registry_type registry(options);
        require(registry.all().size() == 1,
            "failure survives restart");
        const auto& value = registry.all().front();
        require(value.curve_number == 5,
            "curve identity survives restart");
        require(
            value.lower_source_point == 14 &&
                value.upper_source_point == 15,
            "source bracket survives restart");
        require(value.observations.size() == 2,
            "alternative signatures survive restart");
        require(
            value.observations[0].signature.real == 6 &&
                value.observations[0].occurrences == 3 &&
                value.observations[1].signature.real == 7 &&
                value.observations[1].occurrences == 1,
            "alternative signature counts are exact");

        require(
            registry.record_failure(
                5,
                14,
                19.98295790554961,
                15,
                20.45964272926033,
                20.45219452888985,
                stage_type::transition_refinement,
                observations(),
                "second failure"),
            "repeated failure updates the same record");
        require(
            registry.all().size() == 1 &&
                registry.all().front().failure_count == 2,
            "failure count is accumulated without duplicate entries");
        require(
            registry.resolve(
                5,
                14,
                15,
                stage_type::transition_refinement),
            "successful replay resolves the failed bracket");
        require(registry.unresolved_count() == 0,
            "resolved bracket is no longer active");
    }

    {
        registry_type registry(options);
        require(
            registry.all().size() == 1 &&
                registry.all().front().resolved &&
                registry.all().front().resolution_count == 1,
            "resolution survives restart");
        require(
            registry.resolve(
                5,
                14,
                15,
                stage_type::transition_refinement),
            "resolving an already resolved record succeeds as a no-op");
    }
}

void test_remove_resolved(const std::filesystem::path& directory)
{
    registry_type::options options;
    options.file_name = directory/"remove_resolved.json";
    options.retain_resolved = false;

    registry_type registry(options);
    require(
        registry.record_failure(
            2,
            41,
            12.0,
            41,
            12.0,
            12.0,
            stage_type::point_classification,
            {},
            "solver failure"),
        "point failure is saved");
    require(
        registry.resolve(
            2,
            41,
            41,
            stage_type::point_classification),
        "point failure is removed after recovery");
    require(registry.all().empty(),
        "resolved record is removed when retention is disabled");
}

void test_topology_uncertainty_round_trip(
    const std::filesystem::path& directory)
{
    registry_type::options options;
    options.file_name = directory/"topology.json";
    {
        registry_type registry(options);
        require(
            registry.record_failure(
                15,
                938,
                20.28955777834493,
                939,
                20.29726463415439,
                20.29726463415439,
                stage_type::source_path_topology,
                {{{5, 0}, 1}, {{5, 0}, 1}},
                "forward and backward states differ"),
            "source-path topology uncertainty is persisted");
    }
    registry_type restored(options);
    require(
        restored.all().size() == 1 &&
            restored.all().front().stage ==
                stage_type::source_path_topology &&
            !restored.all().front().resolved,
        "source-path topology uncertainty survives restart unresolved");
}

void test_corrupt_registry_is_not_overwritten(
    const std::filesystem::path& directory)
{
    const auto file_name = directory/"corrupt.json";
    {
        std::ofstream stream(file_name);
        stream << "{broken";
    }
    registry_type::options options;
    options.file_name = file_name;
    bool rejected = false;
    try
    {
        registry_type registry(options);
    }
    catch(const std::runtime_error&)
    {
        rejected = true;
    }
    require(rejected, "corrupt registry fails closed");
    std::ifstream stream(file_name);
    std::string contents;
    std::getline(stream, contents);
    require(contents == "{broken",
        "corrupt registry is preserved for diagnosis");
}

void test_failed_save_rolls_back_memory(
    const std::filesystem::path& directory)
{
    const auto file_name = directory/"blocked.json";
    std::filesystem::create_directory(file_name.string() + ".tmp");
    registry_type::options options;
    options.file_name = file_name;
    registry_type registry(options);
    require(
        !registry.record_failure(
            1,
            2,
            3.0,
            2,
            3.0,
            3.0,
            stage_type::point_classification,
            {},
            "not writable"),
        "failed atomic save is reported");
    require(registry.all().empty(),
        "failed atomic save rolls back the in-memory update");
}

} // namespace

int main()
{
    const std::filesystem::path directory =
        "build/classification_uncertainty_registry_test";
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory);

    try
    {
        test_restart_and_resolution(directory);
        test_remove_resolved(directory);
        test_topology_uncertainty_round_trip(directory);
        test_corrupt_registry_is_not_overwritten(directory);
        test_failed_save_rolls_back_memory(directory);
    }
    catch(const std::exception& exception)
    {
        ++failures;
        std::cerr << "Unexpected exception: " << exception.what() << '\n';
    }

    std::filesystem::remove_all(directory, error);
    std::cout
        << "Classification uncertainty registry checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
