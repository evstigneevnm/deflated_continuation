#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <symmetry/finite_group_manifest.h>

namespace
{

void require(const bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    const auto unique = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()/
        ("finite_group_manifest_test_" + std::to_string(unique));
    const std::filesystem::path file_name =
        directory/"symmetry_group.json";
    try
    {
        std::filesystem::create_directories(directory);
        const auto missing =
            symmetry::load_finite_group_manifest(file_name);
        require(
            missing.status ==
                symmetry::finite_group_manifest_status::missing,
            "missing manifest must be reported");

        const symmetry::finite_group_manifest first{
            symmetry::finite_group_manifest::current_version,
            "group-a",
            {"identity", "shift_x", "shift_y", "shift_xy"}};
        require(
            symmetry::save_finite_group_manifest(
                file_name,
                first).succeeded(),
            "first manifest save failed");
        require(
            !std::filesystem::exists(file_name.string() + ".tmp"),
            "temporary manifest survived a successful commit");
        const auto loaded_first =
            symmetry::load_finite_group_manifest(file_name);
        require(loaded_first.succeeded(), "saved manifest did not load");
        require(
            symmetry::finite_group_manifest_matches(
                loaded_first.manifest,
                first.fingerprint,
                first.actions),
            "saved manifest definition changed");
        require(
            !symmetry::finite_group_manifest_matches(
                loaded_first.manifest,
                "group-b",
                first.actions),
            "different fingerprints matched");

        const symmetry::finite_group_manifest replacement{
            symmetry::finite_group_manifest::current_version,
            "group-b",
            {"identity", "swap"}};
        require(
            symmetry::save_finite_group_manifest(
                file_name,
                replacement).succeeded(),
            "manifest replacement failed");
        const auto loaded_replacement =
            symmetry::load_finite_group_manifest(file_name);
        require(
            loaded_replacement.succeeded() &&
                loaded_replacement.manifest.fingerprint == "group-b" &&
                loaded_replacement.manifest.actions.size() == 2,
            "replacement manifest was not committed atomically");

        {
            std::ofstream legacy(file_name);
            legacy
                << "{\"version\":1,\"fingerprint\":\"group-b\","
                << "\"order\":2,\"actions\":[\"identity\",\"swap\"]}\n";
        }
        const auto loaded_legacy =
            symmetry::load_finite_group_manifest(file_name);
        require(
            loaded_legacy.succeeded(),
            "supported legacy manifest did not load for audit");
        require(
            !symmetry::finite_group_manifest_matches(
                loaded_legacy.manifest,
                replacement.fingerprint,
                replacement.actions),
            "legacy quotient-metric manifest bypassed audit");

        {
            std::ofstream wrong_order(file_name);
            wrong_order
                << "{\"version\":1,\"fingerprint\":\"group-b\","
                << "\"order\":3,\"actions\":[\"identity\",\"swap\"]}\n";
        }
        require(
            symmetry::load_finite_group_manifest(file_name).status ==
                symmetry::finite_group_manifest_status::malformed,
            "inconsistent manifest order was accepted");

        {
            std::ofstream duplicate_actions(file_name);
            duplicate_actions
                << "{\"version\":1,\"fingerprint\":\"group-b\","
                << "\"order\":2,\"actions\":[\"identity\",\"identity\"]}\n";
        }
        require(
            symmetry::load_finite_group_manifest(file_name).status ==
                symmetry::finite_group_manifest_status::malformed,
            "duplicate manifest actions were accepted");

        {
            std::ofstream malformed(file_name);
            malformed << "{not-json\n";
        }
        require(
            symmetry::load_finite_group_manifest(file_name).status ==
                symmetry::finite_group_manifest_status::malformed,
            "malformed manifest was accepted");

        std::filesystem::remove_all(directory);
        std::cout << "finite group manifest: PASS\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        std::cerr << "finite group manifest: FAIL: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
