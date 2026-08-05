#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <containers/bifurcation_diagram/curve_provenance.h>

namespace
{

void require(const bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

}

int main()
{
    const auto directory =
        std::filesystem::temp_directory_path()/
        "deflated_continuation_curve_provenance_test";
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory);

    try
    {
        container::curve_provenance first;
        first.origin = container::curve_origin::analytical;
        first.analytical_branch_id = 3;
        first.analytical_branch_name = "laminar branch";
        const auto first_file = (directory/"first.dat").string();
        require(
            container::write_curve_provenance(first_file, first),
            "first provenance write");

        container::curve_provenance second;
        second.origin = container::curve_origin::analytical;
        second.analytical_branch_id = 9;
        second.analytical_branch_name = "second exact branch";
        const auto second_file = (directory/"second.dat").string();
        require(
            container::write_curve_provenance(second_file, second),
            "second provenance write");

        container::curve_provenance loaded_first;
        container::curve_provenance loaded_second;
        require(
            container::load_curve_provenance(first_file, loaded_first),
            "first provenance restart load");
        require(
            container::load_curve_provenance(second_file, loaded_second),
            "second provenance restart load");
        require(
            loaded_first.is_analytical() &&
                loaded_first.analytical_branch_id == 3 &&
                loaded_first.analytical_branch_name == "laminar branch",
            "first analytical branch identity");
        require(
            loaded_second.is_analytical() &&
                loaded_second.analytical_branch_id == 9 &&
                loaded_second.analytical_branch_name == "second exact branch",
            "second analytical branch identity");

        container::curve_provenance legacy;
        legacy.origin = container::curve_origin::analytical;
        legacy.analytical_branch_id = 99;
        require(
            !container::load_curve_provenance(
                (directory/"missing.dat").string(),
                legacy),
            "missing legacy provenance is reported");
        require(
            !legacy.is_analytical(),
            "legacy curve without a sidecar defaults to computed");
    }
    catch(const std::exception& exception)
    {
        std::filesystem::remove_all(directory, error);
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }

    std::filesystem::remove_all(directory, error);
    std::cout << "PASSED\n";
    return 0;
}
