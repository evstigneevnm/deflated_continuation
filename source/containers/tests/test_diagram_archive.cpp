#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <containers/bifurcation_diagram/diagram_archive.h>

namespace
{

struct payload
{
    int curve_number = 0;
    std::vector<double> values;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& archive, const unsigned int)
    {
        archive & curve_number;
        archive & values;
    }
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
    const auto unique_id = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    const auto archive_path = std::filesystem::temp_directory_path()/
        ("deflated_continuation_diagram_archive_" +
         std::to_string(unique_id) + ".dat");
    const auto missing_path = archive_path.string() + ".missing";

    try
    {
        payload source;
        source.curve_number = 11;
        source.values = {1.25, -3.5, 8.0};
        const auto saved = container::save_diagram_archive(
            archive_path.string(),
            source);
        require(saved.succeeded(), "save status");

        payload restored;
        const auto loaded = container::load_diagram_archive(
            archive_path.string(),
            restored);
        require(loaded.succeeded(), "load status");
        require(restored.curve_number == source.curve_number, "curve number round trip");
        require(restored.values == source.values, "vector round trip");

        payload absent;
        const auto missing = container::load_diagram_archive(missing_path, absent);
        require(
            missing.status == container::diagram_archive_status::missing,
            "missing archive status");

        {
            std::ofstream corrupt(archive_path);
            corrupt << "not a boost archive\n";
        }
        const auto corrupt = container::load_diagram_archive(
            archive_path.string(),
            restored);
        require(
            corrupt.status == container::diagram_archive_status::archive_failed,
            "corrupt archive status");
    }
    catch(const std::exception& error)
    {
        std::filesystem::remove(archive_path);
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::filesystem::remove(archive_path);
    std::cout << "PASSED\n";
    return 0;
}
