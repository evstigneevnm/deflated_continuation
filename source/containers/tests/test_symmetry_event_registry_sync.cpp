#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <containers/bifurcation_diagram/symmetry_event_registry_sync.h>
#include <containers/symmetry_event_record.h>
#include <containers/symmetry_event_registry.h>

namespace
{

struct fake_vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void init_vector(vector_type&) {}
    void start_use_vector(vector_type&) {}
    void stop_use_vector(vector_type&) {}
    void free_vector(vector_type&) {}

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
};

struct fake_file_operations
{
    void read_vector(const std::string& path, std::vector<double>& output)
    {
        output = vectors.at(path);
    }

    std::unordered_map<std::string, std::vector<double>> vectors;
};

struct fake_log
{
    template<class... Args>
    void warning_f(const char*, Args&&...)
    {
    }
};

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

container::symmetry_event_record<double> make_record(
    int curve_number,
    uint64_t point_index,
    uint64_t vector_file_id,
    double lambda)
{
    container::symmetry_event_record<double> record;
    record.curve_number = curve_number;
    record.point_index = point_index;
    record.vector_available = true;
    record.vector_file_id = vector_file_id;
    record.lambda = lambda;
    record.previous_order = 1;
    record.candidate_order = 2;
    record.previous_orbit_type =
        symmetry::translation::orbit_type::cyclic_1d(1);
    record.candidate_orbit_type =
        symmetry::translation::orbit_type::cyclic_1d(2);
    return record;
}

}

int main()
{
    const auto unique_id = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    const auto project_directory = std::filesystem::temp_directory_path()/
        ("deflated_continuation_event_sync_" + std::to_string(unique_id));
    const auto registry_path = project_directory/"symmetry_event_registry.json";

    try
    {
        std::filesystem::create_directories(project_directory/"0");
        std::filesystem::create_directories(project_directory/"1");

        fake_vector_operations vector_operations;
        fake_file_operations file_operations;
        fake_log log;
        file_operations.vectors[(project_directory/"0"/"10").string()] = {1.0, 2.0};
        file_operations.vectors[(project_directory/"1"/"20").string()] = {1.0, 2.0 + 1.0e-10};

        std::vector<container::symmetry_event_record<double>> records;
        records.push_back(make_record(0, 3, 10, 6.0));
        records.push_back(make_record(1, 4, 20, 6.0 + 1.0e-10));

        container::symmetry_event_registry<double> registry(registry_path);
        container::symmetry_event_registry_sync<
            fake_vector_operations,
            fake_file_operations,
            fake_log> synchronizer(&vector_operations, &file_operations, &log);
        const auto result = synchronizer.synchronize(
            registry,
            records,
            project_directory,
            1.0e-6,
            1.0e-6,
            [](std::vector<double>&) {});
        require(result.attempted, "sync attempted");
        require(result.saved, "registry saved");
        require(result.event_count == 1, "equivalent events merged");

        container::symmetry_event_registry<double> restored(registry_path);
        require(restored.load(), "registry reload");
        require(restored.all().size() == 1, "reloaded event count");
        require(restored.all()[0].incidents.size() == 2, "reloaded incidents");
    }
    catch(const std::exception& error)
    {
        std::filesystem::remove_all(project_directory);
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::filesystem::remove_all(project_directory);
    std::cout << "PASSED\n";
    return 0;
}
