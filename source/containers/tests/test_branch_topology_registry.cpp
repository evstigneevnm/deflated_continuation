#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <containers/bifurcation_diagram/topology/branch_topology_registry.h>

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
    double norm_l2(const vector_type& value) const
    {
        return std::sqrt(scalar_prod(value, value));
    }
    double scalar_prod(
        const vector_type& left,
        const vector_type& right) const
    {
        double result = 0.0;
        for(std::size_t index = 0; index < left.size(); ++index)
        {
            result += left[index]*right[index];
        }
        return result;
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
            throw std::runtime_error("failed to write topology fixture");
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
            throw std::runtime_error("failed to read topology fixture");
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
        "branch_topology_registry_test_data";
    try
    {
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        vector_operations operations;
        vector_file_operations file_operations;
        using registry_type = container::topology::branch_topology_registry<
            vector_operations,
            vector_file_operations>;
        registry_type::settings settings;
        settings.project_directory = directory;
        settings.policy_generation = 4;
        settings.symmetry_fingerprint = "quotient-v2";
        settings.match.absolute_parameter_tolerance = 1.0e-5;
        settings.match.state_tolerance = 1.0e-6;
        settings.match.minimum_tangent_line_similarity = 0.95;
        auto geometry = [](
            const std::vector<double>& reference,
            const std::vector<double>& reference_tangent,
            const double reference_parameter_tangent,
            const std::vector<double>& source,
            const std::vector<double>& source_tangent,
            const double source_parameter_tangent)
        {
            double identity_distance_sq = 0.0;
            double reflected_distance_sq = 0.0;
            for(std::size_t index = 0; index < reference.size(); ++index)
            {
                const double identity = source[index] - reference[index];
                const double reflected = -source[index] - reference[index];
                identity_distance_sq += identity*identity;
                reflected_distance_sq += reflected*reflected;
            }
            const double action = reflected_distance_sq < identity_distance_sq
                ? -1.0
                : 1.0;
            const double state_distance_sq = std::min(
                identity_distance_sq,
                reflected_distance_sq);
            double tangent_product =
                reference_parameter_tangent*source_parameter_tangent;
            double reference_norm_sq =
                reference_parameter_tangent*reference_parameter_tangent;
            double source_norm_sq =
                source_parameter_tangent*source_parameter_tangent;
            for(std::size_t index = 0;
                index < reference_tangent.size();
                ++index)
            {
                tangent_product += reference_tangent[index]*
                    action*source_tangent[index];
                reference_norm_sq += reference_tangent[index]*
                    reference_tangent[index];
                source_norm_sq += source_tangent[index]*source_tangent[index];
            }
            container::topology::endpoint_geometry_metrics<double> result;
            result.state_distance = std::sqrt(state_distance_sq);
            result.tangent_line_similarity = std::abs(tangent_product)/
                std::sqrt(reference_norm_sq*source_norm_sq);
            return result;
        };

        {
            registry_type registry(
                &operations,
                &file_operations,
                settings,
                geometry);
            const auto open = registry.record_endpoint(
                3,
                1,
                1,
                80,
                21.6063,
                std::vector<double>{1.0, 2.0},
                std::vector<double>{1.0, 0.0},
                0.1,
                container::curve_endpoint_reason::hard_failure,
                41);
            require(!open.matched, "first open endpoint has no match");

            const auto joined = registry.record_endpoint(
                9,
                2,
                2,
                17,
                21.6063001,
                std::vector<double>{-1.0, -2.0 - 1.0e-8},
                std::vector<double>{-1.0, 0.0},
                0.1,
                container::curve_endpoint_reason::known_branch,
                0);
            require(joined.matched, "collinear endpoint is matched");
            require(joined.kind ==
                        container::topology::connection_kind::continuation_join,
                    "opposite tangent orientations form one branch");
            require(joined.matched_recovery_task_id == 41,
                    "join identifies recovery task");
            require(registry.logical_branch_count() == 1,
                    "joined physical segments form one logical branch");

            registry.record_endpoint(
                12,
                1,
                1,
                9,
                32.0,
                std::vector<double>{3.0, 4.0},
                std::vector<double>{1.0, 0.0},
                0.0,
                container::curve_endpoint_reason::hard_failure,
                77);
            const auto transverse = registry.record_endpoint(
                15,
                1,
                1,
                4,
                32.0,
                std::vector<double>{-3.0, -4.0},
                std::vector<double>{0.0, -1.0},
                0.0,
                container::curve_endpoint_reason::known_branch,
                0);
            require(transverse.matched, "transverse endpoint is recorded");
            require(transverse.kind ==
                        container::topology::connection_kind::transverse_junction,
                    "transverse tangents are not joined");
            require(registry.logical_branch_count() == 3,
                    "junction does not merge logical branches");
        }

        {
            registry_type registry(
                &operations,
                &file_operations,
                settings,
                geometry);
            require(registry.endpoints().size() == 4,
                    "endpoint registry survives restart");
            require(registry.connections().size() == 2,
                    "connection registry survives restart");
            require(registry.logical_branch_count() == 3,
                    "logical branch view survives restart");
        }

        require(operations.active_vectors == 0,
                "topology matcher releases vector handles");
        require(!contains_temporary_file(directory),
                "topology transaction leaves no temporary files");
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
