#ifndef __BIFURCATION_DIAGRAM_TOPOLOGY_BRANCH_TOPOLOGY_REGISTRY_H__
#define __BIFURCATION_DIAGRAM_TOPOLOGY_BRANCH_TOPOLOGY_REGISTRY_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>
#include <common/scalar_math.h>
#include <containers/curve_endpoint_reason.h>

namespace container
{
namespace topology
{

enum class endpoint_status
{
    terminal,
    recovery_open,
    joined
};

inline const char* to_string(const endpoint_status status)
{
    switch(status)
    {
    case endpoint_status::terminal:
        return "terminal";
    case endpoint_status::recovery_open:
        return "recovery_open";
    case endpoint_status::joined:
        return "joined";
    }
    return "terminal";
}

enum class connection_kind
{
    continuation_join,
    transverse_junction
};

inline const char* to_string(const connection_kind kind)
{
    return kind == connection_kind::continuation_join
        ? "continuation_join"
        : "transverse_junction";
}

template<class Scalar>
struct branch_endpoint_record
{
    std::uint64_t id = 0;
    int curve_number = -1;
    std::uint64_t segment_id = 0;
    std::uint64_t semicurve_id = 0;
    std::uint64_t point_index = 0;
    Scalar parameter = Scalar(0);
    Scalar parameter_tangent = Scalar(0);
    curve_endpoint_reason reason = curve_endpoint_reason::none;
    endpoint_status status = endpoint_status::terminal;
    std::uint64_t recovery_task_id = 0;
    std::string state_file;
    std::string tangent_file;
};

template<class Scalar>
struct branch_connection_record
{
    std::uint64_t id = 0;
    std::uint64_t first_endpoint_id = 0;
    std::uint64_t second_endpoint_id = 0;
    connection_kind kind = connection_kind::continuation_join;
    Scalar parameter_distance = Scalar(0);
    Scalar state_distance = Scalar(0);
    Scalar tangent_line_similarity = Scalar(0);
};

template<class Scalar>
struct endpoint_match_policy
{
    Scalar absolute_parameter_tolerance = Scalar(1.0e-7);
    Scalar relative_parameter_tolerance = Scalar(1.0e-9);
    Scalar state_tolerance = Scalar(1.0e-8);
    Scalar minimum_tangent_line_similarity = Scalar(0.9);
    bool record_transverse_junctions = true;
};

template<class Scalar>
struct endpoint_record_result
{
    std::uint64_t endpoint_id = 0;
    bool matched = false;
    std::uint64_t matched_endpoint_id = 0;
    std::uint64_t connection_id = 0;
    connection_kind kind = connection_kind::continuation_join;
    std::uint64_t current_recovery_task_id = 0;
    std::uint64_t matched_recovery_task_id = 0;
};

template<class Scalar>
struct endpoint_geometry_metrics
{
    Scalar state_distance = Scalar(0);
    Scalar tangent_line_similarity = Scalar(0);
};

template<class VectorOperations, class VectorFileOperations>
class branch_topology_registry
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using endpoint_type = branch_endpoint_record<scalar_type>;
    using connection_type = branch_connection_record<scalar_type>;
    using result_type = endpoint_record_result<scalar_type>;
    using endpoint_geometry_type = std::function<
        endpoint_geometry_metrics<scalar_type>(
        const vector_type&,
        const vector_type&,
        const scalar_type&,
        const vector_type&,
        const vector_type&,
        const scalar_type&)>;

    struct settings
    {
        bool enabled = true;
        std::filesystem::path project_directory;
        std::string registry_file = "branch_topology.json";
        std::string endpoint_directory = "branch_topology/endpoints";
        std::uint64_t policy_generation = 1;
        std::string symmetry_fingerprint;
        endpoint_match_policy<scalar_type> match;
    };

    branch_topology_registry(
        VectorOperations* vector_operations,
        VectorFileOperations* file_operations,
        settings settings_,
        endpoint_geometry_type endpoint_geometry):
        vector_operations_(vector_operations),
        file_operations_(file_operations),
        settings_(std::move(settings_)),
        endpoint_geometry_(std::move(endpoint_geometry))
    {
        if(vector_operations_ == nullptr || file_operations_ == nullptr ||
           !endpoint_geometry_)
        {
            throw std::invalid_argument(
                "branch_topology_registry requires vector operations, file operations, and quotient distance");
        }
        validate_policy();
        load();
    }

    result_type record_endpoint(
        const int curve_number,
        const std::uint64_t segment_id,
        const std::uint64_t semicurve_id,
        const std::uint64_t point_index,
        const scalar_type parameter,
        const vector_type& state,
        const vector_type& tangent,
        const scalar_type parameter_tangent,
        const curve_endpoint_reason reason,
        const std::uint64_t recovery_task_id)
    {
        result_type result;
        if(!settings_.enabled)
        {
            return result;
        }

        endpoint_type endpoint;
        endpoint.id = next_endpoint_id_++;
        endpoint.curve_number = curve_number;
        endpoint.segment_id = segment_id;
        endpoint.semicurve_id = semicurve_id;
        endpoint.point_index = point_index;
        endpoint.parameter = parameter;
        endpoint.parameter_tangent = parameter_tangent;
        endpoint.reason = reason;
        endpoint.status = recovery_task_id == 0
            ? endpoint_status::terminal
            : endpoint_status::recovery_open;
        endpoint.recovery_task_id = recovery_task_id;
        const std::string stem = std::to_string(endpoint.id);
        endpoint.state_file = settings_.endpoint_directory + "/" +
            stem + "_state.dat";
        endpoint.tangent_file = settings_.endpoint_directory + "/" +
            stem + "_tangent.dat";
        write_vector_atomic(resolve(endpoint.state_file), state);
        write_vector_atomic(resolve(endpoint.tangent_file), tangent);

        result.endpoint_id = endpoint.id;
        result.current_recovery_task_id = recovery_task_id;
        find_and_connect(endpoint, state, tangent, result);
        endpoints_.push_back(std::move(endpoint));
        save();
        return result;
    }

    const std::vector<endpoint_type>& endpoints() const
    {
        return endpoints_;
    }

    const std::vector<connection_type>& connections() const
    {
        return connections_;
    }

    std::size_t logical_branch_count() const
    {
        using segment_key = std::pair<int, std::uint64_t>;
        std::map<segment_key, std::size_t> indices;
        for(const auto& endpoint: endpoints_)
        {
            const segment_key key(endpoint.curve_number, endpoint.segment_id);
            if(indices.find(key) == indices.end())
            {
                indices.emplace(key, indices.size());
            }
        }
        std::vector<std::size_t> parent(indices.size());
        for(std::size_t index = 0; index < parent.size(); ++index)
        {
            parent[index] = index;
        }
        auto root = [&parent](std::size_t value)
        {
            while(parent[value] != value)
            {
                value = parent[value];
            }
            return value;
        };
        for(const auto& connection: connections_)
        {
            if(connection.kind != connection_kind::continuation_join)
            {
                continue;
            }
            const auto* first = find_endpoint(connection.first_endpoint_id);
            const auto* second = find_endpoint(connection.second_endpoint_id);
            if(first == nullptr || second == nullptr)
            {
                continue;
            }
            const auto first_index = indices.at(
                segment_key(first->curve_number, first->segment_id));
            const auto second_index = indices.at(
                segment_key(second->curve_number, second->segment_id));
            const auto first_root = root(first_index);
            const auto second_root = root(second_index);
            if(first_root != second_root)
            {
                parent[second_root] = first_root;
            }
        }
        std::size_t count = 0;
        for(std::size_t index = 0; index < parent.size(); ++index)
        {
            if(root(index) == index)
            {
                ++count;
            }
        }
        return count;
    }

    void save() const
    {
        if(!settings_.enabled)
        {
            return;
        }
        nlohmann::json root;
        root["version"] = 1;
        root["policy_generation"] = settings_.policy_generation;
        root["symmetry_fingerprint"] = settings_.symmetry_fingerprint;
        root["endpoints"] = nlohmann::json::array();
        root["connections"] = nlohmann::json::array();
        for(const auto& endpoint: endpoints_)
        {
            root["endpoints"].push_back({
                {"id", endpoint.id},
                {"curve_number", endpoint.curve_number},
                {"segment_id", endpoint.segment_id},
                {"semicurve_id", endpoint.semicurve_id},
                {"point_index", endpoint.point_index},
                {"parameter", double(endpoint.parameter)},
                {"parameter_tangent", double(endpoint.parameter_tangent)},
                {"reason", container::to_string(endpoint.reason)},
                {"status", to_string(endpoint.status)},
                {"recovery_task_id", endpoint.recovery_task_id},
                {"state_file", endpoint.state_file},
                {"tangent_file", endpoint.tangent_file}
            });
        }
        for(const auto& connection: connections_)
        {
            root["connections"].push_back({
                {"id", connection.id},
                {"first_endpoint_id", connection.first_endpoint_id},
                {"second_endpoint_id", connection.second_endpoint_id},
                {"kind", to_string(connection.kind)},
                {"parameter_distance", double(connection.parameter_distance)},
                {"state_distance", double(connection.state_distance)},
                {"tangent_line_similarity", double(connection.tangent_line_similarity)}
            });
        }
        write_json_atomic(resolve(settings_.registry_file), root);
    }

private:
    void validate_policy() const
    {
        const auto& policy = settings_.match;
        if(policy.absolute_parameter_tolerance < scalar_type(0) ||
           policy.relative_parameter_tolerance < scalar_type(0) ||
           policy.state_tolerance < scalar_type(0) ||
           policy.minimum_tangent_line_similarity < scalar_type(0) ||
           policy.minimum_tangent_line_similarity > scalar_type(1))
        {
            throw std::invalid_argument(
                "invalid branch topology endpoint matching policy");
        }
    }

    scalar_type parameter_tolerance(
        const scalar_type left,
        const scalar_type right) const
    {
        const scalar_type scale = std::max<scalar_type>(
            scalar_type(1),
            std::max<scalar_type>(
                common::scalar_math::abs(left),
                common::scalar_math::abs(right)));
        return settings_.match.absolute_parameter_tolerance +
            settings_.match.relative_parameter_tolerance*scale;
    }

    void find_and_connect(
        endpoint_type& candidate,
        const vector_type& state,
        const vector_type& tangent,
        result_type& result)
    {
        if(endpoints_.empty())
        {
            return;
        }
        vector_type other_state;
        vector_type other_tangent;
        vector_operations_->init_vector(other_state);
        vector_operations_->start_use_vector(other_state);
        vector_operations_->init_vector(other_tangent);
        vector_operations_->start_use_vector(other_tangent);

        endpoint_type* best_join = nullptr;
        endpoint_type* best_transverse = nullptr;
        scalar_type best_join_distance =
            std::numeric_limits<scalar_type>::infinity();
        scalar_type best_transverse_distance =
            std::numeric_limits<scalar_type>::infinity();
        scalar_type best_join_similarity = scalar_type(0);
        scalar_type best_transverse_similarity = scalar_type(0);

        try
        {
            for(auto& existing: endpoints_)
            {
                if(existing.curve_number == candidate.curve_number &&
                   existing.segment_id == candidate.segment_id)
                {
                    continue;
                }
                if(existing.status != endpoint_status::recovery_open &&
                   candidate.status != endpoint_status::recovery_open)
                {
                    continue;
                }
                const scalar_type parameter_distance =
                    common::scalar_math::abs(
                        existing.parameter - candidate.parameter);
                if(parameter_distance > parameter_tolerance(
                       existing.parameter,
                       candidate.parameter))
                {
                    continue;
                }
                file_operations_->read_vector(
                    resolve(existing.state_file).string(),
                    other_state);
                file_operations_->read_vector(
                    resolve(existing.tangent_file).string(),
                    other_tangent);
                const auto geometry = endpoint_geometry_(
                    state,
                    tangent,
                    candidate.parameter_tangent,
                    other_state,
                    other_tangent,
                    existing.parameter_tangent);
                const scalar_type distance = geometry.state_distance;
                if(distance > settings_.match.state_tolerance)
                {
                    continue;
                }
                const scalar_type similarity =
                    geometry.tangent_line_similarity;
                if(similarity >=
                       settings_.match.minimum_tangent_line_similarity &&
                   distance < best_join_distance)
                {
                    best_join = &existing;
                    best_join_distance = distance;
                    best_join_similarity = similarity;
                }
                else if(settings_.match.record_transverse_junctions &&
                        distance < best_transverse_distance)
                {
                    best_transverse = &existing;
                    best_transverse_distance = distance;
                    best_transverse_similarity = similarity;
                }
            }

            endpoint_type* matched = best_join != nullptr
                ? best_join
                : best_transverse;
            if(matched != nullptr)
            {
                connection_type connection;
                connection.id = next_connection_id_++;
                connection.first_endpoint_id = matched->id;
                connection.second_endpoint_id = candidate.id;
                connection.kind = best_join != nullptr
                    ? connection_kind::continuation_join
                    : connection_kind::transverse_junction;
                connection.parameter_distance =
                    common::scalar_math::abs(
                        matched->parameter - candidate.parameter);
                connection.state_distance = best_join != nullptr
                    ? best_join_distance
                    : best_transverse_distance;
                connection.tangent_line_similarity = best_join != nullptr
                    ? best_join_similarity
                    : best_transverse_similarity;
                connections_.push_back(connection);
                result.matched = true;
                result.matched_endpoint_id = matched->id;
                result.connection_id = connection.id;
                result.kind = connection.kind;
                result.matched_recovery_task_id = matched->recovery_task_id;
                if(connection.kind == connection_kind::continuation_join)
                {
                    if(matched->status == endpoint_status::recovery_open)
                    {
                        matched->status = endpoint_status::joined;
                    }
                    if(candidate.status == endpoint_status::recovery_open)
                    {
                        candidate.status = endpoint_status::joined;
                    }
                }
            }
        }
        catch(...)
        {
            vector_operations_->stop_use_vector(other_tangent);
            vector_operations_->free_vector(other_tangent);
            vector_operations_->stop_use_vector(other_state);
            vector_operations_->free_vector(other_state);
            throw;
        }
        vector_operations_->stop_use_vector(other_tangent);
        vector_operations_->free_vector(other_tangent);
        vector_operations_->stop_use_vector(other_state);
        vector_operations_->free_vector(other_state);
    }

    const endpoint_type* find_endpoint(const std::uint64_t id) const
    {
        const auto item = std::find_if(
            endpoints_.begin(),
            endpoints_.end(),
            [id](const endpoint_type& endpoint)
            {
                return endpoint.id == id;
            });
        return item == endpoints_.end() ? nullptr : &*item;
    }

    static endpoint_status endpoint_status_from_string(
        const std::string& value)
    {
        if(value == "recovery_open")
        {
            return endpoint_status::recovery_open;
        }
        if(value == "joined")
        {
            return endpoint_status::joined;
        }
        return endpoint_status::terminal;
    }

    static connection_kind connection_kind_from_string(
        const std::string& value)
    {
        return value == "transverse_junction"
            ? connection_kind::transverse_junction
            : connection_kind::continuation_join;
    }

    std::filesystem::path resolve(const std::string& path) const
    {
        const std::filesystem::path value(path);
        return value.is_absolute()
            ? value
            : settings_.project_directory/value;
    }

    static void replace_file(
        const std::filesystem::path& source,
        const std::filesystem::path& destination)
    {
        std::error_code error;
        std::filesystem::rename(source, destination, error);
        if(error)
        {
            std::error_code ignored;
            std::filesystem::remove(source, ignored);
            throw std::runtime_error(
                "failed to commit branch topology file " +
                destination.string() + ": " + error.message());
        }
    }

    void write_vector_atomic(
        const std::filesystem::path& destination,
        const vector_type& value) const
    {
        std::error_code directory_error;
        std::filesystem::create_directories(
            destination.parent_path(),
            directory_error);
        if(directory_error)
        {
            throw std::runtime_error(
                "failed to create branch topology endpoint directory: " +
                directory_error.message());
        }
        const std::filesystem::path temporary =
            destination.string() + ".tmp";
        file_operations_->write_vector(temporary.string(), value);
        replace_file(temporary, destination);
    }

    static void write_json_atomic(
        const std::filesystem::path& destination,
        const nlohmann::json& root)
    {
        std::error_code directory_error;
        std::filesystem::create_directories(
            destination.parent_path(),
            directory_error);
        if(directory_error)
        {
            throw std::runtime_error(
                "failed to create branch topology registry directory: " +
                directory_error.message());
        }
        const std::filesystem::path temporary =
            destination.string() + ".tmp";
        {
            std::ofstream stream(temporary, std::ofstream::trunc);
            if(!stream)
            {
                throw std::runtime_error(
                    "failed to write branch topology registry");
            }
            stream << root.dump(2) << '\n';
        }
        replace_file(temporary, destination);
    }

    void load()
    {
        if(!settings_.enabled)
        {
            return;
        }
        std::ifstream stream(resolve(settings_.registry_file));
        if(!stream)
        {
            return;
        }
        nlohmann::json root;
        stream >> root;
        if(root.value("version", 0) != 1 ||
           root.value("policy_generation", std::uint64_t(0)) !=
               settings_.policy_generation ||
           root.value("symmetry_fingerprint", std::string{}) !=
               settings_.symmetry_fingerprint)
        {
            return;
        }
        for(const auto& value: root.value(
                "endpoints",
                nlohmann::json::array()))
        {
            endpoint_type endpoint;
            endpoint.id = value.value("id", std::uint64_t(0));
            endpoint.curve_number = value.value("curve_number", -1);
            endpoint.segment_id = value.value("segment_id", std::uint64_t(0));
            endpoint.semicurve_id = value.value("semicurve_id", std::uint64_t(0));
            endpoint.point_index = value.value("point_index", std::uint64_t(0));
            endpoint.parameter = scalar_type(value.value("parameter", 0.0));
            endpoint.parameter_tangent = scalar_type(
                value.value("parameter_tangent", 0.0));
            endpoint.reason = curve_endpoint_reason_from_string(
                value.value("reason", std::string("none")));
            endpoint.status = endpoint_status_from_string(
                value.value("status", std::string("terminal")));
            endpoint.recovery_task_id = value.value(
                "recovery_task_id",
                std::uint64_t(0));
            endpoint.state_file = value.value("state_file", std::string{});
            endpoint.tangent_file = value.value("tangent_file", std::string{});
            endpoints_.push_back(std::move(endpoint));
            next_endpoint_id_ = std::max(
                next_endpoint_id_,
                endpoints_.back().id + 1);
        }
        for(const auto& value: root.value(
                "connections",
                nlohmann::json::array()))
        {
            connection_type connection;
            connection.id = value.value("id", std::uint64_t(0));
            connection.first_endpoint_id = value.value(
                "first_endpoint_id",
                std::uint64_t(0));
            connection.second_endpoint_id = value.value(
                "second_endpoint_id",
                std::uint64_t(0));
            connection.kind = connection_kind_from_string(
                value.value("kind", std::string("continuation_join")));
            connection.parameter_distance = scalar_type(
                value.value("parameter_distance", 0.0));
            connection.state_distance = scalar_type(
                value.value("state_distance", 0.0));
            connection.tangent_line_similarity = scalar_type(
                value.value("tangent_line_similarity", 0.0));
            connections_.push_back(std::move(connection));
            next_connection_id_ = std::max(
                next_connection_id_,
                connections_.back().id + 1);
        }
    }

    VectorOperations* vector_operations_;
    VectorFileOperations* file_operations_;
    settings settings_;
    endpoint_geometry_type endpoint_geometry_;
    std::vector<endpoint_type> endpoints_;
    std::vector<connection_type> connections_;
    std::uint64_t next_endpoint_id_ = 1;
    std::uint64_t next_connection_id_ = 1;
};

} // namespace topology
} // namespace container

#endif // __BIFURCATION_DIAGRAM_TOPOLOGY_BRANCH_TOPOLOGY_REGISTRY_H__
