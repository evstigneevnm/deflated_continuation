#ifndef __MAIN_DEFLATION_CONTINUATION_RECOVERY_REGISTRY_H__
#define __MAIN_DEFLATION_CONTINUATION_RECOVERY_REGISTRY_H__

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>
#include <continuation/continuation_result.h>

namespace main_classes
{
namespace deflation_continuation_detail
{

enum class recovery_task_status
{
    pending,
    running,
    recovered,
    recovered_by_connection,
    exhausted,
    superseded
};

inline const char* to_string(const recovery_task_status status)
{
    switch(status)
    {
    case recovery_task_status::pending:
        return "pending";
    case recovery_task_status::running:
        return "running";
    case recovery_task_status::recovered:
        return "recovered";
    case recovery_task_status::recovered_by_connection:
        return "recovered_by_connection";
    case recovery_task_status::exhausted:
        return "exhausted";
    case recovery_task_status::superseded:
        return "superseded";
    }
    return "pending";
}

template<class Scalar>
struct continuation_recovery_entry
{
    std::uint64_t id = 0;
    int curve_number = -1;
    std::uint64_t segment_id = 0;
    std::uint64_t endpoint_point_index = 0;
    int direction = 0;
    Scalar seed_parameter = Scalar(0);
    Scalar endpoint_parameter = Scalar(0);
    Scalar endpoint_parameter_tangent = Scalar(0);
    Scalar attempted_step = Scalar(0);
    unsigned int retry_count = 0;
    unsigned int accepted_points = 0;
    continuation::continuation_failure_kind failure =
        continuation::continuation_failure_kind::unknown;
    std::string state_file;
    std::string tangent_file;
    std::string message;
    recovery_task_status status = recovery_task_status::pending;
    std::uint64_t connection_id = 0;
    unsigned int recovery_attempts = 0;
    std::string last_strategy;
    std::string last_result;
};

template<class VectorFileOperations>
class continuation_recovery_registry
{
public:
    using scalar_type = typename VectorFileOperations::scalar_type;
    using vector_type = typename VectorFileOperations::vector_type;
    using entry_type = continuation_recovery_entry<scalar_type>;

    struct settings
    {
        bool enabled = true;
        std::filesystem::path project_directory;
        std::string registry_file = "continuation_recovery_registry.json";
        std::string checkpoint_directory = "recovery_checkpoints";
        std::uint64_t policy_generation = 1;
    };

    continuation_recovery_registry(
        VectorFileOperations* file_operations,
        settings settings_):
        file_operations_(file_operations),
        settings_(std::move(settings_))
    {
        if(file_operations_ == nullptr)
        {
            throw std::invalid_argument(
                "continuation_recovery_registry requires file operations");
        }
        load();
    }

    std::uint64_t record(
        const int curve_number,
        const scalar_type seed_parameter,
        const continuation::semicurve_result<scalar_type>& semicurve,
        const vector_type& state,
        const vector_type& tangent,
        const scalar_type endpoint_parameter,
        const scalar_type endpoint_parameter_tangent)
    {
        if(!settings_.enabled)
        {
            return 0;
        }
        const std::uint64_t id = next_id_++;
        const std::filesystem::path directory =
            resolve(settings_.checkpoint_directory);
        std::error_code directory_error;
        std::filesystem::create_directories(directory, directory_error);
        if(directory_error)
        {
            throw std::runtime_error(
                "failed to create recovery checkpoint directory: " +
                directory_error.message());
        }

        const std::string stem = std::to_string(id);
        const std::string state_file =
            settings_.checkpoint_directory + "/" + stem + "_state.dat";
        const std::string tangent_file =
            settings_.checkpoint_directory + "/" + stem + "_tangent.dat";
        write_vector_atomic(resolve(state_file), state);
        write_vector_atomic(resolve(tangent_file), tangent);

        entry_type entry;
        entry.id = id;
        entry.curve_number = curve_number;
        entry.segment_id = semicurve.segment_id;
        entry.endpoint_point_index = semicurve.last_point_index;
        entry.direction = semicurve.direction;
        entry.seed_parameter = seed_parameter;
        entry.endpoint_parameter = endpoint_parameter;
        entry.endpoint_parameter_tangent = endpoint_parameter_tangent;
        entry.attempted_step = semicurve.attempted_step;
        entry.retry_count = semicurve.retry_count;
        entry.accepted_points = semicurve.accepted_points;
        entry.failure = semicurve.failure;
        entry.state_file = state_file;
        entry.tangent_file = tangent_file;
        entry.message = semicurve.message;
        entries_.push_back(std::move(entry));
        save();
        return id;
    }

    bool resolve_by_connection(
        const std::uint64_t recovery_id,
        const std::uint64_t connection_id)
    {
        for(auto& entry: entries_)
        {
            if(entry.id != recovery_id)
            {
                continue;
            }
            entry.status = recovery_task_status::recovered_by_connection;
            entry.connection_id = connection_id;
            save();
            return true;
        }
        return false;
    }

    bool read_checkpoint(
        const std::uint64_t recovery_id,
        vector_type& state,
        vector_type& tangent,
        scalar_type& parameter,
        scalar_type& parameter_tangent) const
    {
        const auto* entry = find(recovery_id);
        if(entry == nullptr)
        {
            return false;
        }
        file_operations_->read_vector(
            resolve(entry->state_file).string(),
            state);
        file_operations_->read_vector(
            resolve(entry->tangent_file).string(),
            tangent);
        parameter = entry->endpoint_parameter;
        parameter_tangent = entry->endpoint_parameter_tangent;
        return true;
    }

    bool begin_attempt(
        const std::uint64_t recovery_id,
        const std::string& strategy)
    {
        auto* entry = find(recovery_id);
        if(entry == nullptr ||
           (entry->status != recovery_task_status::pending &&
            entry->status != recovery_task_status::running))
        {
            return false;
        }
        entry->status = recovery_task_status::running;
        ++entry->recovery_attempts;
        entry->last_strategy = strategy;
        entry->last_result.clear();
        save();
        return true;
    }

    bool mark_pending(
        const std::uint64_t recovery_id,
        const std::string& result)
    {
        return update_attempt_result(
            recovery_id,
            recovery_task_status::pending,
            result);
    }

    bool mark_recovered(
        const std::uint64_t recovery_id,
        const std::string& result)
    {
        return update_attempt_result(
            recovery_id,
            recovery_task_status::recovered,
            result);
    }

    bool mark_exhausted(
        const std::uint64_t recovery_id,
        const std::string& result)
    {
        return update_attempt_result(
            recovery_id,
            recovery_task_status::exhausted,
            result);
    }

    const std::vector<entry_type>& entries() const
    {
        return entries_;
    }

    std::vector<entry_type> pending() const
    {
        std::vector<entry_type> result;
        for(const auto& entry: entries_)
        {
            if(entry.status == recovery_task_status::pending)
            {
                result.push_back(entry);
            }
        }
        return result;
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
        root["entries"] = nlohmann::json::array();
        for(const auto& entry: entries_)
        {
            root["entries"].push_back({
                {"id", entry.id},
                {"curve_number", entry.curve_number},
                {"segment_id", entry.segment_id},
                {"endpoint_point_index", entry.endpoint_point_index},
                {"direction", entry.direction},
                {"seed_parameter", double(entry.seed_parameter)},
                {"endpoint_parameter", double(entry.endpoint_parameter)},
                {"endpoint_parameter_tangent", double(entry.endpoint_parameter_tangent)},
                {"attempted_step", double(entry.attempted_step)},
                {"retry_count", entry.retry_count},
                {"accepted_points", entry.accepted_points},
                {"failure", continuation::to_string(entry.failure)},
                {"state_file", entry.state_file},
                {"tangent_file", entry.tangent_file},
                {"message", entry.message},
                {"status", to_string(entry.status)},
                {"connection_id", entry.connection_id},
                {"recovery_attempts", entry.recovery_attempts},
                {"last_strategy", entry.last_strategy},
                {"last_result", entry.last_result}
            });
        }
        write_json_atomic(resolve(settings_.registry_file), root);
    }

private:
    entry_type* find(const std::uint64_t id)
    {
        const auto item = std::find_if(
            entries_.begin(),
            entries_.end(),
            [id](const entry_type& entry)
            {
                return entry.id == id;
            });
        return item == entries_.end() ? nullptr : &*item;
    }

    const entry_type* find(const std::uint64_t id) const
    {
        const auto item = std::find_if(
            entries_.begin(),
            entries_.end(),
            [id](const entry_type& entry)
            {
                return entry.id == id;
            });
        return item == entries_.end() ? nullptr : &*item;
    }

    bool update_attempt_result(
        const std::uint64_t recovery_id,
        const recovery_task_status status,
        const std::string& result)
    {
        auto* entry = find(recovery_id);
        if(entry == nullptr)
        {
            return false;
        }
        entry->status = status;
        entry->last_result = result;
        save();
        return true;
    }

    static recovery_task_status status_from_string(const std::string& value)
    {
        for(const auto status: {
                recovery_task_status::pending,
                recovery_task_status::running,
                recovery_task_status::recovered,
                recovery_task_status::recovered_by_connection,
                recovery_task_status::exhausted,
                recovery_task_status::superseded})
        {
            if(value == to_string(status))
            {
                return status;
            }
        }
        return recovery_task_status::pending;
    }

    static continuation::continuation_failure_kind failure_from_string(
        const std::string& value)
    {
        using kind = continuation::continuation_failure_kind;
        for(const kind candidate: {
                kind::none,
                kind::initial_tangent,
                kind::predictor_chart,
                kind::corrector_retry_limit,
                kind::minimum_step,
                kind::invalid_number,
                kind::linear_solver,
                kind::tangent,
                kind::no_progress,
                kind::maximum_steps,
                kind::unresolved_intersection,
                kind::unknown})
        {
            if(value == continuation::to_string(candidate))
            {
                return candidate;
            }
        }
        return kind::unknown;
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
            std::error_code remove_error;
            std::filesystem::remove(source, remove_error);
            throw std::runtime_error(
                "failed to commit recovery registry file " +
                destination.string() + ": " + error.message());
        }
    }

    void write_vector_atomic(
        const std::filesystem::path& destination,
        const vector_type& vector) const
    {
        const std::filesystem::path temporary =
            destination.string() + ".tmp";
        file_operations_->write_vector(temporary.string(), vector);
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
                "failed to create recovery registry directory: " +
                directory_error.message());
        }
        const std::filesystem::path temporary =
            destination.string() + ".tmp";
        std::ofstream stream(temporary, std::ofstream::trunc);
        if(!stream)
        {
            throw std::runtime_error(
                "failed to open recovery registry temporary file");
        }
        stream << root.dump(4) << '\n';
        stream.close();
        if(!stream)
        {
            throw std::runtime_error("failed to write recovery registry");
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
        if(root.value("policy_generation", std::uint64_t(0)) !=
           settings_.policy_generation)
        {
            return;
        }
        for(const auto& stored:
            root.value("entries", nlohmann::json::array()))
        {
            entry_type entry;
            entry.id = stored.value("id", std::uint64_t(0));
            entry.curve_number = stored.value("curve_number", -1);
            entry.segment_id = stored.value("segment_id", std::uint64_t(0));
            entry.endpoint_point_index = stored.value(
                "endpoint_point_index",
                std::uint64_t(0));
            entry.direction = stored.value("direction", 0);
            entry.seed_parameter = scalar_type(
                stored.value("seed_parameter", 0.0));
            entry.endpoint_parameter = scalar_type(
                stored.value("endpoint_parameter", 0.0));
            entry.endpoint_parameter_tangent = scalar_type(
                stored.value("endpoint_parameter_tangent", 0.0));
            entry.attempted_step = scalar_type(
                stored.value("attempted_step", 0.0));
            entry.retry_count = stored.value("retry_count", 0u);
            entry.accepted_points = stored.value("accepted_points", 0u);
            entry.failure = failure_from_string(
                stored.value("failure", std::string("unknown")));
            entry.state_file = stored.value("state_file", std::string{});
            entry.tangent_file = stored.value("tangent_file", std::string{});
            entry.message = stored.value("message", std::string{});
            entry.status = status_from_string(
                stored.value("status", std::string("pending")));
            entry.connection_id = stored.value(
                "connection_id",
                std::uint64_t(0));
            entry.recovery_attempts = stored.value(
                "recovery_attempts",
                0u);
            entry.last_strategy = stored.value(
                "last_strategy",
                std::string{});
            entry.last_result = stored.value(
                "last_result",
                std::string{});
            next_id_ = std::max(next_id_, entry.id + 1);
            entries_.push_back(std::move(entry));
        }
    }

    VectorFileOperations* file_operations_;
    settings settings_;
    std::vector<entry_type> entries_;
    std::uint64_t next_id_ = 1;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_RECOVERY_REGISTRY_H__
