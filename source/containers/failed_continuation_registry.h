#ifndef __CONTAINERS_FAILED_CONTINUATION_REGISTRY_H__
#define __CONTAINERS_FAILED_CONTINUATION_REGISTRY_H__

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>
#include <continuation/continuation_result.h>

namespace container
{

template<class VectorOperations, class VectorFileOperations>
class failed_continuation_registry
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using distance_function = std::function<scalar_type(
        const vector_type&,
        const vector_type&)>;

    struct settings
    {
        bool enabled = false;
        std::filesystem::path project_directory;
        std::string registry_file = "failed_continuations/registry.json";
        std::string state_directory = "failed_continuations/states";
        std::uint64_t policy_generation = 1;
        std::string symmetry_fingerprint;
        std::string continuation_fingerprint;
    };

    struct entry
    {
        std::uint64_t id = 0;
        scalar_type requested_parameter = scalar_type(0);
        scalar_type effective_parameter = scalar_type(0);
        std::string seed_file;
        continuation::continuation_failure_kind failure =
            continuation::continuation_failure_kind::unknown;
        scalar_type last_parameter = scalar_type(0);
        scalar_type attempted_step = scalar_type(0);
        unsigned int retry_count = 0;
        unsigned int accepted_points = 0;
        int direction = 0;
        std::uint64_t occurrences = 1;
        bool active = true;
        std::string message;
    };

    failed_continuation_registry(
        VectorOperations* vector_operations,
        VectorFileOperations* file_operations,
        settings settings_,
        distance_function distance):
        vector_operations_(vector_operations),
        file_operations_(file_operations),
        settings_(std::move(settings_)),
        distance_(std::move(distance))
    {
        if(vector_operations_ == nullptr || file_operations_ == nullptr)
        {
            throw std::invalid_argument(
                "failed_continuation_registry requires vector and file operations");
        }
        vector_operations_->init_vector(difference_);
        vector_operations_->start_use_vector(difference_);
        load();
    }

    ~failed_continuation_registry()
    {
        clear_vectors();
        vector_operations_->stop_use_vector(difference_);
        vector_operations_->free_vector(difference_);
    }

    failed_continuation_registry(const failed_continuation_registry&) = delete;
    failed_continuation_registry& operator=(
        const failed_continuation_registry&) = delete;

    bool enabled() const
    {
        return settings_.enabled;
    }

    std::size_t size() const
    {
        return entries_.size();
    }

    const std::vector<entry>& entries() const
    {
        return entries_;
    }

    bool nearest_active(
        const scalar_type parameter,
        const vector_type& value,
        scalar_type& distance,
        std::uint64_t& id) const
    {
        bool found = false;
        distance = std::numeric_limits<scalar_type>::infinity();
        id = 0;
        for(std::size_t index = 0; index < entries_.size(); ++index)
        {
            const auto& item = entries_[index];
            if(!item.active ||
               !same_parameter(parameter, item.effective_parameter))
            {
                continue;
            }
            const scalar_type current = distance_between(
                stored_vectors_[index],
                value);
            if(current < distance)
            {
                distance = current;
                id = item.id;
                found = true;
            }
        }
        return found;
    }

    std::uint64_t record(
        const scalar_type requested_parameter,
        const scalar_type effective_parameter,
        const vector_type& seed,
        const continuation::continuation_curve_result<scalar_type>& result)
    {
        if(!enabled())
        {
            return 0;
        }

        const std::uint64_t id = next_id_++;
        const std::filesystem::path state_directory =
            resolve(settings_.state_directory);
        std::error_code directory_error;
        std::filesystem::create_directories(
            state_directory,
            directory_error);
        if(directory_error)
        {
            throw std::runtime_error(
                "failed to create failed-continuation state directory: " +
                directory_error.message());
        }

        const std::string relative_state =
            settings_.state_directory + "/" + std::to_string(id) +
            "_seed.dat";
        const std::filesystem::path destination = resolve(relative_state);
        const std::filesystem::path temporary =
            destination.string() + ".tmp";
        file_operations_->write_vector(temporary.string(), seed);
        replace_file(temporary, destination);

        entry item;
        item.id = id;
        item.requested_parameter = requested_parameter;
        item.effective_parameter = effective_parameter;
        item.seed_file = relative_state;
        populate_failure(result, item);

        vector_type stored;
        vector_operations_->init_vector(stored);
        vector_operations_->start_use_vector(stored);
        vector_operations_->assign(seed, stored);
        entries_.push_back(std::move(item));
        stored_vectors_.push_back(std::move(stored));
        save();
        return id;
    }

    bool mark_seen(const std::uint64_t id)
    {
        for(auto& item: entries_)
        {
            if(item.id == id)
            {
                ++item.occurrences;
                save();
                return true;
            }
        }
        return false;
    }

    bool mark_resolved(const std::uint64_t id)
    {
        for(auto& item: entries_)
        {
            if(item.id == id)
            {
                item.active = false;
                save();
                return true;
            }
        }
        return false;
    }

    void save() const
    {
        if(!enabled())
        {
            return;
        }
        const std::filesystem::path destination =
            resolve(settings_.registry_file);
        std::error_code directory_error;
        std::filesystem::create_directories(
            destination.parent_path(),
            directory_error);
        if(directory_error)
        {
            throw std::runtime_error(
                "failed to create failed-continuation registry directory: " +
                directory_error.message());
        }

        nlohmann::json root;
        root["version"] = 1;
        root["policy_generation"] = settings_.policy_generation;
        root["symmetry_fingerprint"] = settings_.symmetry_fingerprint;
        root["continuation_fingerprint"] =
            settings_.continuation_fingerprint;
        root["entries"] = nlohmann::json::array();
        for(const auto& item: entries_)
        {
            root["entries"].push_back({
                {"id", item.id},
                {"requested_parameter", double(item.requested_parameter)},
                {"effective_parameter", double(item.effective_parameter)},
                {"seed_file", item.seed_file},
                {"failure", continuation::to_string(item.failure)},
                {"last_parameter", double(item.last_parameter)},
                {"attempted_step", double(item.attempted_step)},
                {"retry_count", item.retry_count},
                {"accepted_points", item.accepted_points},
                {"direction", item.direction},
                {"occurrences", item.occurrences},
                {"active", item.active},
                {"message", item.message}
            });
        }

        const std::filesystem::path temporary =
            destination.string() + ".tmp";
        std::ofstream stream(temporary, std::ofstream::trunc);
        if(!stream)
        {
            throw std::runtime_error(
                "failed to open failed-continuation registry temporary file");
        }
        stream << root.dump(4) << '\n';
        stream.close();
        if(!stream)
        {
            std::error_code remove_error;
            std::filesystem::remove(temporary, remove_error);
            throw std::runtime_error(
                "failed to write failed-continuation registry");
        }
        replace_file(temporary, destination);
    }

private:
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

    void populate_failure(
        const continuation::continuation_curve_result<scalar_type>& result,
        entry& item) const
    {
        for(unsigned int index = 0;
            index < result.semicurves_started;
            ++index)
        {
            const auto& semicurve = result.semicurves[index];
            if(semicurve.failure ==
               continuation::continuation_failure_kind::none)
            {
                continue;
            }
            item.failure = semicurve.failure;
            item.last_parameter = semicurve.last_parameter;
            item.attempted_step = semicurve.attempted_step;
            item.retry_count = semicurve.retry_count;
            item.accepted_points = semicurve.accepted_points;
            item.direction = semicurve.direction;
            item.message = semicurve.message;
            return;
        }
    }

    scalar_type distance_between(
        const vector_type& left,
        const vector_type& right) const
    {
        if(distance_)
        {
            return distance_(left, right);
        }
        vector_operations_->assign_mul(
            scalar_type(1),
            left,
            scalar_type(-1),
            right,
            difference_);
        return vector_operations_->norm_l2(difference_);
    }

    std::filesystem::path resolve(const std::string& value) const
    {
        const std::filesystem::path path(value);
        return path.is_absolute()
            ? path
            : settings_.project_directory/path;
    }

    static scalar_type abs_value(const scalar_type value)
    {
        return value < scalar_type(0) ? -value : value;
    }

    static bool same_parameter(
        const scalar_type left,
        const scalar_type right)
    {
        const scalar_type scale = std::max<scalar_type>(
            scalar_type(1),
            std::max<scalar_type>(abs_value(left), abs_value(right)));
        return abs_value(left-right) <=
            scalar_type(64)*
            std::numeric_limits<scalar_type>::epsilon()*scale;
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
                "failed to commit file " + destination.string() + ": " +
                error.message());
        }
    }

    bool compatible(const nlohmann::json& root) const
    {
        return root.value("policy_generation", std::uint64_t(0)) ==
                settings_.policy_generation &&
            root.value("symmetry_fingerprint", std::string{}) ==
                settings_.symmetry_fingerprint &&
            root.value("continuation_fingerprint", std::string{}) ==
                settings_.continuation_fingerprint;
    }

    void load()
    {
        if(!enabled())
        {
            return;
        }
        const std::filesystem::path registry =
            resolve(settings_.registry_file);
        std::ifstream stream(registry);
        if(!stream)
        {
            return;
        }
        nlohmann::json root;
        stream >> root;
        if(!compatible(root))
        {
            return;
        }

        for(const auto& stored:
            root.value("entries", nlohmann::json::array()))
        {
            entry item;
            item.id = stored.value("id", std::uint64_t(0));
            item.requested_parameter = scalar_type(
                stored.value("requested_parameter", 0.0));
            item.effective_parameter = scalar_type(
                stored.value("effective_parameter", 0.0));
            item.seed_file = stored.value("seed_file", std::string{});
            item.failure = failure_from_string(
                stored.value("failure", std::string("unknown")));
            item.last_parameter = scalar_type(
                stored.value("last_parameter", 0.0));
            item.attempted_step = scalar_type(
                stored.value("attempted_step", 0.0));
            item.retry_count = stored.value("retry_count", 0u);
            item.accepted_points = stored.value("accepted_points", 0u);
            item.direction = stored.value("direction", 0);
            item.occurrences = stored.value(
                "occurrences",
                std::uint64_t(1));
            item.active = stored.value("active", true);
            item.message = stored.value("message", std::string{});
            if(item.seed_file.empty() ||
               !std::filesystem::is_regular_file(resolve(item.seed_file)))
            {
                throw std::runtime_error(
                    "failed-continuation registry references a missing seed file: " +
                    item.seed_file);
            }

            vector_type seed;
            vector_operations_->init_vector(seed);
            vector_operations_->start_use_vector(seed);
            file_operations_->read_vector(
                resolve(item.seed_file).string(),
                seed);
            next_id_ = std::max(next_id_, item.id + 1);
            entries_.push_back(std::move(item));
            stored_vectors_.push_back(std::move(seed));
        }
    }

    void clear_vectors()
    {
        for(auto& value: stored_vectors_)
        {
            vector_operations_->stop_use_vector(value);
            vector_operations_->free_vector(value);
        }
        stored_vectors_.clear();
    }

    VectorOperations* vector_operations_;
    VectorFileOperations* file_operations_;
    settings settings_;
    distance_function distance_;
    std::vector<entry> entries_;
    std::vector<vector_type> stored_vectors_;
    mutable vector_type difference_;
    std::uint64_t next_id_ = 0;
};

} // namespace container

#endif // __CONTAINERS_FAILED_CONTINUATION_REGISTRY_H__
