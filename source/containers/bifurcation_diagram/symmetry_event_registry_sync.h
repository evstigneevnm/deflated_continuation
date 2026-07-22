#ifndef __BIFURCATION_DIAGRAM_SYMMETRY_EVENT_REGISTRY_SYNC_H__
#define __BIFURCATION_DIAGRAM_SYMMETRY_EVENT_REGISTRY_SYNC_H__

#include <cstddef>
#include <exception>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>

namespace container
{

struct symmetry_event_registry_sync_result
{
    bool attempted = false;
    bool registry_loaded = false;
    bool saved = false;
    std::size_t previous_event_count = 0;
    std::size_t event_count = 0;
};

template<class VectorOperations, class VectorFileOperations, class Log>
class symmetry_event_registry_sync
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    symmetry_event_registry_sync(
        VectorOperations* vector_operations,
        VectorFileOperations* file_operations,
        Log* log):
        vector_operations_(vector_operations),
        file_operations_(file_operations),
        log_(log),
        left_(vector_operations),
        right_(vector_operations),
        difference_(vector_operations)
    {
    }

    symmetry_event_registry_sync(const symmetry_event_registry_sync&) = delete;
    symmetry_event_registry_sync& operator=(const symmetry_event_registry_sync&) = delete;

    template<class Registry, class Records, class Stabilize>
    symmetry_event_registry_sync_result synchronize(
        Registry& registry,
        const Records& records,
        const std::filesystem::path& project_directory,
        const scalar_type& lambda_tolerance,
        const scalar_type& state_tolerance,
        Stabilize&& stabilize)
    {
        symmetry_event_registry_sync_result result;
        std::error_code directory_error;
        if(!std::filesystem::is_directory(project_directory, directory_error))
        {
            return result;
        }

        result.registry_loaded = registry.load();
        result.previous_event_count = result.registry_loaded
            ? registry.all().size()
            : std::size_t(0);
        if(records.empty() && !result.registry_loaded)
        {
            return result;
        }
        result.attempted = true;

        const auto state_distance =
            [this, &project_directory, &stabilize](
                const typename Registry::record_type& left_record,
                const typename Registry::record_type& right_record) -> scalar_type
            {
                try
                {
                    const std::filesystem::path left_path =
                        project_directory/
                        std::to_string(left_record.curve_number)/
                        std::to_string(left_record.vector_file_id);
                    const std::filesystem::path right_path =
                        project_directory/
                        std::to_string(right_record.curve_number)/
                        std::to_string(right_record.vector_file_id);
                    file_operations_->read_vector(left_path.string(), left_.get());
                    file_operations_->read_vector(right_path.string(), right_.get());
                    stabilize(left_.get());
                    stabilize(right_.get());
                    vector_operations_->assign_mul(
                        scalar_type(1),
                        left_.get(),
                        scalar_type(-1),
                        right_.get(),
                        difference_.get());
                    return vector_operations_->norm_l2(difference_.get());
                }
                catch(const std::exception& error)
                {
                    if(log_ != nullptr)
                    {
                        log_->warning_f(
                            "symmetry_event_registry_sync: failed to compare event states: %s",
                            error.what());
                    }
                    return std::numeric_limits<scalar_type>::infinity();
                }
            };

        registry.rebuild(
            records,
            lambda_tolerance,
            state_tolerance,
            state_distance);
        result.event_count = registry.all().size();
        result.saved = registry.save();
        return result;
    }

private:
    class owned_vector
    {
    public:
        explicit owned_vector(VectorOperations* vector_operations):
            vector_operations_(vector_operations)
        {
            vector_operations_->init_vector(value_);
            vector_operations_->start_use_vector(value_);
        }

        ~owned_vector()
        {
            vector_operations_->stop_use_vector(value_);
            vector_operations_->free_vector(value_);
        }

        owned_vector(const owned_vector&) = delete;
        owned_vector& operator=(const owned_vector&) = delete;

        vector_type& get()
        {
            return value_;
        }

    private:
        VectorOperations* vector_operations_;
        vector_type value_;
    };

    VectorOperations* vector_operations_;
    VectorFileOperations* file_operations_;
    Log* log_;
    owned_vector left_;
    owned_vector right_;
    owned_vector difference_;
};

} // namespace container

#endif // __BIFURCATION_DIAGRAM_SYMMETRY_EVENT_REGISTRY_SYNC_H__
