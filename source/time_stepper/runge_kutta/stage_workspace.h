#ifndef TIME_STEPPER_RUNGE_KUTTA_STAGE_WORKSPACE_H
#define TIME_STEPPER_RUNGE_KUTTA_STAGE_WORKSPACE_H

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace time_steppers
{
namespace runge_kutta
{

template<class VectorOperations>
class stage_workspace
{
public:
    using vector_operations_type = VectorOperations;
    using vector_type = typename vector_operations_type::vector_type;

    stage_workspace(vector_operations_type& vector_operations, const std::size_t stage_count):
        vector_operations_(&vector_operations),
        stage_rates_(stage_count)
    {
        if(stage_count == 0)
        {
            throw std::invalid_argument("stage_workspace requires at least one Runge-Kutta stage.");
        }

        try
        {
            start(stage_state_);
            for(auto& stage_rate: stage_rates_)
            {
                start(stage_rate);
            }
        }
        catch(...)
        {
            release();
            throw;
        }
    }

    stage_workspace(const stage_workspace&) = delete;
    stage_workspace& operator=(const stage_workspace&) = delete;
    stage_workspace(stage_workspace&&) = delete;
    stage_workspace& operator=(stage_workspace&&) = delete;

    ~stage_workspace()
    {
        release();
    }

    vector_type& stage_state()
    {
        return stage_state_;
    }

    vector_type& stage_rate(const std::size_t stage)
    {
        return stage_rates_.at(stage);
    }

    const vector_type& stage_rate(const std::size_t stage) const
    {
        return stage_rates_.at(stage);
    }

    std::size_t stage_count() const
    {
        return stage_rates_.size();
    }

    std::size_t allocation_count() const
    {
        return allocation_count_;
    }

private:
    void start(vector_type& vector)
    {
        vector_operations_->init_vector(vector);
        try
        {
            vector_operations_->start_use_vector(vector);
        }
        catch(...)
        {
            release_vector(vector);
            throw;
        }
        ++started_vector_count_;
        ++allocation_count_;
    }

    void release_vector(vector_type& vector) noexcept
    {
        try
        {
            vector_operations_->stop_use_vector(vector);
            vector_operations_->free_vector(vector);
        }
        catch(...)
        {
        }
    }

    void release() noexcept
    {
        if(started_vector_count_ == 0)
        {
            return;
        }

        const std::size_t started_rates = started_vector_count_-1;
        for(std::size_t index = 0; index < started_rates; ++index)
        {
            release_vector(stage_rates_[index]);
        }
        release_vector(stage_state_);
        started_vector_count_ = 0;
    }

    vector_operations_type* vector_operations_;
    vector_type stage_state_;
    std::vector<vector_type> stage_rates_;
    std::size_t started_vector_count_ = 0;
    std::size_t allocation_count_ = 0;
};

} // namespace runge_kutta
} // namespace time_steppers

#endif
