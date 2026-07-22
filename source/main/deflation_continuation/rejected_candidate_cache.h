#ifndef __MAIN_DEFLATION_CONTINUATION_REJECTED_CANDIDATE_CACHE_H__
#define __MAIN_DEFLATION_CONTINUATION_REJECTED_CANDIDATE_CACHE_H__

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace main_classes
{
namespace deflation_continuation_detail
{

template<class VectorOperations>
class rejected_candidate_cache
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit rejected_candidate_cache(VectorOperations* vector_operations):
        vector_operations_(vector_operations)
    {
        vector_operations_->init_vector(difference_);
        vector_operations_->start_use_vector(difference_);
    }

    ~rejected_candidate_cache()
    {
        clear();
        vector_operations_->stop_use_vector(difference_);
        vector_operations_->free_vector(difference_);
    }

    rejected_candidate_cache(const rejected_candidate_cache&) = delete;
    rejected_candidate_cache& operator=(const rejected_candidate_cache&) = delete;

    void clear()
    {
        for(auto& value: rejected_vectors_)
        {
            vector_operations_->stop_use_vector(value);
            vector_operations_->free_vector(value);
        }
        rejected_vectors_.clear();
        parameter_values_.clear();
    }

    void add(const scalar_type& parameter, const vector_type& value)
    {
        vector_type copy;
        vector_operations_->init_vector(copy);
        vector_operations_->start_use_vector(copy);
        vector_operations_->assign(value, copy);
        rejected_vectors_.push_back(std::move(copy));
        parameter_values_.push_back(parameter);
    }

    bool nearest_distance(
        const scalar_type& parameter,
        const vector_type& value,
        scalar_type& distance)
    {
        bool found = false;
        distance = std::numeric_limits<scalar_type>::infinity();
        for(std::size_t index = 0; index < rejected_vectors_.size(); ++index)
        {
            if(!same_parameter(parameter, parameter_values_[index]))
            {
                continue;
            }
            vector_operations_->assign_mul(
                scalar_type(1),
                value,
                scalar_type(-1),
                rejected_vectors_[index],
                difference_);
            const scalar_type current_distance =
                vector_operations_->norm_l2(difference_);
            if(current_distance < distance)
            {
                distance = current_distance;
                found = true;
            }
        }
        return found;
    }

    std::size_t size() const
    {
        return rejected_vectors_.size();
    }

private:
    static scalar_type abs_value(const scalar_type& value)
    {
        return value < scalar_type(0) ? -value : value;
    }

    static bool same_parameter(
        const scalar_type& left,
        const scalar_type& right)
    {
        const scalar_type scale = std::max<scalar_type>(
            scalar_type(1),
            std::max<scalar_type>(abs_value(left), abs_value(right)));
        return abs_value(left - right) <=
            scalar_type(64)*std::numeric_limits<scalar_type>::epsilon()*scale;
    }

    VectorOperations* vector_operations_;
    std::vector<vector_type> rejected_vectors_;
    std::vector<scalar_type> parameter_values_;
    vector_type difference_;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_REJECTED_CANDIDATE_CACHE_H__
