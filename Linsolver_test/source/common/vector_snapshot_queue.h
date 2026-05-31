#ifndef __COMMON_VECTOR_SNAPSHOT_QUEUE_H__
#define __COMMON_VECTOR_SNAPSHOT_QUEUE_H__

#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <vector>

namespace common
{

template<class VectorOperations>
class vector_snapshot_queue
{
public:
    using vector_operations_type = VectorOperations;
    using vector_type = typename VectorOperations::vector_type;

    vector_snapshot_queue(VectorOperations* vec_ops, std::size_t max_elements)
        : vec_ops_(vec_ops),
          max_elements_(max_elements),
          storage_(max_elements)
    {
        if(vec_ops_ == nullptr)
        {
            throw std::logic_error("vector_snapshot_queue: VectorOperations pointer is null");
        }
        if(max_elements_ == 0)
        {
            throw std::logic_error("vector_snapshot_queue: max_elements must be positive");
        }

        for(auto& x : storage_)
        {
            vec_ops_->init_vector(x);
            vec_ops_->start_use_vector(x);
        }
    }

    ~vector_snapshot_queue()
    {
        for(auto& x : storage_)
        {
            vec_ops_->stop_use_vector(x);
            vec_ops_->free_vector(x);
        }
    }

    vector_snapshot_queue(const vector_snapshot_queue&) = delete;
    vector_snapshot_queue& operator=(const vector_snapshot_queue&) = delete;

    void clear()
    {
        order_.clear();
    }

    bool is_queue_filled() const
    {
        return order_.size() == max_elements_;
    }

    std::size_t size() const
    {
        return order_.size();
    }

    const vector_type& at(std::size_t j) const
    {
        return storage_.at(order_.at(j));
    }

    vector_type& at(std::size_t j)
    {
        return storage_.at(order_.at(j));
    }

    void push(const vector_type& x)
    {
        std::size_t slot = 0;
        if(order_.size() == max_elements_)
        {
            slot = order_.front();
            order_.pop_front();
        }
        else
        {
            slot = first_free_slot();
        }

        vec_ops_->assign(x, storage_.at(slot));
        order_.push_back(slot);
    }

private:
    std::size_t first_free_slot() const
    {
        for(std::size_t i = 0; i < max_elements_; ++i)
        {
            if(std::find(order_.begin(), order_.end(), i) == order_.end())
            {
                return i;
            }
        }
        throw std::logic_error("vector_snapshot_queue: no free slot found");
    }

    VectorOperations* vec_ops_;
    std::size_t max_elements_;
    std::vector<vector_type> storage_;
    std::deque<std::size_t> order_;
};

}

#endif
