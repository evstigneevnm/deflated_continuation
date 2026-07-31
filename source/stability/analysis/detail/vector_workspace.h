#ifndef __STABILITY_ANALYSIS_DETAIL_VECTOR_WORKSPACE_H__
#define __STABILITY_ANALYSIS_DETAIL_VECTOR_WORKSPACE_H__

#include <stdexcept>

namespace stability
{
namespace analysis
{
namespace detail
{

template<class VectorOperations>
class vector_workspace
{
public:
    using vector_type = typename VectorOperations::vector_type;

    explicit vector_workspace(VectorOperations* vector_operations)
        : vector_operations_(vector_operations)
    {
        if(vector_operations_ == nullptr)
            throw std::invalid_argument(
                "vector_workspace: vector operations are null");
        vector_operations_->init_vector(vector_);
        try
        {
            vector_operations_->start_use_vector(vector_);
        }
        catch(...)
        {
            vector_operations_->free_vector(vector_);
            throw;
        }
    }

    ~vector_workspace()
    {
        vector_operations_->stop_use_vector(vector_);
        vector_operations_->free_vector(vector_);
    }

    vector_workspace(const vector_workspace&) = delete;
    vector_workspace& operator=(const vector_workspace&) = delete;

    vector_type& get()
    {
        return vector_;
    }

    const vector_type& get() const
    {
        return vector_;
    }

private:
    VectorOperations* vector_operations_;
    vector_type vector_{};
};

} // namespace detail
} // namespace analysis
} // namespace stability

#endif
