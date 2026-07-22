#ifndef __CONTINUATION_SEMICURVE_TANGENT_CACHE_H__
#define __CONTINUATION_SEMICURVE_TANGENT_CACHE_H__

namespace continuation
{

template<class VectorOperations>
class semicurve_tangent_cache
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit semicurve_tangent_cache(VectorOperations* vec_ops_):
        vec_ops(vec_ops_)
    {
        vec_ops->init_vector(tangent);
        vec_ops->start_use_vector(tangent);
    }

    ~semicurve_tangent_cache()
    {
        vec_ops->stop_use_vector(tangent);
        vec_ops->free_vector(tangent);
    }

    semicurve_tangent_cache(const semicurve_tangent_cache&) = delete;
    semicurve_tangent_cache& operator=(const semicurve_tangent_cache&) = delete;

    void clear()
    {
        valid_ = false;
    }

    bool valid() const
    {
        return valid_;
    }

    int stored_direction() const
    {
        return direction_;
    }

    void store(
        const vector_type& source,
        const scalar_type lambda_component,
        const int direction)
    {
        vec_ops->assign(source, tangent);
        lambda_component_ = lambda_component;
        direction_ = direction;
        valid_ = true;
    }

    bool restore(
        const int requested_direction,
        vector_type& destination,
        scalar_type& lambda_component) const
    {
        if(!valid_)
        {
            return false;
        }
        vec_ops->assign(tangent, destination);
        lambda_component = lambda_component_;
        const bool same_direction = requested_direction == direction_;
        if(!same_direction)
        {
            vec_ops->scale(scalar_type(-1), destination);
            lambda_component = -lambda_component;
        }
        return true;
    }

private:
    VectorOperations* vec_ops;
    vector_type tangent;
    scalar_type lambda_component_ = scalar_type(0);
    int direction_ = 0;
    bool valid_ = false;
};

} // namespace continuation

#endif
