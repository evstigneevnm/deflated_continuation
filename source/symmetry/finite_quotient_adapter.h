#ifndef __SYMMETRY_FINITE_QUOTIENT_ADAPTER_H__
#define __SYMMETRY_FINITE_QUOTIENT_ADAPTER_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <symmetry/finite_action_registry.h>

namespace symmetry
{

template<class VectorOperations, class ContinuousAdapter>
class finite_quotient_adapter
{
public:
    using vector_operations_type = VectorOperations;
    using continuous_adapter_type = ContinuousAdapter;
    using finite_action_registry_type = finite_action_registry<VectorOperations>;
    using scalar_type = typename VectorOperations::scalar_type;
    using norm_type = typename VectorOperations::norm_type;
    using vector_type = typename VectorOperations::vector_type;

    finite_quotient_adapter(
        VectorOperations* vec_ops_,
        ContinuousAdapter* continuous_adapter_,
        finite_action_registry_type* finite_actions_):
        vec_ops(vec_ops_),
        continuous_adapter(continuous_adapter_),
        finite_actions(finite_actions_)
    {
        if(vec_ops == nullptr)
        {
            throw std::invalid_argument("finite_quotient_adapter got null vector operations");
        }
        if(continuous_adapter == nullptr)
        {
            throw std::invalid_argument("finite_quotient_adapter got null continuous adapter");
        }
        if(finite_actions == nullptr)
        {
            throw std::invalid_argument("finite_quotient_adapter got null finite action registry");
        }
        if(finite_actions->size() == 0)
        {
            throw std::invalid_argument("finite_quotient_adapter needs at least one finite action");
        }

        vec_ops->init_vector(action_source);
        vec_ops->init_vector(candidate);
        vec_ops->init_vector(best);
        vec_ops->init_vector(action_gradient);
        vec_ops->init_vector(distance);
        vec_ops->start_use_vector(action_source);
        vec_ops->start_use_vector(candidate);
        vec_ops->start_use_vector(best);
        vec_ops->start_use_vector(action_gradient);
        vec_ops->start_use_vector(distance);
    }

    ~finite_quotient_adapter()
    {
        vec_ops->stop_use_vector(distance);
        vec_ops->free_vector(distance);
        vec_ops->stop_use_vector(action_gradient);
        vec_ops->free_vector(action_gradient);
        vec_ops->stop_use_vector(best);
        vec_ops->free_vector(best);
        vec_ops->stop_use_vector(candidate);
        vec_ops->free_vector(candidate);
        vec_ops->stop_use_vector(action_source);
        vec_ops->free_vector(action_source);
    }

    ContinuousAdapter* continuous() const
    {
        return continuous_adapter;
    }

    finite_action_registry_type* finite_registry() const
    {
        return finite_actions;
    }

    std::size_t last_action_index() const
    {
        return last_action_index_;
    }

    const std::string& last_action_name() const
    {
        return finite_actions->name(last_action_index_);
    }

    void stabilize(const vector_type& source, vector_type& destination)
    {
        stabilize_canonical(source, destination);
    }

    void stabilize_canonical(const vector_type& source, vector_type& destination)
    {
        const std::size_t selected_action = select_canonical_action(source, destination);
        finite_actions->apply(selected_action, source, action_source);
        continuous_adapter->stabilize_canonical(action_source, destination);
        last_action_index_ = selected_action;
    }

    void stabilize_closest_to_reference(
        const vector_type& reference,
        const vector_type& source,
        vector_type& destination)
    {
        bool have_best = false;
        norm_type best_distance = norm_type{};
        std::size_t best_index = 0;

        for(std::size_t action_index = 0;
            action_index < finite_actions->size();
            ++action_index)
        {
            finite_actions->apply(
                action_index,
                source,
                action_source);
            continuous_adapter->align_orbit_closest_to_reference(
                reference,
                action_source,
                candidate);
            vec_ops->assign_mul(
                scalar_type(1),
                candidate,
                scalar_type(-1),
                reference,
                distance);
            const norm_type candidate_distance =
                vec_ops->norm_l2(distance);

            if(!have_best || candidate_distance < best_distance)
            {
                have_best = true;
                best_distance = candidate_distance;
                best_index = action_index;
                vec_ops->assign(candidate, best);
            }
        }

        vec_ops->assign(best, destination);
        last_action_index_ = best_index;
    }

    void pullback_distance_gradient(
        const vector_type& source,
        const vector_type& slice_state,
        const vector_type& slice_gradient,
        vector_type& gradient)
    {
        pullback_canonical_distance_gradient(source, slice_state, slice_gradient, gradient);
    }

    void pullback_canonical_distance_gradient(
        const vector_type& source,
        const vector_type& slice_state,
        const vector_type& slice_gradient,
        vector_type& gradient)
    {
        const std::size_t selected_action = select_canonical_action(source, candidate);
        finite_actions->apply(selected_action, source, action_source);
        continuous_adapter->pullback_canonical_distance_gradient(
            action_source,
            slice_state,
            slice_gradient,
            action_gradient);
        finite_actions->pullback(selected_action, action_gradient, gradient);
        last_action_index_ = selected_action;
    }

private:
    std::size_t select_canonical_action(const vector_type& source, vector_type& destination)
    {
        bool have_best = false;
        std::vector<scalar_type> best_host;
        std::size_t best_index = 0;
        const scalar_type tolerance = canonical_zero_tolerance();

        for(std::size_t action_index = 0; action_index < finite_actions->size(); ++action_index)
        {
            finite_actions->apply(action_index, source, action_source);
            continuous_adapter->stabilize_canonical(action_source, candidate);
            auto candidate_host = get_host_state(candidate);
            zero_small_components(candidate_host, tolerance);

            if(!have_best || lexicographically_greater(candidate_host, best_host, tolerance))
            {
                have_best = true;
                best_index = action_index;
                best_host = candidate_host;
                vec_ops->assign(candidate, best);
            }
        }

        vec_ops->assign(best, destination);
        last_action_index_ = best_index;
        return best_index;
    }

    std::vector<scalar_type> get_host_state(const vector_type& x) const
    {
        std::vector<scalar_type> host(vec_ops->get_size(x), scalar_type(0));
        vec_ops->get(x, host.data(), host.size());
        return host;
    }

    scalar_type canonical_zero_tolerance() const
    {
        return scalar_type(64)*std::numeric_limits<scalar_type>::epsilon();
    }

    static void zero_small_components(std::vector<scalar_type>& values, const scalar_type tolerance)
    {
        for(auto& value: values)
        {
            if(std::abs(value) <= tolerance)
            {
                value = scalar_type(0);
            }
        }
    }

    static bool lexicographically_greater(
        const std::vector<scalar_type>& left,
        const std::vector<scalar_type>& right,
        const scalar_type tolerance)
    {
        const std::size_t n = std::min(left.size(), right.size());
        for(std::size_t i = 0; i < n; ++i)
        {
            const scalar_type delta = left[i] - right[i];
            if(delta > tolerance)
            {
                return true;
            }
            if(delta < -tolerance)
            {
                return false;
            }
        }
        return left.size() > right.size();
    }

private:
    VectorOperations* vec_ops;
    ContinuousAdapter* continuous_adapter;
    finite_action_registry_type* finite_actions;
    vector_type action_source;
    vector_type candidate;
    vector_type best;
    vector_type action_gradient;
    vector_type distance;
    std::size_t last_action_index_ = 0;
};

} // namespace symmetry

#endif
