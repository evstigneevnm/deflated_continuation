#ifndef __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_ACTIONS_1D_H__
#define __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_ACTIONS_1D_H__

#include <cstddef>
#include <vector>

#include <symmetry/finite_action_registry.h>

namespace symmetry
{
namespace fourier
{

template<class Registry>
void add_real_packed_negative_reflection_action(Registry& registry)
{
    using vector_operations_type = typename Registry::vector_operations_type;
    using scalar_type = typename vector_operations_type::scalar_type;
    using vector_type = typename vector_operations_type::vector_type;

    auto* vec_ops = registry.vector_operations();
    const auto action = [vec_ops](const vector_type& source, vector_type& destination)
    {
        std::vector<scalar_type> host(vec_ops->get_size(source), scalar_type(0));
        vec_ops->get(source, host.data(), host.size());
        for(std::size_t offset = 0; offset + 1 < host.size(); offset += 2)
        {
            host[offset] = -host[offset];
        }
        vec_ops->set(host.data(), destination, host.size());
    };

    registry.add("real_packed_negative_reflection", action, action);
}

template<class Registry>
void add_sine_half_period_shift_action(Registry& registry)
{
    using vector_operations_type = typename Registry::vector_operations_type;
    using scalar_type = typename vector_operations_type::scalar_type;
    using vector_type = typename vector_operations_type::vector_type;

    auto* vec_ops = registry.vector_operations();
    const auto action = [vec_ops](const vector_type& source, vector_type& destination)
    {
        std::vector<scalar_type> host(vec_ops->get_size(source), scalar_type(0));
        vec_ops->get(source, host.data(), host.size());
        for(std::size_t i = 0; i < host.size(); ++i)
        {
            const std::size_t mode = i + 1;
            if(mode%2 == 1)
            {
                host[i] = -host[i];
            }
        }
        vec_ops->set(host.data(), destination, host.size());
    };

    registry.add("sine_half_period_shift", action, action);
}

} // namespace fourier
} // namespace symmetry

#endif
