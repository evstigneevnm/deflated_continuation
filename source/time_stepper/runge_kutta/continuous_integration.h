#ifndef NMFD_TIME_STEPPERS_RK_CONTINUOUS_INTEGRATION_H
#define NMFD_TIME_STEPPERS_RK_CONTINUOUS_INTEGRATION_H

#include <cstddef>
#include <nmfd/detail/vector_wrap.h>

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{
namespace detail
{
template<class VectorOperations, bool Enabled>
struct dense_storage
{
    dense_storage(VectorOperations&, bool)
    {
    }
};

template<class VectorOperations>
struct dense_storage<VectorOperations, true>
{
    nmfd::detail::vector_wrap<VectorOperations> previous, endpoint_rate;

    dense_storage(VectorOperations& operations, bool hermite):
        previous(operations), endpoint_rate(operations, hermite)
    {
        previous.start_use();
        if (hermite)
        {
            endpoint_rate.start_use();
        }
    }
};
}

// Borrowed view of one pending step. The stepper must outlive the view.
template<class SingleStepMethod>
class continuous_integration
{
public:
    using scalar_type = typename SingleStepMethod::scalar_type;
    using vector_type = typename SingleStepMethod::vector_type;

    scalar_type evaluate(scalar_type theta, vector_type& out) const
    {
        return owner_->evaluate_dense(theta, out, generation_);
    }

private:
    friend SingleStepMethod;

    continuous_integration(const SingleStepMethod& owner, std::size_t generation):
        owner_(&owner), generation_(generation)
    {
    }

    const SingleStepMethod* owner_;
    std::size_t generation_;
};
}
}
}
#endif
