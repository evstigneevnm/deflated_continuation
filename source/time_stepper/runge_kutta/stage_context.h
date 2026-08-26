#ifndef TIME_STEPPER_RUNGE_KUTTA_STAGE_CONTEXT_H
#define TIME_STEPPER_RUNGE_KUTTA_STAGE_CONTEXT_H

#include <cstddef>

namespace time_steppers
{
namespace runge_kutta
{

template<class Scalar>
struct stage_context
{
    using scalar_type = Scalar;

    scalar_type step_start_time{};
    scalar_type stage_time{};
    scalar_type step_size{};
    std::size_t stage_index = 0;
    std::size_t stage_count = 0;
};

} // namespace runge_kutta
} // namespace time_steppers

#endif
