#ifndef TIME_STEPPER_RUNGE_KUTTA_STEP_RESULT_H
#define TIME_STEPPER_RUNGE_KUTTA_STEP_RESULT_H

#include <cstddef>
#include <limits>

namespace time_steppers
{
namespace runge_kutta
{

enum class step_status
{
    success,
    rate_evaluation_failure
};

struct step_result
{
    static constexpr std::size_t no_stage = std::numeric_limits<std::size_t>::max();

    step_status status = step_status::success;
    std::size_t rate_evaluations = 0;
    std::size_t failed_stage = no_stage;
    bool error_estimate_available = false;

    explicit operator bool() const
    {
        return status == step_status::success;
    }
};

} // namespace runge_kutta
} // namespace time_steppers

#endif
