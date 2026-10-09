#ifndef NMFD_TIME_STEPPERS_STATUS_H
#define NMFD_TIME_STEPPERS_STATUS_H
namespace nmfd
{
namespace time_steppers
{
enum class adaptation_status
{
    accepted,
    accepted_modified,
    rejected,
    failed
};
enum class single_step_status
{
    converged,
    failed_minimum_dt,
    failed_nonfinite,
    failed_solver,
    error_estimate_unavailable,
    attempt_limit_reached,
    step_size_underflow
};
enum class integration_status
{
    running,
    completed,
    stopped_by_external_operation,
    step_failure,
    external_operation_failure,
    attempt_limit_reached
};
namespace detail
{
struct no_external_operations
{
};
}
}
}
#endif
