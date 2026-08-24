#ifndef __STABILITY_EIGENSOLVERS_EIGENSOLVER_RESULT_H__
#define __STABILITY_EIGENSOLVERS_EIGENSOLVER_RESULT_H__

#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace stability
{
namespace eigensolvers
{

enum class eigensolver_status
{
    success,
    no_convergence,
    operator_failure,
    inner_solver_failure,
    numerical_breakdown,
    invalid_input,
    dense_solver_failure
};

inline const char* eigensolver_status_name(eigensolver_status status)
{
    switch(status)
    {
    case eigensolver_status::success:
        return "success";
    case eigensolver_status::no_convergence:
        return "no_convergence";
    case eigensolver_status::operator_failure:
        return "operator_failure";
    case eigensolver_status::inner_solver_failure:
        return "inner_solver_failure";
    case eigensolver_status::numerical_breakdown:
        return "numerical_breakdown";
    case eigensolver_status::invalid_input:
        return "invalid_input";
    case eigensolver_status::dense_solver_failure:
        return "dense_solver_failure";
    }
    return "unknown";
}

template<class Real>
struct eigenpair_estimate
{
    std::complex<Real> value{};
    Real residual = std::numeric_limits<Real>::quiet_NaN();
    Real relative_residual = std::numeric_limits<Real>::quiet_NaN();
    bool converged = false;
    std::size_t projected_index = 0;
};

template<class Real>
struct eigensolver_result
{
    eigensolver_status status = eigensolver_status::invalid_input;
    std::vector<eigenpair_estimate<Real>> eigenpairs;
    std::size_t iterations = 0;
    std::size_t restarts = 0;
    std::size_t operator_calls = 0;
    std::size_t inner_solver_calls = 0;
    std::size_t coverage_recoveries = 0;
    std::size_t effective_subspace_dimension = 0;
    std::size_t scans_requested = 1;
    std::size_t scans_succeeded = 0;
    bool coverage_complete = true;
    std::string diagnostic;

    bool succeeded() const
    {
        return status == eigensolver_status::success;
    }
};

} // namespace eigensolvers
} // namespace stability

#endif
