#ifndef __STABILITY_ANALYSIS_EIGENSOLVER_ADAPTER_H__
#define __STABILITY_ANALYSIS_EIGENSOLVER_ADAPTER_H__

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <stability/eigensolvers/eigensolver_result.h>

namespace stability
{
namespace analysis
{

template<class Real, class Vector, class Callable>
class callable_eigensolver_adapter
{
public:
    using real_type = Real;
    using vector_type = Vector;
    using result_type = eigensolvers::eigensolver_result<real_type>;

    explicit callable_eigensolver_adapter(Callable callable)
        : callable_(std::move(callable))
    {
    }

    result_type execute(const vector_type& initial_vector)
    {
        return callable_(initial_vector);
    }

private:
    Callable callable_;
};

/**
 * Exposes the recovered physical spectrum of
 * matrix_free_factorized_krylov_schur through the common structured
 * eigensolver interface used by stability_evaluator.
 */
template<class MatrixFreeEigensolver>
class matrix_free_factorized_eigensolver_adapter
{
public:
    using eigensolver_type = MatrixFreeEigensolver;
    using vector_type = typename eigensolver_type::real_vector_type;
    using options_type = typename eigensolver_type::options_type;
    using real_type =
        typename eigensolver_type::real_space_type::norm_type;
    using result_type = eigensolvers::eigensolver_result<real_type>;
    using raw_result_type = typename eigensolver_type::result_type;

    matrix_free_factorized_eigensolver_adapter(
        const eigensolver_type* eigensolver,
        options_type options = {})
        : eigensolver_(eigensolver),
          options_(std::move(options))
    {
        if(eigensolver_ == nullptr)
            throw std::invalid_argument(
                "matrix_free_factorized_eigensolver_adapter: "
                "eigensolver is null");
    }

    result_type execute(const vector_type& initial_vector) const
    {
        return flatten(eigensolver_->execute(initial_vector, options_));
    }

    void set_options(options_type options)
    {
        options_ = std::move(options);
    }

    const options_type& options() const
    {
        return options_;
    }

    static result_type flatten(raw_result_type raw)
    {
        result_type result;
        result.iterations = raw.transformed.iterations;
        result.restarts = raw.transformed.restarts;
        result.operator_calls =
            raw.transformed.operator_calls +
            raw.recovered.original_operator_calls;
        result.inner_solver_calls =
            raw.transformed_solver_calls;
        result.effective_subspace_dimension =
            raw.recovered.projection_dimension;
        result.scans_requested = 1;
        result.scans_succeeded = raw.succeeded() ? 1 : 0;
        result.coverage_complete = raw.succeeded();

        if(!raw.transformed.succeeded())
        {
            result.status = raw.transformed.status;
            result.diagnostic = raw.transformed.diagnostic;
            return result;
        }
        if(raw.transformed_solver_failures != 0)
        {
            result.status =
                eigensolvers::eigensolver_status::
                    inner_solver_failure;
            result.diagnostic =
                "matrix-free affine factor solve failed";
            return result;
        }

        result.status = raw.recovered.status;
        result.eigenpairs = std::move(raw.recovered.eigenpairs);
        result.eigenpairs.erase(
            std::remove_if(
                result.eigenpairs.begin(),
                result.eigenpairs.end(),
                [](const auto& estimate)
                {
                    return !estimate.converged;
                }),
            result.eigenpairs.end());
        if(
            result.status ==
                eigensolvers::eigensolver_status::success &&
            result.eigenpairs.size() <
                raw.minimum_converged_physical_eigenpairs)
        {
            result.status =
                eigensolvers::eigensolver_status::no_convergence;
        }
        result.diagnostic = std::move(raw.recovered.diagnostic);
        return result;
    }

private:
    const eigensolver_type* eigensolver_;
    options_type options_;
};

} // namespace analysis
} // namespace stability

#endif
