#ifndef __STABILITY_EIGENSOLVERS_DIRECT_SCALAR_EIGENSOLVER_H__
#define __STABILITY_EIGENSOLVERS_DIRECT_SCALAR_EIGENSOLVER_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <stability/analysis/detail/vector_workspace.h>

#include "eigensolver_result.h"

namespace stability
{
namespace eigensolvers
{

/**
 * Exact eigensolver for a one-dimensional real linear operator.
 *
 * This is intentionally separate from Krylov-Schur. Scalar continuation
 * problems exercise stability traversal and transition bookkeeping without
 * introducing an artificial Krylov space.
 */
template<class VectorSpace, class LinearOperator>
class direct_scalar_eigensolver
{
public:
    using vector_space_type = VectorSpace;
    using operator_type = LinearOperator;
    using scalar_type = typename vector_space_type::scalar_type;
    using real_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using result_type = eigensolver_result<real_type>;

    direct_scalar_eigensolver(
        vector_space_type& vector_space,
        const operator_type& linear_operator)
        : vector_space_(vector_space),
          linear_operator_(linear_operator),
          basis_(&vector_space_),
          image_(&vector_space_),
          residual_(&vector_space_)
    {
        if(vector_space_.get_default_size() != 1)
        {
            throw std::invalid_argument(
                "direct scalar eigensolver requires vector dimension one");
        }
    }

    result_type execute(const vector_type&) const
    {
        result_type result;
        result.scans_requested = 1;
        result.effective_subspace_dimension = 1;

        vector_space_.assign_scalar(scalar_type(1), basis_.get());
        if(!apply(basis_.get(), image_.get()))
        {
            result.status = eigensolver_status::operator_failure;
            result.coverage_complete = false;
            result.diagnostic =
                "direct scalar eigensolver operator application failed";
            return result;
        }
        result.operator_calls = 1;

        const scalar_type eigenvalue =
            vector_space_.scalar_prod(basis_.get(), image_.get());
        vector_space_.assign(image_.get(), residual_.get());
        vector_space_.add_lin_comb(
            -eigenvalue,
            basis_.get(),
            scalar_type(1),
            residual_.get());
        const real_type residual = vector_space_.norm_l2(residual_.get());
        const real_type relative_residual =
            residual/std::max(
                real_type(1),
                static_cast<real_type>(std::abs(eigenvalue)));

        eigenpair_estimate<real_type> estimate;
        estimate.value = {
            static_cast<real_type>(eigenvalue),
            real_type{}};
        estimate.residual = residual;
        estimate.relative_residual = relative_residual;
        estimate.converged =
            std::isfinite(static_cast<real_type>(eigenvalue)) &&
            std::isfinite(residual);
        estimate.projected_index = 0;
        result.eigenpairs.push_back(estimate);

        result.status = estimate.converged
            ? eigensolver_status::success
            : eigensolver_status::no_convergence;
        result.scans_succeeded = estimate.converged ? 1 : 0;
        result.coverage_complete = estimate.converged;
        if(!estimate.converged)
        {
            result.diagnostic =
                "direct scalar eigensolver produced a non-finite result";
        }
        return result;
    }

private:
    vector_space_type& vector_space_;
    const operator_type& linear_operator_;
    mutable analysis::detail::vector_workspace<vector_space_type> basis_;
    mutable analysis::detail::vector_workspace<vector_space_type> image_;
    mutable analysis::detail::vector_workspace<vector_space_type> residual_;

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        using return_type = decltype(
            linear_operator_.apply(source, destination));
        if constexpr(std::is_void<return_type>::value)
        {
            linear_operator_.apply(source, destination);
            return vector_space_.check_is_valid_number(destination);
        }
        else
        {
            return
                static_cast<bool>(
                    linear_operator_.apply(source, destination)) &&
                vector_space_.check_is_valid_number(destination);
        }
    }
};

} // namespace eigensolvers
} // namespace stability

#endif
