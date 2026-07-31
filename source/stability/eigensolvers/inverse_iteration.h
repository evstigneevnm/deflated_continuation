#ifndef __STABILITY_EIGENSOLVERS_INVERSE_ITERATION_H__
#define __STABILITY_EIGENSOLVERS_INVERSE_ITERATION_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

#include "eigensolver_result.h"

namespace stability
{
namespace eigensolvers
{

template<class Real>
struct inverse_iteration_options
{
    std::size_t max_iterations = 100;
    Real absolute_tolerance = Real{};
    Real relative_tolerance =
        Real(100)*std::sqrt(std::numeric_limits<Real>::epsilon());
    Real minimum_vector_norm =
        Real(64)*std::numeric_limits<Real>::min();
    std::complex<Real> shift{};
};

template<class VectorSpace>
class inverse_iteration
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using norm_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using result_type = eigensolver_result<norm_type>;
    using options_type = inverse_iteration_options<norm_type>;

    static_assert(
        std::is_floating_point<scalar_type>::value,
        "real inverse iteration requires a real floating-point vector space");

    explicit inverse_iteration(const vector_space_type& vector_space)
        : vector_space_(vector_space)
    {
    }

    template<class ApplyOperation, class ShiftedSolveOperation>
    result_type execute(
        ApplyOperation&& apply_original,
        ShiftedSolveOperation&& solve_shifted,
        const vector_type& initial_vector,
        vector_type& eigenvector,
        const options_type& options = {}) const
    {
        result_type result;
        if(
            options.max_iterations == 0 ||
            !(options.relative_tolerance >= norm_type{}) ||
            !(options.absolute_tolerance >= norm_type{}) ||
            !(options.minimum_vector_norm > norm_type{}))
        {
            result.status = eigensolver_status::invalid_input;
            result.diagnostic = "invalid inverse-iteration options";
            return result;
        }

        nmfd::detail::vector_wrap<vector_space_type, true, true> current(
            vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true> next(
            vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true> applied(
            vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true> residual(
            vector_space_);

        vector_space_.assign(initial_vector, *current);
        const norm_type initial_norm = vector_space_.norm(*current);
        if(
            !std::isfinite(initial_norm) ||
            !(initial_norm > options.minimum_vector_norm))
        {
            result.status = eigensolver_status::invalid_input;
            result.diagnostic =
                "inverse iteration received a zero or non-finite initial vector";
            return result;
        }
        vector_space_.scale(
            scalar_type(norm_type(1)/initial_norm),
            *current);

        eigenpair_estimate<norm_type> estimate;
        for(std::size_t iteration = 0;
            iteration < options.max_iterations;
            ++iteration)
        {
            vector_space_.assign_scalar(scalar_type{}, *next);
            ++result.inner_solver_calls;
            if(!nmfd::solvers::krylov::invoke_status(
                solve_shifted,
                static_cast<const vector_type&>(*current),
                *next))
            {
                result.status = eigensolver_status::inner_solver_failure;
                result.iterations = iteration;
                result.diagnostic =
                    "shifted linear solve failed during inverse iteration";
                return result;
            }

            const norm_type solved_norm = vector_space_.norm(*next);
            if(
                !std::isfinite(solved_norm) ||
                !(solved_norm > options.minimum_vector_norm))
            {
                result.status = eigensolver_status::numerical_breakdown;
                result.iterations = iteration + 1;
                result.diagnostic =
                    "shifted solve produced a zero or non-finite vector";
                return result;
            }
            vector_space_.scale(
                scalar_type(norm_type(1)/solved_norm),
                *next);

            if(vector_space_.scalar_prod(*current, *next) < scalar_type{})
                vector_space_.scale(scalar_type(-1), *next);

            ++result.operator_calls;
            if(!nmfd::solvers::krylov::invoke_status(
                apply_original,
                static_cast<const vector_type&>(*next),
                *applied))
            {
                result.status = eigensolver_status::operator_failure;
                result.iterations = iteration + 1;
                result.diagnostic =
                    "original operator failed during inverse iteration";
                return result;
            }

            const scalar_type denominator =
                vector_space_.scalar_prod(*next, *next);
            if(
                !std::isfinite(denominator) ||
                !(std::abs(denominator) >
                    static_cast<scalar_type>(options.minimum_vector_norm)))
            {
                result.status = eigensolver_status::numerical_breakdown;
                result.iterations = iteration + 1;
                result.diagnostic = "invalid Rayleigh-quotient denominator";
                return result;
            }
            const scalar_type rayleigh =
                vector_space_.scalar_prod(*next, *applied)/denominator;

            vector_space_.assign(*applied, *residual);
            vector_space_.add_lin_comb(
                -rayleigh,
                *next,
                scalar_type(1),
                *residual);
            const norm_type residual_norm = vector_space_.norm(*residual);
            const norm_type applied_norm = vector_space_.norm(*applied);
            if(
                !std::isfinite(rayleigh) ||
                !std::isfinite(residual_norm) ||
                !std::isfinite(applied_norm))
            {
                result.status = eigensolver_status::numerical_breakdown;
                result.iterations = iteration + 1;
                result.diagnostic =
                    "non-finite Rayleigh quotient or residual";
                return result;
            }

            estimate.value = std::complex<norm_type>(
                static_cast<norm_type>(rayleigh),
                norm_type{});
            estimate.residual = residual_norm;
            estimate.converged =
                residual_norm <=
                options.absolute_tolerance +
                    options.relative_tolerance*
                        std::max(
                            applied_norm,
                            static_cast<norm_type>(std::abs(rayleigh)));
            result.iterations = iteration + 1;
            vector_space_.assign(*next, *current);

            if(estimate.converged)
            {
                vector_space_.assign(*current, eigenvector);
                result.status = eigensolver_status::success;
                result.eigenpairs = {estimate};
                result.effective_subspace_dimension = 1;
                return result;
            }
        }

        vector_space_.assign(*current, eigenvector);
        result.status = eigensolver_status::no_convergence;
        result.eigenpairs = {estimate};
        result.effective_subspace_dimension = 1;
        result.diagnostic = "inverse iteration reached its iteration limit";
        return result;
    }

private:
    const vector_space_type& vector_space_;
};

} // namespace eigensolvers
} // namespace stability

#endif
