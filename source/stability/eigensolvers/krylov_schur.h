#ifndef __STABILITY_EIGENSOLVERS_KRYLOV_SCHUR_H__
#define __STABILITY_EIGENSOLVERS_KRYLOV_SCHUR_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/arnoldi.h>
#include <nmfd/solvers/krylov/basis_storage.h>
#include <nmfd/solvers/krylov/operator_apply.h>

#include "detail/ordered_real_schur.h"
#include "eigensolver_result.h"
#include "eigenvalue_target.h"
#include "ritz_recovery.h"

namespace stability
{
namespace eigensolvers
{

template<class Real>
struct krylov_schur_options
{
    std::size_t desired_eigenvalues = 6;
    std::size_t krylov_dimension = 24;
    std::size_t restart_dimension = 0;
    std::size_t max_restarts = 100;
    Real absolute_tolerance = Real{};
    Real relative_tolerance = Real(1.0e-8);
    bool preserve_conjugate_pairs = true;
    eigenvalue_target<Real> target;
    nmfd::solvers::krylov::orthogonalization_options<Real>
        orthogonalization;
};

template<class VectorSpace, class SmallDenseLapack>
class krylov_schur
{
public:
    using vector_space_type = VectorSpace;
    using dense_lapack_type = SmallDenseLapack;
    using scalar_type = typename vector_space_type::scalar_type;
    using norm_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using multivector_type = typename vector_space_type::multivector_type;
    using ordinal_type = typename vector_space_type::ordinal_type;
    using matrix_type = typename dense_lapack_type::matrix_type;
    using result_type = eigensolver_result<norm_type>;
    using options_type = krylov_schur_options<norm_type>;

    static_assert(
        std::is_floating_point<scalar_type>::value,
        "real Krylov-Schur requires a real floating-point vector space");

    krylov_schur(
        const vector_space_type& vector_space,
        const dense_lapack_type& dense_lapack)
        : vector_space_(vector_space),
          dense_lapack_(dense_lapack)
    {
    }

    template<class ApplyOperation>
    result_type execute(
        ApplyOperation&& apply_operation,
        const vector_type& initial_vector,
        const options_type& options = {},
        ritz_vector_storage<vector_space_type>* recovered_vectors = nullptr)
        const
    {
        result_type result;
        const std::string invalid_reason = validate_options(
            options,
            recovered_vectors);
        if(!invalid_reason.empty())
        {
            result.status = eigensolver_status::invalid_input;
            result.diagnostic = invalid_reason;
            return result;
        }

        const std::size_t dimension = options.krylov_dimension;
        const std::size_t basis_capacity = dimension + 1;
        nmfd::solvers::krylov::basis_storage<vector_space_type> basis_a(
            vector_space_,
            basis_capacity);
        nmfd::solvers::krylov::basis_storage<vector_space_type> basis_b(
            vector_space_,
            basis_capacity);
        multivector_type* current_basis = &basis_a.data();
        multivector_type* restart_basis = &basis_b.data();

        nmfd::detail::vector_wrap<vector_space_type, true, true> basis_vector(
            vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true> candidate(
            vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true> combination(
            vector_space_);

        matrix_type projected(dimension, dimension);
        matrix_type arnoldi_matrix(basis_capacity, dimension);
        fill_zero(arnoldi_matrix);
        std::vector<scalar_type> coefficients(basis_capacity, scalar_type{});
        std::vector<scalar_type> pass_coefficients(
            basis_capacity,
            scalar_type{});

        auto apply = [&apply_operation](
                         const vector_type& source,
                         vector_type& destination)
        {
            return nmfd::solvers::krylov::invoke_status(
                apply_operation,
                source,
                destination);
        };

        auto factorization =
            nmfd::solvers::krylov::build_arnoldi_factorization(
                vector_space_,
                apply,
                initial_vector,
                *current_basis,
                basis_capacity,
                dimension,
                *basis_vector,
                *candidate,
                coefficients.data(),
                pass_coefficients.data(),
                arnoldi_matrix,
                options.orthogonalization);
        result.operator_calls += factorization.operator_calls;
        result.iterations += factorization.completed_steps;
        if(!factorization.succeeded())
        {
            set_arnoldi_failure(result, factorization.status);
            return result;
        }

        std::size_t active_dimension = factorization.completed_steps;
        bool happy_breakdown =
            factorization.status ==
            nmfd::solvers::krylov::arnoldi_status::happy_breakdown;
        norm_type residual_factor =
            factorization.last_step.subdiagonal;

        for(std::size_t cycle = 0;; ++cycle)
        {
            if(active_dimension == 0)
            {
                result.status = eigensolver_status::numerical_breakdown;
                result.diagnostic = "Krylov-Schur produced an empty subspace";
                return result;
            }
            if(active_dimension < options.desired_eigenvalues)
            {
                result.status = eigensolver_status::numerical_breakdown;
                result.diagnostic =
                    "Arnoldi invariant subspace is smaller than the requested "
                    "eigenspace";
                return result;
            }

            copy_projected_matrix(
                arnoldi_matrix,
                active_dimension,
                projected);

            try
            {
                const auto projected_eigensystem =
                    dense_lapack_.eigensystem(projected);
                const auto ordered = options.target.ordered_indices(
                    projected_eigensystem.eigenvalues);
                const auto selected = selected_indices(
                    projected_eigensystem.eigenvalues,
                    ordered,
                    options.desired_eigenvalues,
                    options.preserve_conjugate_pairs);

                result.eigenpairs = make_estimates(
                    projected_eigensystem,
                    selected,
                    active_dimension,
                    residual_factor,
                    projected,
                    options);
                result.effective_subspace_dimension = active_dimension;

                const bool converged = std::all_of(
                    result.eigenpairs.begin(),
                    result.eigenpairs.end(),
                    [](const eigenpair_estimate<norm_type>& estimate)
                    {
                        return estimate.converged;
                    });
                if(converged || happy_breakdown)
                {
                    if(recovered_vectors != nullptr)
                    {
                        recover_ritz_vectors(
                            vector_space_,
                            *current_basis,
                            basis_capacity,
                            active_dimension,
                            projected_eigensystem,
                            selected,
                            *recovered_vectors);
                    }
                    if(converged)
                    {
                        result.status = eigensolver_status::success;
                        return result;
                    }

                    result.status = eigensolver_status::numerical_breakdown;
                    result.diagnostic =
                        "Arnoldi breakdown occurred before requested Ritz "
                        "pairs met their tolerances";
                    return result;
                }
            }
            catch(const std::exception& error)
            {
                result.status = eigensolver_status::dense_solver_failure;
                result.diagnostic = error.what();
                return result;
            }

            if(cycle >= options.max_restarts)
            {
                result.status = eigensolver_status::no_convergence;
                result.diagnostic =
                    "Krylov-Schur reached its restart limit";
                return result;
            }

            std::size_t retained = 0;
            try
            {
                auto schur = dense_lapack_.schur(projected);
                retained = detail::order_leading_real_schur_blocks<norm_type>(
                    dense_lapack_,
                    schur,
                    options.target,
                    requested_restart_dimension(options));
                if(retained >= dimension)
                    throw std::runtime_error(
                        "Krylov-Schur restart retained the full subspace");

                transform_restart_basis(
                    *current_basis,
                    *restart_basis,
                    basis_capacity,
                    active_dimension,
                    retained,
                    schur.orthogonal_vectors,
                    *combination);

                fill_zero(arnoldi_matrix);
                for(std::size_t col = 0; col < retained; ++col)
                {
                    for(std::size_t row = 0; row < retained; ++row)
                    {
                        arnoldi_matrix(row, col) =
                            schur.quasi_triangular(row, col);
                    }
                    arnoldi_matrix(retained, col) =
                        static_cast<scalar_type>(residual_factor)*
                        schur.orthogonal_vectors(
                            active_dimension - 1,
                            col);
                }
            }
            catch(const std::exception& error)
            {
                result.status = eigensolver_status::dense_solver_failure;
                result.diagnostic = error.what();
                return result;
            }

            std::swap(current_basis, restart_basis);
            happy_breakdown = false;
            active_dimension = retained;
            for(std::size_t step = retained; step < dimension; ++step)
            {
                const auto step_result =
                    nmfd::solvers::krylov::arnoldi_step(
                        vector_space_,
                        apply,
                        *current_basis,
                        basis_capacity,
                        step,
                        *basis_vector,
                        *candidate,
                        coefficients.data(),
                        pass_coefficients.data(),
                        arnoldi_matrix,
                        options.orthogonalization);
                ++result.operator_calls;
                if(!step_result.succeeded())
                {
                    set_arnoldi_failure(result, step_result.status);
                    return result;
                }

                ++result.iterations;
                active_dimension = step + 1;
                residual_factor = step_result.subdiagonal;
                if(
                    step_result.status ==
                    nmfd::solvers::krylov::arnoldi_status::happy_breakdown)
                {
                    happy_breakdown = true;
                    break;
                }
            }
            ++result.restarts;
        }
    }

private:
    static void fill_zero(matrix_type& matrix)
    {
        for(std::size_t col = 0; col < matrix.cols(); ++col)
            for(std::size_t row = 0; row < matrix.rows(); ++row)
                matrix(row, col) = scalar_type{};
    }

    static std::string validate_options(
        const options_type& options,
        const ritz_vector_storage<vector_space_type>* recovered_vectors)
    {
        if(options.desired_eigenvalues == 0)
            return "Krylov-Schur requires at least one eigenvalue";
        if(
            options.krylov_dimension <
            options.desired_eigenvalues + 2)
        {
            return "Krylov dimension must leave room for a restart residual";
        }
        if(
            options.restart_dimension != 0 &&
            (options.restart_dimension < options.desired_eigenvalues ||
             options.restart_dimension > options.krylov_dimension - 2))
        {
            return "invalid Krylov-Schur restart dimension";
        }
        if(
            !(options.absolute_tolerance >= norm_type{}) ||
            !(options.relative_tolerance >= norm_type{}))
        {
            return "Krylov-Schur tolerances must be nonnegative";
        }
        if(
            recovered_vectors != nullptr &&
            recovered_vectors->capacity() <
                options.desired_eigenvalues +
                    (options.preserve_conjugate_pairs ? 1 : 0))
        {
            return "Ritz-vector output capacity is too small";
        }
        return {};
    }

    static std::size_t requested_restart_dimension(
        const options_type& options)
    {
        if(options.restart_dimension != 0)
            return options.restart_dimension;
        const std::size_t available =
            options.krylov_dimension - options.desired_eigenvalues;
        return std::min(
            options.krylov_dimension - 2,
            options.desired_eigenvalues +
                std::max(std::size_t(1), available/2));
    }

    static void copy_projected_matrix(
        const matrix_type& source,
        std::size_t dimension,
        matrix_type& destination)
    {
        destination.resize(dimension, dimension);
        for(std::size_t col = 0; col < dimension; ++col)
            for(std::size_t row = 0; row < dimension; ++row)
                destination(row, col) = source(row, col);
    }

    static std::vector<std::size_t> selected_indices(
        const std::vector<std::complex<norm_type>>& eigenvalues,
        const std::vector<std::size_t>& ordered,
        std::size_t requested,
        bool preserve_conjugate_pairs)
    {
        const std::size_t count = std::min(requested, ordered.size());
        std::vector<bool> selected(eigenvalues.size(), false);
        for(std::size_t index = 0; index < count; ++index)
            selected[ordered[index]] = true;

        if(preserve_conjugate_pairs)
        {
            const norm_type tolerance =
                norm_type(256)*std::numeric_limits<norm_type>::epsilon();
            for(std::size_t index = 0; index < eigenvalues.size(); ++index)
            {
                if(
                    !selected[index] ||
                    std::abs(eigenvalues[index].imag()) <= tolerance)
                {
                    continue;
                }

                const auto conjugate = std::conj(eigenvalues[index]);
                std::size_t best = eigenvalues.size();
                norm_type best_error =
                    std::numeric_limits<norm_type>::infinity();
                for(std::size_t candidate = 0;
                    candidate < eigenvalues.size();
                    ++candidate)
                {
                    if(candidate == index)
                        continue;
                    const norm_type error =
                        std::abs(eigenvalues[candidate] - conjugate);
                    if(error < best_error)
                    {
                        best_error = error;
                        best = candidate;
                    }
                }
                const norm_type scale =
                    std::max(norm_type(1), std::abs(conjugate));
                if(best < eigenvalues.size() &&
                   best_error <= tolerance*scale)
                {
                    selected[best] = true;
                }
            }
        }

        std::vector<std::size_t> result;
        for(const std::size_t index : ordered)
            if(selected[index])
                result.push_back(index);
        return result;
    }

    template<class Eigensystem>
    static std::vector<eigenpair_estimate<norm_type>> make_estimates(
        const Eigensystem& eigensystem,
        const std::vector<std::size_t>& selected,
        std::size_t dimension,
        norm_type residual_factor,
        const matrix_type& projected,
        const options_type& options)
    {
        norm_type projected_norm_sq = norm_type{};
        for(std::size_t col = 0; col < dimension; ++col)
            for(std::size_t row = 0; row < dimension; ++row)
                projected_norm_sq +=
                    projected(row, col)*projected(row, col);
        const norm_type projected_scale = std::sqrt(projected_norm_sq);

        std::vector<eigenpair_estimate<norm_type>> result;
        result.reserve(selected.size());
        for(const std::size_t index : selected)
        {
            eigenpair_estimate<norm_type> estimate;
            estimate.value = eigensystem.eigenvalues[index];
            estimate.residual =
                std::abs(residual_factor)*
                std::abs(
                    eigensystem.right_eigenvectors(
                        dimension - 1,
                        index));
            estimate.converged =
                estimate.residual <=
                options.absolute_tolerance +
                    options.relative_tolerance*
                        std::max(
                            std::abs(estimate.value),
                            projected_scale);
            estimate.projected_index = index;
            result.emplace_back(estimate);
        }
        return result;
    }

    void transform_restart_basis(
        const multivector_type& source,
        multivector_type& destination,
        std::size_t basis_capacity,
        std::size_t active_dimension,
        std::size_t retained_dimension,
        const matrix_type& orthogonal_vectors,
        vector_type& combination) const
    {
        for(std::size_t output = 0;
            output < retained_dimension;
            ++output)
        {
            vector_space_.assign_scalar(scalar_type{}, combination);
            for(std::size_t input = 0;
                input < active_dimension;
                ++input)
            {
                vector_space_.add_lin_comb(
                    orthogonal_vectors(input, output),
                    source,
                    static_cast<ordinal_type>(basis_capacity),
                    static_cast<ordinal_type>(input),
                    scalar_type(1),
                    combination);
            }
            vector_space_.assign(
                combination,
                destination,
                static_cast<ordinal_type>(basis_capacity),
                static_cast<ordinal_type>(output));
        }

        vector_space_.assign(
            source,
            static_cast<ordinal_type>(basis_capacity),
            static_cast<ordinal_type>(active_dimension),
            combination);
        vector_space_.assign(
            combination,
            destination,
            static_cast<ordinal_type>(basis_capacity),
            static_cast<ordinal_type>(retained_dimension));
    }

    static void set_arnoldi_failure(
        result_type& result,
        nmfd::solvers::krylov::arnoldi_status status)
    {
        if(status == nmfd::solvers::krylov::arnoldi_status::operator_failure)
            result.status = eigensolver_status::operator_failure;
        else if(
            status ==
                nmfd::solvers::krylov::arnoldi_status::invalid_input ||
            status ==
                nmfd::solvers::krylov::arnoldi_status::invalid_initial_vector ||
            status ==
                nmfd::solvers::krylov::arnoldi_status::capacity_exceeded)
            result.status = eigensolver_status::invalid_input;
        else
            result.status = eigensolver_status::numerical_breakdown;
        result.diagnostic =
            std::string("Arnoldi failure: ") +
            nmfd::solvers::krylov::arnoldi_status_name(status);
    }

    const vector_space_type& vector_space_;
    const dense_lapack_type& dense_lapack_;
};

} // namespace eigensolvers
} // namespace stability

#endif
