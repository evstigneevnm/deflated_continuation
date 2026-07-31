#ifndef __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_SOLVER_H__
#define __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_SOLVER_H__

#include <memory>
#include <utility>
#include <vector>

#include <stability/eigensolvers/matrix_free_factorized_krylov_schur.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>

#include "eigensolver_adapter.h"

namespace stability
{
namespace analysis
{

/**
 * Owns the host small-dense backend and the physical-spectrum adapter for a
 * matrix-free factor solver bundle. The large vectors remain on the bundle's
 * configured SCFD backend; only projected Hessenberg/Schur work is delegated
 * to SmallDenseLapack.
 */
template<class FactorSolverBundle, class SmallDenseLapack>
class matrix_free_stability_solver
{
public:
    using factor_bundle_type = FactorSolverBundle;
    using dense_lapack_type = SmallDenseLapack;
    using eigensolver_type =
        eigensolvers::matrix_free_factorized_krylov_schur<
            factor_bundle_type,
            dense_lapack_type>;
    using adapter_type =
        matrix_free_factorized_eigensolver_adapter<
            eigensolver_type>;
    using options_type = typename adapter_type::options_type;
    using vector_type = typename adapter_type::vector_type;
    using real_type = typename adapter_type::real_type;
    using result_type = typename adapter_type::result_type;
    using recovered_vector_storage_type =
        eigensolvers::ritz_vector_storage<
            typename eigensolver_type::real_space_type>;

    explicit matrix_free_stability_solver(
        const factor_bundle_type& factor_bundle,
        options_type options = {})
        : dense_lapack_(),
          eigensolver_(factor_bundle, dense_lapack_),
          adapter_(&eigensolver_, std::move(options))
    {
    }

    matrix_free_stability_solver(
        const matrix_free_stability_solver&) = delete;
    matrix_free_stability_solver& operator=(
        const matrix_free_stability_solver&) = delete;
    matrix_free_stability_solver(
        matrix_free_stability_solver&&) = delete;
    matrix_free_stability_solver& operator=(
        matrix_free_stability_solver&&) = delete;

    result_type execute(const vector_type& initial_vector) const
    {
        return adapter_.execute(initial_vector);
    }

    result_type execute(
        const vector_type& initial_vector,
        recovered_vector_storage_type* recovered_vectors) const
    {
        return adapter_type::flatten(
            eigensolver_.execute(
                initial_vector,
                adapter_.options(),
                recovered_vectors));
    }

    void set_options(options_type options)
    {
        adapter_.set_options(std::move(options));
    }

    const options_type& options() const
    {
        return adapter_.options();
    }

    const eigensolver_type& eigensolver() const
    {
        return eigensolver_;
    }

    const dense_lapack_type& dense_lapack() const
    {
        return dense_lapack_;
    }

private:
    dense_lapack_type dense_lapack_;
    eigensolver_type eigensolver_;
    adapter_type adapter_;
};

/**
 * Complete reusable assembly from a real Jacobian operator and a real affine
 * inverse provider to the structured stability eigensolver interface.
 *
 * RealOperator and Provider may be ordinary or symmetry-projected. Complex
 * vectors, complex affine factors, inner solvers, and physical-spectrum
 * recovery remain internal to the eigensolver stack.
 */
template<
    class FactorizationTypes,
    class InnerLinearSolver,
    class SmallDenseLapack>
class matrix_free_stability_assembly
{
public:
    using factorization_types = FactorizationTypes;
    using inner_solver_type = InnerLinearSolver;
    using dense_lapack_type = SmallDenseLapack;
    using real_space_type =
        typename factorization_types::real_space_type;
    using complex_space_type =
        typename factorization_types::complex_space_type;
    using real_operator_type =
        typename factorization_types::real_operator_type;
    using provider_type =
        typename factorization_types::provider_type;
    using factor_bundle_type =
        eigensolvers::transformations::
            matrix_free_complex_factor_solver_bundle<
                factorization_types,
                inner_solver_type>;
    using solver_type =
        matrix_free_stability_solver<
            factor_bundle_type,
            dense_lapack_type>;
    using options_type = typename solver_type::options_type;
    using vector_type = typename solver_type::vector_type;
    using real_type = typename solver_type::real_type;
    using result_type = typename solver_type::result_type;
    using factor_type = typename factor_bundle_type::factor_type;
    using inner_parameters_type =
        typename factor_bundle_type::solver_parameters_type;
    using log_type = typename factor_bundle_type::log_type;
    using recovered_vector_storage_type =
        typename solver_type::recovered_vector_storage_type;

    matrix_free_stability_assembly(
        std::shared_ptr<real_space_type> real_space,
        std::shared_ptr<complex_space_type> complex_space,
        const real_operator_type& real_operator,
        std::shared_ptr<const provider_type> provider,
        std::vector<factor_type> factors,
        const inner_parameters_type& inner_parameters,
        options_type options = {},
        log_type* log = nullptr)
        : factor_bundle_(
              std::move(real_space),
              std::move(complex_space),
              real_operator,
              std::move(provider),
              std::move(factors),
              inner_parameters,
              log),
          solver_(factor_bundle_, std::move(options))
    {
    }

    matrix_free_stability_assembly(
        const matrix_free_stability_assembly&) = delete;
    matrix_free_stability_assembly& operator=(
        const matrix_free_stability_assembly&) = delete;
    matrix_free_stability_assembly(
        matrix_free_stability_assembly&&) = delete;
    matrix_free_stability_assembly& operator=(
        matrix_free_stability_assembly&&) = delete;

    result_type execute(const vector_type& initial_vector) const
    {
        return solver_.execute(initial_vector);
    }

    result_type execute(
        const vector_type& initial_vector,
        recovered_vector_storage_type* recovered_vectors) const
    {
        return solver_.execute(
            initial_vector,
            recovered_vectors);
    }

    void set_options(options_type options)
    {
        solver_.set_options(std::move(options));
    }

    const options_type& options() const
    {
        return solver_.options();
    }

    factor_bundle_type& factor_bundle()
    {
        return factor_bundle_;
    }

    const factor_bundle_type& factor_bundle() const
    {
        return factor_bundle_;
    }

    const solver_type& solver() const
    {
        return solver_;
    }

private:
    factor_bundle_type factor_bundle_;
    solver_type solver_;
};

} // namespace analysis
} // namespace stability

#endif
