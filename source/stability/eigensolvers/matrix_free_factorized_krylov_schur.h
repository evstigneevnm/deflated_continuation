#ifndef __STABILITY_EIGENSOLVERS_MATRIX_FREE_FACTORIZED_KRYLOV_SCHUR_H__
#define __STABILITY_EIGENSOLVERS_MATRIX_FREE_FACTORIZED_KRYLOV_SCHUR_H__

#include <cstddef>
#include <sstream>

#include <nmfd/detail/vector_wrap.h>

#include "krylov_schur.h"
#include "rotated_projected_spectrum_recovery.h"
#include "transformations/complex_solver_product_adapter.h"

namespace stability
{
namespace eigensolvers
{

template<class FactorSolverBundle, class SmallDenseLapack>
class matrix_free_factorized_krylov_schur
{
public:
    using factor_bundle_type = FactorSolverBundle;
    using dense_lapack_type = SmallDenseLapack;
    using real_space_type =
        typename factor_bundle_type::real_space_type;
    using complex_space_type =
        typename factor_bundle_type::complex_space_type;
    using product_space_type =
        typename factor_bundle_type::product_space_type;
    using bridge_type = typename factor_bundle_type::bridge_type;
    using real_vector_type = typename real_space_type::vector_type;
    using product_vector_type =
        typename product_space_type::vector_type;
    using transformed_operator_type =
        transformations::complex_solver_product_adapter<
            product_space_type,
            complex_space_type,
            bridge_type,
            factor_bundle_type>;
    using outer_eigensolver_type =
        krylov_schur<
            product_space_type,
            dense_lapack_type>;
    using recovery_type =
        rotated_projected_spectrum_recovery<
            real_space_type,
            product_space_type,
            dense_lapack_type>;

    struct options_type
    {
        typename outer_eigensolver_type::options_type transformed;
        typename recovery_type::options_type recovery;
        std::size_t minimum_converged_physical_eigenpairs = 1;
    };

    struct result_type
    {
        typename outer_eigensolver_type::result_type transformed;
        typename recovery_type::result_type recovered;
        std::size_t transformed_solver_calls = 0;
        std::size_t transformed_solver_failures = 0;
        std::size_t minimum_converged_physical_eigenpairs = 1;

        bool succeeded() const
        {
            return
                transformed.succeeded() &&
                recovered.succeeded() &&
                recovered.converged_eigenpairs >=
                    minimum_converged_physical_eigenpairs;
        }
    };

    matrix_free_factorized_krylov_schur(
        const factor_bundle_type& factor_bundle,
        const dense_lapack_type& dense_lapack)
        : factor_bundle_(factor_bundle),
          transformed_operator_(
              factor_bundle_.product_space(),
              factor_bundle_.complex_space(),
              factor_bundle_.bridge(),
              factor_bundle_),
          outer_eigensolver_(
              factor_bundle_.product_space(),
              dense_lapack),
          recovery_(
              factor_bundle_.real_space(),
              factor_bundle_.product_space(),
              dense_lapack)
    {
    }

    result_type execute(
        const real_vector_type& initial_vector,
        const options_type& options = {},
        ritz_vector_storage<real_space_type>* recovered_vectors = nullptr)
        const
    {
        result_type result;
        result.minimum_converged_physical_eigenpairs =
            options.minimum_converged_physical_eigenpairs;
        if(options.minimum_converged_physical_eigenpairs == 0)
        {
            result.transformed.status =
                eigensolver_status::invalid_input;
            result.transformed.diagnostic =
                "matrix-free physical eigenpair requirement must "
                "be positive";
            return result;
        }
        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true> product_initial(factor_bundle_.product_space());
        factor_bundle_.real_space().assign(
            initial_vector,
            (*product_initial).first);
        factor_bundle_.real_space().assign_scalar(
            typename real_space_type::scalar_type{},
            (*product_initial).second);

        const std::size_t transformed_capacity =
            options.transformed.desired_eigenvalues +
            (options.transformed.preserve_conjugate_pairs ? 1 : 0);
        ritz_vector_storage<product_space_type> transformed_vectors(
            factor_bundle_.product_space(),
            transformed_capacity);

        const std::size_t calls_before =
            transformed_operator_.solve_calls();
        const std::size_t failures_before =
            transformed_operator_.failed_solves();
        result.transformed = outer_eigensolver_.execute(
            [this](
                const product_vector_type& source,
                product_vector_type& destination)
            {
                return transformed_operator_.solve(
                    source,
                    destination);
            },
            *product_initial,
            options.transformed,
            &transformed_vectors);
        result.transformed_solver_calls =
            transformed_operator_.solve_calls() - calls_before;
        result.transformed_solver_failures =
            transformed_operator_.failed_solves() - failures_before;
        result.transformed.inner_solver_calls =
            result.transformed_solver_calls;

        if(result.transformed_solver_failures != 0)
        {
            result.transformed.status =
                eigensolver_status::inner_solver_failure;
            result.transformed.diagnostic =
                "matrix-free affine factor solve failed";
            return result;
        }
        if(!result.transformed.succeeded())
            return result;

        result.recovered = recovery_.execute(
            factor_bundle_.complexified_operator().real_operator(),
            transformed_vectors,
            options.recovery,
            recovered_vectors);
        if(
            result.recovered.succeeded() &&
            result.recovered.converged_eigenpairs <
                options.minimum_converged_physical_eigenpairs)
        {
            std::ostringstream diagnostic;
            diagnostic
                << "physical-spectrum recovery converged "
                << result.recovered.converged_eigenpairs
                << " eigenpairs, but "
                << options.minimum_converged_physical_eigenpairs
                << " are required"
                << "; projected dimension = "
                << result.recovered.projection_dimension
                << "; candidates = "
                << result.recovered.eigenpairs.size();
            append_rejected_physical_ritz_values(
                diagnostic,
                result.recovered);
            result.recovered.status =
                eigensolver_status::no_convergence;
            result.recovered.diagnostic = diagnostic.str();
        }
        return result;
    }

    const transformed_operator_type& transformed_operator() const
    {
        return transformed_operator_;
    }

    const outer_eigensolver_type& outer_eigensolver() const
    {
        return outer_eigensolver_;
    }

    const recovery_type& recovery() const
    {
        return recovery_;
    }

private:
    static void append_rejected_physical_ritz_values(
        std::ostringstream& diagnostic,
        const typename recovery_type::result_type& recovered)
    {
        std::size_t reported = 0;
        for(const auto& estimate : recovered.eigenpairs)
        {
            if(estimate.converged)
                continue;
            if(reported == 0)
                diagnostic << "; rejected physical Ritz values: ";
            else
                diagnostic << ", ";
            diagnostic
                << "(" << estimate.value.real()
                << "," << estimate.value.imag()
                << ") residual=" << estimate.residual
                << " relative=" << estimate.relative_residual;
            ++reported;
            if(reported == 8)
            {
                const std::size_t rejected =
                    recovered.eigenpairs.size() -
                    recovered.converged_eigenpairs;
                if(rejected > reported)
                {
                    diagnostic
                        << ", ... "
                        << (rejected - reported)
                        << " more";
                }
                break;
            }
        }
    }

    const factor_bundle_type& factor_bundle_;
    transformed_operator_type transformed_operator_;
    outer_eigensolver_type outer_eigensolver_;
    recovery_type recovery_;
};

} // namespace eigensolvers
} // namespace stability

#endif
