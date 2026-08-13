#ifndef __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_SCAN_H__
#define __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_SCAN_H__

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "detail/vector_workspace.h"
#include "eigenvector_rank_aggregator.h"
#include "matrix_free_stability_solver.h"
#include "recycled_ritz_subspace.h"
#include "spectrum_scan_aggregator.h"

namespace stability
{
namespace analysis
{

template<class Factor>
struct matrix_free_stability_scan_attempt
{
    std::string label;
    std::vector<Factor> factors;
};

template<class Factor, class SolverOptions, class Real>
struct matrix_free_stability_scan_definition
{
    std::string label;
    std::vector<Factor> factors;
    std::vector<matrix_free_stability_scan_attempt<Factor>>
        retry_attempts;
    Real preconditioner_pole_absolute_tolerance = Real{};
    Real preconditioner_pole_relative_tolerance = Real{};
    std::vector<unsigned int> inner_basis_retry_sizes;
    SolverOptions solver_options;
};

/**
 * Executes transformed spectral scans sequentially. Only one factor bundle
 * and one set of large Krylov workspaces exist at a time.
 */
template<
    class FactorizationTypes,
    class InnerLinearSolver,
    class SmallDenseLapack>
class matrix_free_stability_scan
{
public:
    using factorization_types = FactorizationTypes;
    using inner_solver_type = InnerLinearSolver;
    using dense_lapack_type = SmallDenseLapack;
    using assembly_type =
        matrix_free_stability_assembly<
            factorization_types,
            inner_solver_type,
            dense_lapack_type>;
    using real_space_type =
        typename assembly_type::real_space_type;
    using complex_space_type =
        typename assembly_type::complex_space_type;
    using real_operator_type =
        typename assembly_type::real_operator_type;
    using provider_type =
        typename assembly_type::provider_type;
    using vector_type = typename assembly_type::vector_type;
    using real_type = typename assembly_type::real_type;
    using result_type = typename assembly_type::result_type;
    using factor_type = typename assembly_type::factor_type;
    using solver_options_type =
        typename assembly_type::options_type;
    using inner_parameters_type =
        typename assembly_type::inner_parameters_type;
    using log_type = typename assembly_type::log_type;
    using scan_attempt_type =
        matrix_free_stability_scan_attempt<factor_type>;
    using scan_definition_type =
        matrix_free_stability_scan_definition<
            factor_type,
            solver_options_type,
            real_type>;
    using aggregation_options_type =
        spectrum_scan_aggregation_options<real_type>;
    using recovered_vector_storage_type =
        typename assembly_type::recovered_vector_storage_type;
    using recycled_subspace_type =
        recycled_ritz_subspace<real_space_type>;
    using recycling_options_type =
        typename recycled_subspace_type::options_type;
    using probe_generator_type =
        std::function<void(
            std::size_t,
            const vector_type&,
            vector_type&)>;

    matrix_free_stability_scan(
        std::shared_ptr<real_space_type> real_space,
        std::shared_ptr<complex_space_type> complex_space,
        const real_operator_type& real_operator,
        std::shared_ptr<const provider_type> provider,
        std::vector<scan_definition_type> scans,
        inner_parameters_type inner_parameters,
        aggregation_options_type aggregation_options = {},
        log_type* log = nullptr,
        probe_generator_type probe_generator = {})
        : real_space_(checked(std::move(real_space), "real space")),
          complex_space_(
              checked(std::move(complex_space), "complex space")),
          real_operator_(real_operator),
          provider_(checked(std::move(provider), "provider")),
          scans_(std::move(scans)),
          inner_parameters_(std::move(inner_parameters)),
          aggregation_options_(std::move(aggregation_options)),
          log_(log),
          probe_generator_(std::move(probe_generator)),
          recycled_subspace_(*real_space_)
    {
        if(scans_.empty())
            throw std::invalid_argument(
                "matrix-free stability scan requires at least one scan");
        for(const auto& scan : scans_)
        {
            if(scan.factors.empty())
                throw std::invalid_argument(
                    "matrix-free stability scan factors are empty");
            unsigned int previous_basis_size =
                inner_parameters_.basis_size;
            for(const auto basis_size :
                scan.inner_basis_retry_sizes)
            {
                if(basis_size <= previous_basis_size)
                {
                    throw std::invalid_argument(
                        "matrix-free stability scan basis retries "
                        "must be strictly increasing");
                }
                previous_basis_size = basis_size;
            }
            for(const auto& attempt : scan.retry_attempts)
            {
                if(attempt.factors.empty())
                    throw std::invalid_argument(
                        "matrix-free stability retry factors are empty");
            }
        }
    }

    result_type execute(const vector_type& initial_vector) const
    {
        spectrum_scan_aggregator<real_type> aggregate(
            scans_.size(),
            aggregation_options_);
        detail::vector_workspace<real_space_type> probe(
            real_space_.get());
        detail::vector_workspace<real_space_type> recycled_probe(
            real_space_.get());

        typename eigenvector_rank_aggregator<
            real_space_type>::options_type rank_options;
        rank_options.eigenvalue_absolute_tolerance =
            aggregation_options_.absolute_tolerance;
        rank_options.eigenvalue_relative_tolerance =
            aggregation_options_.relative_tolerance;
        rank_options.independence_tolerance =
            aggregation_options_.eigenvector_independence_tolerance;
        rank_options.orthogonalization_passes =
            aggregation_options_.eigenvector_orthogonalization_passes;
        eigenvector_rank_aggregator<real_space_type>
            recycle_candidates(*real_space_, rank_options);
        const auto recycle_validation =
            recycled_subspace_.validate(real_operator_);
        std::size_t recycled_probe_seeds = 0;

        const std::size_t required_successful_probes =
            aggregation_options_.require_all_probes
            ? aggregation_options_.probe_count
            : aggregation_options_.minimum_successful_probes;

        for(const auto& scan : scans_)
        {
            eigenvector_rank_aggregator<real_space_type>
                physical_subspace(*real_space_, rank_options);

            result_type scan_result;
            std::size_t successful_probes = 0;
            std::size_t successful_fresh_probes = 0;
            eigensolvers::eigensolver_status first_failure =
                eigensolvers::eigensolver_status::success;
            std::string probe_failures;
            std::string retry_recoveries;

            const std::size_t transformed_capacity =
                scan.solver_options.transformed.
                    desired_eigenvalues +
                (scan.solver_options.transformed.
                     preserve_conjugate_pairs
                     ? 1
                     : 0);
            const std::size_t recovered_capacity =
                4*transformed_capacity;
            for(std::size_t probe_index = 0;
                probe_index < aggregation_options_.probe_count;
                ++probe_index)
            {
                const vector_type* probe_vector =
                    &initial_vector;
                if(probe_index != 0)
                {
                    generate_probe(
                        probe_index,
                        initial_vector,
                        probe.get());
                    probe_vector = &probe.get();
                }
                const bool recycled_seed =
                    probe_index >= required_successful_probes &&
                    recycled_subspace_.make_seed(
                        probe_index,
                        *probe_vector,
                        recycle_validation.accepted_indices,
                        recycled_probe.get());
                if(recycled_seed)
                {
                    probe_vector = &recycled_probe.get();
                    ++recycled_probe_seeds;
                }

                const std::size_t attempt_count =
                    std::size_t(1) +
                    scan.retry_attempts.size();
                result_type terminal_failure;
                std::string attempt_failures;
                bool probe_succeeded = false;
                for(std::size_t attempt_index = 0;
                    attempt_index < attempt_count;
                    ++attempt_index)
                {
                    const bool primary = attempt_index == 0;
                    const auto& factors = primary
                        ? scan.factors
                        : scan.retry_attempts[
                              attempt_index - 1].factors;
                    const std::string attempt_label = primary
                        ? std::string("primary")
                        : scan.retry_attempts[
                              attempt_index - 1].label;
                    const std::size_t basis_attempt_count =
                        std::size_t(1) +
                        scan.inner_basis_retry_sizes.size();
                    for(std::size_t basis_attempt = 0;
                        basis_attempt < basis_attempt_count;
                        ++basis_attempt)
                    {
                        const bool basis_retry =
                            basis_attempt != 0;
                        auto inner_parameters =
                            inner_parameters_;
                        if(basis_retry)
                        {
                            inner_parameters.basis_size =
                                scan.inner_basis_retry_sizes[
                                    basis_attempt - 1];
                        }
                        std::string solver_attempt_label =
                            attempt_label;
                        if(basis_retry)
                        {
                            solver_attempt_label +=
                                " basis=" +
                                std::to_string(
                                    inner_parameters.basis_size);
                        }

                        assembly_type assembly(
                            real_space_,
                            complex_space_,
                            real_operator_,
                            provider_,
                            factors,
                            inner_parameters,
                            scan.solver_options,
                            log_);

                        std::string health_diagnostic;
                        const bool near_pole =
                            preconditioner_near_pole(
                                assembly,
                                scan.
                                    preconditioner_pole_absolute_tolerance,
                                scan.
                                    preconditioner_pole_relative_tolerance,
                                health_diagnostic);
                        if(
                            near_pole &&
                            attempt_index + 1 < attempt_count)
                        {
                            result_type skipped;
                            skipped.status =
                                eigensolvers::eigensolver_status::
                                    no_convergence;
                            skipped.diagnostic =
                                "preconditioner near-pole detected";
                            append_attempt_failure(
                                attempt_failures,
                                solver_attempt_label,
                                skipped,
                                health_diagnostic,
                                {});
                            terminal_failure =
                                std::move(skipped);
                            break;
                        }

                        recovered_vector_storage_type
                            recovered_vectors(
                                *real_space_,
                                recovered_capacity);
                        result_type probe_result =
                            assembly.execute(
                                *probe_vector,
                                &recovered_vectors);
                        accumulate_work(
                            scan_result,
                            probe_result);
                        const std::string solver_diagnostic =
                            factor_solver_diagnostic(
                                assembly,
                                inner_parameters);
                        if(
                            probe_result.succeeded() &&
                            probe_result.coverage_complete)
                        {
                            ++successful_probes;
                            if(!recycled_seed)
                                ++successful_fresh_probes;
                            physical_subspace.add(
                                probe_result,
                                recovered_vectors);
                            recycle_candidates.add(
                                probe_result,
                                recovered_vectors);
                            probe_succeeded = true;
                            if(!primary || basis_retry)
                            {
                                append_retry_recovery(
                                    retry_recoveries,
                                    probe_index,
                                    solver_attempt_label,
                                    attempt_failures,
                                    health_diagnostic,
                                    solver_diagnostic);
                            }
                            break;
                        }

                        terminal_failure = probe_result;
                        append_attempt_failure(
                            attempt_failures,
                            solver_attempt_label,
                            probe_result,
                            health_diagnostic,
                            solver_diagnostic);
                    }
                    if(probe_succeeded)
                        break;
                }

                if(!probe_succeeded)
                {
                    if(
                        first_failure ==
                        eigensolvers::eigensolver_status::success)
                    {
                        first_failure =
                            terminal_failure.status ==
                                eigensolvers::eigensolver_status::
                                    success
                            ? eigensolvers::eigensolver_status::
                                  no_convergence
                            : terminal_failure.status;
                    }
                    append_probe_failure(
                        probe_failures,
                        probe_index,
                        terminal_failure,
                        attempt_failures);
                }
            }

            scan_result.eigenpairs =
                physical_subspace.estimates();
            const bool accepted_probes =
                successful_probes >= required_successful_probes &&
                successful_fresh_probes >=
                    required_successful_probes;
            scan_result.status =
                accepted_probes &&
                    !scan_result.eigenpairs.empty()
                ? eigensolvers::eigensolver_status::success
                : (
                      first_failure !=
                          eigensolvers::eigensolver_status::success
                      ? first_failure
                      : eigensolvers::eigensolver_status::
                            no_convergence);
            scan_result.coverage_complete = accepted_probes;
            scan_result.scans_requested =
                aggregation_options_.probe_count;
            scan_result.scans_succeeded = successful_probes;
            std::ostringstream diagnostic;
            diagnostic
                << "multiplicity probes: "
                << successful_probes << "/"
                << aggregation_options_.probe_count
                << " succeeded (minimum "
                << required_successful_probes << "), "
                << successful_fresh_probes
                << " fresh probes succeeded (minimum "
                << required_successful_probes << "), "
                << scan_result.eigenpairs.size()
                << " independent physical eigenpairs";
            if(!probe_failures.empty())
                diagnostic << "; failures: " << probe_failures;
            if(!retry_recoveries.empty())
                diagnostic
                    << "; recovered by retry: "
                    << retry_recoveries;
            scan_result.diagnostic = diagnostic.str();
            aggregate.add(
                std::move(scan_result),
                scan.label);
        }
        result_type result = aggregate.finish();
        result.operator_calls +=
            recycle_validation.operator_calls;
        if(result.succeeded())
            recycled_subspace_.stage(recycle_candidates);
        if(recycled_subspace_.enabled())
        {
            std::ostringstream recycling;
            recycling
                << "Ritz recycling: committed="
                << recycled_subspace_.size()
                << ", accepted="
                << recycle_validation.accepted_indices.size()
                << ", rejected="
                << recycle_validation.rejected_vectors
                << ", seeded_probes="
                << recycled_probe_seeds;
            result.diagnostic = result.diagnostic.empty()
                ? recycling.str()
                : result.diagnostic + "; " + recycling.str();
        }
        return result;
    }

    std::size_t scan_count() const
    {
        return scans_.size();
    }

    const std::vector<scan_definition_type>& scans() const
    {
        return scans_;
    }

    void set_probe_generator(probe_generator_type probe_generator)
    {
        probe_generator_ = std::move(probe_generator);
    }

    bool has_probe_generator() const
    {
        return static_cast<bool>(probe_generator_);
    }

    void set_recycling_options(recycling_options_type options)
    {
        recycled_subspace_.set_options(std::move(options));
    }

    const recycling_options_type& recycling_options() const
    {
        return recycled_subspace_.options();
    }

    void begin_recycling_transaction() const
    {
        recycled_subspace_.begin_transaction();
    }

    void commit_recycling_transaction() const
    {
        recycled_subspace_.commit_transaction();
    }

    void rollback_recycling_transaction() const
    {
        recycled_subspace_.rollback_transaction();
    }

    void reset_recycled_subspace() const
    {
        recycled_subspace_.clear();
    }

    std::size_t recycled_subspace_size() const
    {
        return recycled_subspace_.size();
    }

private:
    template<class Value>
    static std::shared_ptr<Value> checked(
        std::shared_ptr<Value> value,
        const char* label)
    {
        if(!value)
            throw std::invalid_argument(
                std::string("matrix-free stability scan requires ") +
                label);
        return value;
    }

    std::shared_ptr<real_space_type> real_space_;
    std::shared_ptr<complex_space_type> complex_space_;
    const real_operator_type& real_operator_;
    std::shared_ptr<const provider_type> provider_;
    std::vector<scan_definition_type> scans_;
    inner_parameters_type inner_parameters_;
    aggregation_options_type aggregation_options_;
    log_type* log_;
    probe_generator_type probe_generator_;
    mutable recycled_subspace_type recycled_subspace_;

    void generate_probe(
        std::size_t probe_index,
        const vector_type& initial_vector,
        vector_type& probe) const
    {
        if(probe_generator_)
        {
            probe_generator_(
                probe_index,
                initial_vector,
                probe);
        }
        else
        {
            real_space_->assign_random(probe);
        }
    }

    static void accumulate_work(
        result_type& destination,
        const result_type& source)
    {
        destination.iterations += source.iterations;
        destination.restarts += source.restarts;
        destination.operator_calls += source.operator_calls;
        destination.inner_solver_calls +=
            source.inner_solver_calls;
        destination.effective_subspace_dimension = std::max(
            destination.effective_subspace_dimension,
            source.effective_subspace_dimension);
    }

    static void append_probe_failure(
        std::string& failures,
        std::size_t probe_index,
        const result_type& result,
        const std::string& attempt_failures)
    {
        if(!failures.empty())
            failures += " | ";
        failures +=
            "probe[" + std::to_string(probe_index) + "]: ";
        failures +=
            eigensolvers::eigensolver_status_name(
                result.status ==
                    eigensolvers::eigensolver_status::success
                ? eigensolvers::eigensolver_status::no_convergence
                : result.status);
        if(!result.diagnostic.empty())
            failures += " (" + result.diagnostic + ")";
        if(!attempt_failures.empty())
            failures += " {" + attempt_failures + "}";
    }

    static void append_attempt_failure(
        std::string& failures,
        const std::string& label,
        const result_type& result,
        const std::string& health,
        const std::string& solver)
    {
        if(!failures.empty())
            failures += "; ";
        failures += label + ": ";
        failures += eigensolvers::eigensolver_status_name(
            result.status ==
                eigensolvers::eigensolver_status::success
            ? eigensolvers::eigensolver_status::no_convergence
            : result.status);
        if(!result.diagnostic.empty())
            failures += " (" + result.diagnostic + ")";
        if(!health.empty())
            failures += " [" + health + "]";
        if(!solver.empty())
            failures += " [" + solver + "]";
    }

    static void append_retry_recovery(
        std::string& recoveries,
        std::size_t probe_index,
        const std::string& label,
        const std::string& previous_failures,
        const std::string& health,
        const std::string& solver)
    {
        if(!recoveries.empty())
            recoveries += " | ";
        recoveries +=
            "probe[" + std::to_string(probe_index) +
            "] " + label;
        if(!previous_failures.empty())
            recoveries += " after {" + previous_failures + "}";
        if(!health.empty())
            recoveries += " [" + health + "]";
        if(!solver.empty())
            recoveries += " [" + solver + "]";
    }

    static bool preconditioner_near_pole(
        const assembly_type& assembly,
        real_type absolute_tolerance,
        real_type relative_tolerance,
        std::string& diagnostic)
    {
        bool near_pole = false;
        std::ostringstream stream;
        const auto& bundle = assembly.factor_bundle();
        for(std::size_t index = 0;
            index < bundle.factor_count();
            ++index)
        {
            const auto health =
                bundle.factor(index).
                    preconditioner().health();
            if(!health.available)
                continue;
            if(stream.tellp() > 0)
                stream << ", ";
            stream
                << "factor[" << index << "] min|D|="
                << health.minimum_abs_denominator
                << " max|D|="
                << health.maximum_abs_denominator;
            if(health.relative_denominator_available)
            {
                stream
                    << " min_rel|D|="
                    << health.minimum_relative_denominator;
            }
            if(
                health.near_pole(
                    absolute_tolerance,
                    relative_tolerance))
            {
                near_pole = true;
                stream << " near-pole";
            }
        }
        diagnostic = stream.str();
        return near_pole;
    }

    std::string factor_solver_diagnostic(
        const assembly_type& assembly,
        const inner_parameters_type& inner_parameters) const
    {
        const auto statistics =
            assembly.factor_bundle().statistics();
        std::ostringstream stream;
        stream
            << "inner side="
            << inner_parameters.preconditioner_side
            << " basis="
            << inner_parameters.basis_size;
        for(const auto& factor : statistics.factors)
        {
            stream
                << ", factor[" << factor.index << "] calls="
                << factor.solve_calls
                << " failures=" << factor.failed_solves
                << " max_it=" << factor.maximum_iterations;
            if(factor.last_residual_available)
            {
                stream
                    << (
                           inner_parameters.preconditioner_side ==
                                   'R'
                           ? " true_residual="
                           : " monitored_residual=")
                    << factor.last_residual;
            }
        }
        return stream.str();
    }
};

} // namespace analysis
} // namespace stability

#endif
