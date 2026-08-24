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
#include "validated_spectrum_union.h"
#include <stability/tracking/tracked_invariant_subspace.h>

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
    using tracked_subspace_type =
        tracking::tracked_invariant_subspace<real_space_type>;
    using tracking_options_type =
        typename tracked_subspace_type::options_type;
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
          recycled_subspace_(*real_space_),
          tracked_subspace_(*real_space_)
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
        update_confirmation_union_options();
    }

    result_type execute(const vector_type& initial_vector) const
    {
        const auto& tracking = tracked_subspace_.options();
        const std::size_t nominal_seed_limit =
            tracking.maximum_seed_vectors;
        result_type primary = execute_with_tracked_seed_limit(
            initial_vector,
            nominal_seed_limit);
        const std::size_t recovery_seed_limit =
            tracking.coverage_recovery_maximum_seed_vectors;
        const bool aggregate_undercoverage =
            !primary.succeeded() &&
            primary.scans_requested != 0 &&
            primary.scans_succeeded == primary.scans_requested &&
            primary.eigenpairs.size() <
                aggregation_options_.minimum_eigenpairs;
        if(
            !tracking.enabled ||
            !aggregate_undercoverage ||
            recovery_seed_limit <= nominal_seed_limit ||
            tracked_subspace_.seed_size() <= nominal_seed_limit)
        {
            record_confirmation_spectrum(primary);
            return primary;
        }

        result_type recovered = execute_with_tracked_seed_limit(
            initial_vector,
            recovery_seed_limit);
        accumulate_work(recovered, primary);
        ++recovered.coverage_recoveries;
        const std::string recovery_diagnostic =
            std::move(recovered.diagnostic);
        std::ostringstream diagnostic;
        diagnostic
            << "tracked coverage recovery with seed limit "
            << recovery_seed_limit
            << (recovered.succeeded() ? " succeeded" : " failed")
            << " after primary undercoverage {"
            << primary.diagnostic << '}';
        if(!recovery_diagnostic.empty())
            diagnostic << "; recovery {" << recovery_diagnostic << '}';
        recovered.diagnostic = diagnostic.str();
        record_confirmation_spectrum(recovered);
        return recovered;
    }

    bool classification_reconciliation_available() const
    {
        const auto& tracking = tracked_subspace_.options();
        const bool expanded_tracking_available =
            tracking.enabled &&
            tracking.coverage_recovery_maximum_seed_vectors >
                tracking.maximum_seed_vectors &&
            tracked_subspace_.seed_size() >
                tracking.maximum_seed_vectors;
        return
            confirmation_transaction_active_ &&
            (
                confirmation_union_.usable_results() >= 2 ||
                expanded_tracking_available);
    }

    result_type execute_classification_reconciliation(
        const vector_type& initial_vector,
        std::size_t minimum_results) const
    {
        result_type result;
        if(!classification_reconciliation_available())
        {
            result.status =
                eigensolvers::eigensolver_status::invalid_input;
            result.coverage_complete = false;
            result.diagnostic =
                "tracked classification reconciliation is unavailable";
            return result;
        }

        result = confirmation_union_.finish(
            minimum_results,
            aggregation_options_.minimum_eigenpairs);
        if(result.succeeded())
        {
            ++result.coverage_recoveries;
            result.diagnostic =
                "classification reconciled from the transactional "
                "validated spectrum union without an additional scan: " +
                result.diagnostic;
            return result;
        }

        const auto& tracking = tracked_subspace_.options();
        const std::size_t seed_limit =
            tracking.
                coverage_recovery_maximum_seed_vectors;
        const bool expanded_tracking_available =
            tracking.enabled &&
            seed_limit > tracking.maximum_seed_vectors &&
            tracked_subspace_.seed_size() >
                tracking.maximum_seed_vectors;
        if(!expanded_tracking_available)
        {
            result.diagnostic =
                "classification spectrum union is incomplete and "
                "expanded tracked reconciliation is unavailable: " +
                result.diagnostic;
            return result;
        }

        result_type expanded = execute_with_tracked_seed_limit(
            initial_vector,
            seed_limit);
        record_confirmation_spectrum(expanded);
        result = confirmation_union_.finish(
            minimum_results,
            aggregation_options_.minimum_eigenpairs);
        accumulate_work(result, expanded);
        ++result.coverage_recoveries;
        const std::string expanded_detail =
            std::move(expanded.diagnostic);
        const std::string union_detail = std::move(result.diagnostic);
        std::ostringstream diagnostic;
        diagnostic
            << "tracked classification reconciliation with seed limit "
            << seed_limit
            << (result.succeeded() ? " succeeded" : " failed");
        if(!expanded_detail.empty())
            diagnostic << " after expanded scan {" << expanded_detail << '}';
        if(!union_detail.empty())
            diagnostic << "; union {" << union_detail << '}';
        result.diagnostic = diagnostic.str();
        return result;
    }

private:
    result_type execute_with_tracked_seed_limit(
        const vector_type& initial_vector,
        std::size_t tracked_seed_limit) const
    {
        spectrum_scan_aggregator<real_type> aggregate(
            scans_.size(),
            aggregation_options_);
        detail::vector_workspace<real_space_type> probe(
            real_space_.get());
        detail::vector_workspace<real_space_type> recycled_probe(
            real_space_.get());
        detail::vector_workspace<real_space_type> tracked_probe(
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
        const auto tracked_validation =
            tracked_subspace_.validate(real_operator_);
        const std::size_t tracked_seed_count =
            tracked_validation.accepted
            ? std::min(
                  tracked_validation.accepted_indices.size(),
                  tracked_seed_limit)
            : std::size_t(0);
        std::size_t recycled_probe_seeds = 0;
        std::size_t tracked_probe_seeds = 0;

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
            std::size_t successful_tracked_probes = 0;
            const std::size_t scheduled_probe_count =
                aggregation_options_.probe_count + tracked_seed_count;
            for(std::size_t scheduled_probe_index = 0;
                scheduled_probe_index < scheduled_probe_count;
                ++scheduled_probe_index)
            {
                const bool scheduled_tracked_seed =
                    scheduled_probe_index < tracked_seed_count;
                const std::size_t fresh_probe_index =
                    scheduled_tracked_seed
                    ? std::size_t(0)
                    : scheduled_probe_index - tracked_seed_count;
                const vector_type* probe_vector =
                    &initial_vector;
                bool tracked_seed = false;
                if(scheduled_tracked_seed)
                {
                    generate_probe(
                        aggregation_options_.probe_count +
                            scheduled_probe_index,
                        initial_vector,
                        probe.get());
                    tracked_seed = tracked_subspace_.make_seed(
                        scheduled_probe_index,
                        probe.get(),
                        tracked_validation.accepted_indices,
                        tracked_probe.get());
                    probe_vector = tracked_seed
                        ? &tracked_probe.get()
                        : &probe.get();
                    if(tracked_seed)
                        ++tracked_probe_seeds;
                }
                else if(fresh_probe_index != 0)
                {
                    generate_probe(
                        fresh_probe_index,
                        initial_vector,
                        probe.get());
                    probe_vector = &probe.get();
                }
                const bool recycled_seed =
                    !scheduled_tracked_seed &&
                    fresh_probe_index >= required_successful_probes &&
                    recycled_subspace_.make_seed(
                        fresh_probe_index,
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
                            if(tracked_seed)
                                ++successful_tracked_probes;
                            else if(!recycled_seed)
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
                                    scheduled_probe_index,
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
                        scheduled_probe_index,
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
                scheduled_probe_count;
            scan_result.scans_succeeded = successful_probes;
            std::ostringstream diagnostic;
            diagnostic
                << "multiplicity probes: "
                << successful_probes << "/"
                << scheduled_probe_count
                << " succeeded (minimum "
                << required_successful_probes << "), "
                << successful_fresh_probes
                << " fresh probes succeeded (minimum "
                << required_successful_probes << "), "
                << successful_tracked_probes
                << " tracked probes succeeded, "
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
        result.operator_calls +=
            tracked_validation.operator_calls;
        if(result.succeeded())
        {
            recycled_subspace_.stage(recycle_candidates);
            tracked_subspace_.stage(recycle_candidates);
        }
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
        if(tracked_subspace_.enabled())
        {
            std::ostringstream tracking_diagnostic;
            tracking_diagnostic
                << "invariant-subspace tracking: committed="
                << tracked_subspace_.size()
                << ", validation="
                << (tracked_validation.accepted
                        ? "accepted"
                        : "rejected")
                << ", residual="
                << tracked_validation.residual_frobenius
                << ", relative_residual="
                << tracked_validation.relative_residual
                << ", accepted_dimension="
                << tracked_validation.accepted_indices.size()
                << ", seeded_probes="
                << tracked_probe_seeds
                << ", staged_retained_columns="
                << tracked_subspace_.pending_retained_columns();
            if(tracked_subspace_.pending_overlap())
            {
                const auto& overlap =
                    *tracked_subspace_.pending_overlap();
                tracking_diagnostic
                    << ", staged_overlap_rank="
                    << overlap.numerical_rank
                    << ", staged_dimension_gap="
                    << overlap.dimension_gap
                    << ", staged_max_angle="
                    << overlap.maximum_angle;
            }
            result.diagnostic = result.diagnostic.empty()
                ? tracking_diagnostic.str()
                : result.diagnostic + "; " +
                    tracking_diagnostic.str();
        }
        return result;
    }

public:

    std::size_t scan_count() const
    {
        return scans_.size();
    }

    const std::vector<scan_definition_type>& scans() const
    {
        return scans_;
    }

    void set_aggregation_options(
        aggregation_options_type aggregation_options)
    {
        aggregation_options_ = std::move(aggregation_options);
        update_confirmation_union_options();
    }

    const aggregation_options_type& aggregation_options() const
    {
        return aggregation_options_;
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

    void set_tracking_options(tracking_options_type options)
    {
        tracked_subspace_.set_options(std::move(options));
    }

    const tracking_options_type& tracking_options() const
    {
        return tracked_subspace_.options();
    }

    void begin_recycling_transaction() const
    {
        recycled_subspace_.begin_transaction();
        tracked_subspace_.begin_transaction();
        confirmation_union_.reset();
        confirmation_transaction_active_ = true;
    }

    void commit_recycling_transaction() const
    {
        recycled_subspace_.commit_transaction();
        tracked_subspace_.commit_transaction();
        confirmation_union_.reset();
        confirmation_transaction_active_ = false;
    }

    void rollback_recycling_transaction() const
    {
        recycled_subspace_.rollback_transaction();
        tracked_subspace_.rollback_transaction();
        confirmation_union_.reset();
        confirmation_transaction_active_ = false;
    }

    void reset_recycled_subspace() const
    {
        recycled_subspace_.clear();
        tracked_subspace_.clear();
    }

    void reset_recycled_ritz_subspace() const
    {
        recycled_subspace_.clear();
    }

    void reset_tracked_invariant_subspace() const
    {
        tracked_subspace_.clear();
    }

    std::size_t recycled_subspace_size() const
    {
        return recycled_subspace_.size();
    }

    std::size_t tracked_subspace_size() const
    {
        return tracked_subspace_.size();
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
    mutable tracked_subspace_type tracked_subspace_;
    mutable validated_spectrum_union<real_type> confirmation_union_;
    mutable bool confirmation_transaction_active_ = false;

    void update_confirmation_union_options()
    {
        typename validated_spectrum_union<real_type>::options_type options;
        options.absolute_tolerance =
            aggregation_options_.absolute_tolerance;
        options.relative_tolerance =
            aggregation_options_.relative_tolerance;
        confirmation_union_.set_options(std::move(options));
    }

    void record_confirmation_spectrum(const result_type& result) const
    {
        if(confirmation_transaction_active_)
            confirmation_union_.add(result);
    }

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
        destination.coverage_recoveries +=
            source.coverage_recoveries;
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
