#ifndef __STABILITY_ANALYSIS_STABILITY_EVALUATOR_H__
#define __STABILITY_ANALYSIS_STABILITY_EVALUATOR_H__

#include <algorithm>
#include <exception>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <stability/model_adapter_contract.h>

#include "detail/vector_workspace.h"
#include "stability_point_result.h"

namespace stability
{
namespace analysis
{

namespace detail
{

template<class Eigensolver, class Vector, class = void>
struct has_classification_fallback : std::false_type
{
};

template<class Eigensolver, class Vector>
struct has_classification_fallback<
    Eigensolver,
    Vector,
    std::void_t<
        decltype(
            std::declval<const Eigensolver&>().
                classification_fallback_available()),
        decltype(
            std::declval<const Eigensolver&>().
                execute_classification_fallback(
                    std::declval<const Vector&>()))>>
    : std::true_type
{
};

template<class Eigensolver, class Vector, class = void>
struct has_classification_confirmation : std::false_type
{
};

template<class Eigensolver, class Vector>
struct has_classification_confirmation<
    Eigensolver,
    Vector,
    std::void_t<
        decltype(
            std::declval<const Eigensolver&>().
                classification_confirmation_available()),
        decltype(
            std::declval<const Eigensolver&>().
                execute_classification_confirmation(
                    std::declval<const Vector&>()))>>
    : std::true_type
{
};

template<class Eigensolver, class Vector, class = void>
struct has_classification_reconciliation : std::false_type
{
};

template<class Eigensolver, class Vector>
struct has_classification_reconciliation<
    Eigensolver,
    Vector,
    std::void_t<
        decltype(
            std::declval<const Eigensolver&>().
                classification_reconciliation_available()),
        decltype(
            std::declval<const Eigensolver&>().
                execute_classification_reconciliation(
                    std::declval<const Vector&>(),
                    std::declval<std::size_t>()))>>
    : std::true_type
{
};

template<class Eigensolver, class = void>
struct has_recycling_transaction : std::false_type
{
};

template<class Eigensolver>
struct has_recycling_transaction<
    Eigensolver,
    std::void_t<
        decltype(
            std::declval<const Eigensolver&>().
                begin_recycling_transaction()),
        decltype(
            std::declval<const Eigensolver&>().
                commit_recycling_transaction()),
        decltype(
            std::declval<const Eigensolver&>().
                rollback_recycling_transaction())>>
    : std::true_type
{
};

template<class Eigensolver, class = void>
struct has_recycled_subspace_reset : std::false_type
{
};

template<class Eigensolver>
struct has_recycled_subspace_reset<
    Eigensolver,
    std::void_t<decltype(
        std::declval<const Eigensolver&>().
            reset_recycled_subspace())>>
    : std::true_type
{
};

template<class Eigensolver, class = void>
struct has_recycled_ritz_subspace_reset : std::false_type
{
};

template<class Eigensolver>
struct has_recycled_ritz_subspace_reset<
    Eigensolver,
    std::void_t<decltype(
        std::declval<const Eigensolver&>().
            reset_recycled_ritz_subspace())>>
    : std::true_type
{
};

template<class Eigensolver>
class recycling_transaction
{
public:
    explicit recycling_transaction(Eigensolver* eigensolver)
        : eigensolver_(eigensolver)
    {
        if constexpr(has_recycling_transaction<Eigensolver>::value)
        {
            eigensolver_->begin_recycling_transaction();
            active_ = true;
        }
    }

    recycling_transaction(const recycling_transaction&) = delete;
    recycling_transaction& operator=(
        const recycling_transaction&) = delete;

    ~recycling_transaction()
    {
        if constexpr(has_recycling_transaction<Eigensolver>::value)
        {
            if(active_)
                eigensolver_->rollback_recycling_transaction();
        }
    }

    template<class Result>
    Result finish(Result result)
    {
        if constexpr(has_recycling_transaction<Eigensolver>::value)
        {
            if(active_)
            {
                if(result.succeeded())
                    eigensolver_->commit_recycling_transaction();
                else
                    eigensolver_->rollback_recycling_transaction();
                active_ = false;
            }
        }
        return result;
    }

private:
    Eigensolver* eigensolver_;
    bool active_ = false;
};

} // namespace detail

template<
    class VectorOperations,
    class LinearizationProvider,
    class EigensolverAdapter,
    class SpectrumClassifier,
    class InitialVectorPolicy>
class stability_evaluator
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using result_type = stability_point_result<scalar_type>;

    static_assert(
        model_adapter::evaluator_contract<
            VectorOperations,
            LinearizationProvider,
            EigensolverAdapter>::linearization_provider,
        "stability linearization provider must implement "
        "set_linearization_point(const vector_type&, scalar_type)");
    static_assert(
        model_adapter::evaluator_contract<
            VectorOperations,
            LinearizationProvider,
            EigensolverAdapter>::eigensolver_adapter,
        "stability eigensolver adapter must implement execute(const "
        "vector_type&) returning eigensolver_result<scalar_type>");

    stability_evaluator(
        VectorOperations* vector_operations,
        LinearizationProvider* linearization_provider,
        EigensolverAdapter* eigensolver,
        SpectrumClassifier classifier,
        InitialVectorPolicy initial_vector_policy)
        : vector_operations_(vector_operations),
          linearization_provider_(linearization_provider),
          eigensolver_(eigensolver),
          classifier_(std::move(classifier)),
          initial_vector_policy_(std::move(initial_vector_policy)),
          initial_vector_(vector_operations)
    {
        if(vector_operations_ == nullptr)
            throw std::invalid_argument(
                "stability_evaluator: vector operations are null");
        if(linearization_provider_ == nullptr)
            throw std::invalid_argument(
                "stability_evaluator: linearization provider is null");
        if(eigensolver_ == nullptr)
            throw std::invalid_argument(
                "stability_evaluator: eigensolver is null");
    }

    result_type analyze(
        const vector_type& state,
        scalar_type parameter)
    {
        const result_type preparation =
            prepare(state, parameter, true);
        if(
            preparation.classification_status !=
            spectrum_classification_status::invalid_input)
            return preparation;

        detail::recycling_transaction<EigensolverAdapter>
            transaction(eigensolver_);
        return transaction.finish(
            analyze_prepared(initial_vector_.get()));
    }

    result_type analyze(
        const vector_type& state,
        scalar_type parameter,
        const vector_type& initial_vector)
    {
        const result_type preparation =
            prepare(state, parameter, false);
        if(
            preparation.classification_status !=
            spectrum_classification_status::invalid_input)
            return preparation;

        detail::recycling_transaction<EigensolverAdapter>
            transaction(eigensolver_);
        return transaction.finish(
            analyze_prepared(initial_vector));
    }

    result_type analyze_confirmed(
        const vector_type& state,
        scalar_type parameter)
    {
        const result_type preparation =
            prepare(state, parameter, true);
        if(
            preparation.classification_status !=
            spectrum_classification_status::invalid_input)
            return preparation;

        detail::recycling_transaction<EigensolverAdapter>
            transaction(eigensolver_);

        if constexpr(
            detail::has_classification_confirmation<
                EigensolverAdapter,
                vector_type>::value)
        {
            if(eigensolver_->
                   classification_confirmation_available())
            {
                result_type result = classifier_.classify(
                    eigensolver_->
                        execute_classification_confirmation(
                            initial_vector_.get()));
                result.classification_attempts = 1;
                if(result.succeeded())
                {
                    result.observed_unstable_dimensions = {{
                        result.unstable,
                        std::size_t(1)}};
                    result.diagnostic =
                        result.diagnostic.empty()
                        ? "classification confirmed by dedicated "
                          "eigensolver"
                        : "classification confirmed by dedicated "
                          "eigensolver: " + result.diagnostic;
                }
                return transaction.finish(std::move(result));
            }
        }

        struct consensus_candidate
        {
            unstable_dimension signature;
            result_type result;
            std::size_t occurrences = 0;
        };

        std::vector<consensus_candidate> candidates;
        std::size_t total_attempts = 0;
        std::size_t total_coverage_recoveries = 0;
        std::size_t failed_runs = 0;
        std::vector<std::string> failed_run_diagnostics;
        bool have_last_failure = false;
        result_type last_failure;
        bool disagreement_observed = false;
        const std::size_t maximum_runs =
            classification_confirmation_count_ +
            classification_retry_count_;
        for(std::size_t run = 0; run < maximum_runs; ++run)
        {
            if(run != 0)
            {
                try
                {
                    initial_vector_policy_(initial_vector_.get());
                }
                catch(const std::exception& error)
                {
                    return failure(
                        eigensolvers::eigensolver_status::operator_failure,
                        error.what());
                }
                catch(...)
                {
                    return failure(
                        eigensolvers::eigensolver_status::operator_failure,
                        "failed to prepare an independent stability "
                        "confirmation vector");
                }
            }

            result_type repeated =
                analyze_prepared(initial_vector_.get());
            total_attempts += repeated.classification_attempts;
            total_coverage_recoveries +=
                repeated.coverage_recoveries;
            if(!repeated.succeeded())
            {
                ++failed_runs;
                failed_run_diagnostics.push_back(
                    confirmation_failure_summary(
                        run,
                        repeated));
                have_last_failure = true;
                last_failure = std::move(repeated);
                continue;
            }

            auto candidate = candidates.begin();
            for(; candidate != candidates.end(); ++candidate)
            {
                if(candidate->signature == repeated.unstable)
                    break;
            }
            if(candidate == candidates.end())
            {
                disagreement_observed = !candidates.empty();
                candidates.push_back(consensus_candidate{
                    repeated.unstable,
                    repeated,
                    1});
            }
            else
            {
                candidate->result = repeated;
                ++candidate->occurrences;
            }

            if(
                !disagreement_observed &&
                candidates.size() == 1 &&
                candidates.front().occurrences >=
                    classification_confirmation_count_)
            {
                result_type confirmed = candidates.front().result;
                confirmed.classification_attempts = total_attempts;
                confirmed.coverage_recoveries =
                    total_coverage_recoveries;
                append_observations(confirmed, candidates);
                confirmed.diagnostic =
                    "classification confirmed by " +
                    std::to_string(
                        classification_confirmation_count_) +
                    " successful independent eigensolver run(s)" +
                    failed_run_suffix(failed_runs);
                append_failed_run_diagnostics(
                    confirmed.diagnostic,
                    failed_run_diagnostics);
                return transaction.finish(std::move(confirmed));
            }
        }

        const auto has_unique_maximum_consensus =
            [this, &candidates]()
            {
                int maximum_dimension = -1;
                for(const auto& candidate : candidates)
                {
                    maximum_dimension = std::max(
                        maximum_dimension,
                        candidate.signature.real_subspace_dimension());
                }

                std::size_t qualifying_candidates = 0;
                for(const auto& candidate : candidates)
                {
                    if(
                        candidate.signature.real_subspace_dimension() ==
                            maximum_dimension &&
                        candidate.occurrences >=
                            classification_confirmation_count_)
                    {
                        ++qualifying_candidates;
                    }
                }
                return qualifying_candidates == 1;
            };

        bool reconciliation_attempted = false;
        if(!candidates.empty() && !has_unique_maximum_consensus())
        {
            if constexpr(
                detail::has_classification_reconciliation<
                    EigensolverAdapter,
                    vector_type>::value)
            {
                if(eigensolver_->classification_reconciliation_available())
                {
                    reconciliation_attempted = true;
                    try
                    {
                        initial_vector_policy_(initial_vector_.get());
                    }
                    catch(const std::exception& error)
                    {
                        return transaction.finish(failure(
                            eigensolvers::eigensolver_status::
                                operator_failure,
                            error.what()));
                    }
                    catch(...)
                    {
                        return transaction.finish(failure(
                            eigensolvers::eigensolver_status::
                                operator_failure,
                            "failed to prepare a stability "
                            "reconciliation vector"));
                    }

                    result_type reconciled = classifier_.classify(
                        eigensolver_->
                            execute_classification_reconciliation(
                                initial_vector_.get(),
                                classification_confirmation_count_));
                    reconciled.classification_attempts = 1;
                    total_attempts += reconciled.classification_attempts;
                    total_coverage_recoveries +=
                        reconciled.coverage_recoveries;
                    if(!reconciled.succeeded())
                    {
                        ++failed_runs;
                        failed_run_diagnostics.push_back(
                            confirmation_failure_summary(
                                maximum_runs,
                                reconciled));
                        have_last_failure = true;
                        last_failure = std::move(reconciled);
                    }
                    else
                    {
                        const std::string reconciliation_detail =
                            reconciled.diagnostic;
                        auto candidate = candidates.begin();
                        for(; candidate != candidates.end(); ++candidate)
                        {
                            if(candidate->signature == reconciled.unstable)
                                break;
                        }
                        if(candidate == candidates.end())
                        {
                            disagreement_observed = !candidates.empty();
                            candidates.push_back(consensus_candidate{
                                reconciled.unstable,
                                reconciled,
                                1});
                        }
                        else
                        {
                            candidate->result = reconciled;
                            ++candidate->occurrences;
                        }

                        reconciled.classification_attempts =
                            total_attempts;
                        reconciled.coverage_recoveries =
                            total_coverage_recoveries;
                        append_observations(reconciled, candidates);
                        reconciled.diagnostic =
                            "classification accepted from an "
                            "authoritative validated-spectrum "
                            "reconciliation after " +
                            std::to_string(candidates.size()) +
                            " observed unstable signature(s)" +
                            failed_run_suffix(failed_runs);
                        if(!reconciliation_detail.empty())
                        {
                            reconciled.diagnostic +=
                                ": " + reconciliation_detail;
                        }
                        append_failed_run_diagnostics(
                            reconciled.diagnostic,
                            failed_run_diagnostics);
                        return transaction.finish(
                            std::move(reconciled));
                    }
                }
            }
        }

        if(candidates.empty())
        {
            if(!have_last_failure)
            {
                return transaction.finish(failure(
                    eigensolvers::eigensolver_status::no_convergence,
                    "transition classification confirmation produced no "
                    "successful eigensolver run"));
            }
            last_failure.classification_attempts = total_attempts;
            last_failure.coverage_recoveries =
                total_coverage_recoveries;
            last_failure.diagnostic =
                "transition classification confirmation failed in " +
                std::to_string(failed_runs) +
                " independent eigensolver run(s)";
            append_failed_run_diagnostics(
                last_failure.diagnostic,
                failed_run_diagnostics);
            return transaction.finish(std::move(last_failure));
        }

        int maximum_subspace_dimension = -1;
        for(const auto& candidate : candidates)
        {
            maximum_subspace_dimension = std::max(
                maximum_subspace_dimension,
                candidate.signature.real_subspace_dimension());
        }

        const consensus_candidate* selected = nullptr;
        bool ambiguous = false;
        for(const auto& candidate : candidates)
        {
            if(
                candidate.signature.real_subspace_dimension() !=
                    maximum_subspace_dimension ||
                candidate.occurrences <
                    classification_confirmation_count_)
            {
                continue;
            }
            if(selected != nullptr)
            {
                ambiguous = true;
                break;
            }
            selected = &candidate;
        }
        if(selected != nullptr && !ambiguous)
        {
            result_type confirmed = selected->result;
            confirmed.classification_attempts = total_attempts;
            confirmed.coverage_recoveries =
                total_coverage_recoveries;
            append_observations(confirmed, candidates);
                confirmed.diagnostic =
                    "classification consensus recovered after " +
                    std::to_string(candidates.size()) +
                    " observed unstable signature(s) in " +
                    std::to_string(maximum_runs) +
                    " scheduled independent eigensolver run(s)" +
                    (
                        reconciliation_attempted
                        ? " plus one tracked reconciliation pass"
                        : std::string{}) +
                    failed_run_suffix(failed_runs);
            append_failed_run_diagnostics(
                confirmed.diagnostic,
                failed_run_diagnostics);
            return transaction.finish(std::move(confirmed));
        }

        result_type inconsistent = candidates.back().result;
        inconsistent.classification_status =
            spectrum_classification_status::incomplete;
        inconsistent.classification_attempts = total_attempts;
        inconsistent.coverage_recoveries =
            total_coverage_recoveries;
        append_observations(inconsistent, candidates);
        std::ostringstream diagnostic;
        diagnostic
            << "transition classification is inconsistent across "
               "independent eigensolver runs"
            << failed_run_suffix(failed_runs);
        if(reconciliation_attempted)
            diagnostic << "; tracked reconciliation did not establish "
                           "consensus";
        for(const auto& candidate : candidates)
        {
            diagnostic
                << "; signature ("
                << candidate.signature.real << ','
                << candidate.signature.complex_pairs
                << ") occurred " << candidate.occurrences
                << " time(s) {"
                << classification_summary(candidate.result)
                << '}';
        }
        append_failed_run_diagnostics(
            diagnostic,
            failed_run_diagnostics);
        inconsistent.diagnostic = diagnostic.str();
        return transaction.finish(std::move(inconsistent));
    }

    SpectrumClassifier& classifier()
    {
        return classifier_;
    }

    const SpectrumClassifier& classifier() const
    {
        return classifier_;
    }

    void set_classification_retry_count(std::size_t retry_count)
    {
        classification_retry_count_ = retry_count;
    }

    std::size_t classification_retry_count() const
    {
        return classification_retry_count_;
    }

    void set_classification_confirmation_count(
        std::size_t confirmation_count)
    {
        if(confirmation_count == 0)
            throw std::invalid_argument(
                "stability_evaluator: classification confirmation "
                "count must be nonzero");
        classification_confirmation_count_ = confirmation_count;
    }

    std::size_t classification_confirmation_count() const
    {
        return classification_confirmation_count_;
    }

    void reset_recycled_subspace()
    {
        if constexpr(
            detail::has_recycled_subspace_reset<
                EigensolverAdapter>::value)
        {
            eigensolver_->reset_recycled_subspace();
        }
    }

    void reset_recycled_ritz_subspace()
    {
        if constexpr(
            detail::has_recycled_ritz_subspace_reset<
                EigensolverAdapter>::value)
        {
            eigensolver_->reset_recycled_ritz_subspace();
        }
        else if constexpr(
            detail::has_recycled_subspace_reset<
                EigensolverAdapter>::value)
        {
            eigensolver_->reset_recycled_subspace();
        }
    }

private:
    VectorOperations* vector_operations_;
    LinearizationProvider* linearization_provider_;
    EigensolverAdapter* eigensolver_;
    SpectrumClassifier classifier_;
    InitialVectorPolicy initial_vector_policy_;
    detail::vector_workspace<VectorOperations> initial_vector_;
    std::size_t classification_retry_count_ = 0;
    std::size_t classification_confirmation_count_ = 2;

    static std::string classification_summary(
        const result_type& result)
    {
        std::ostringstream message;
        message
            << "stable_real=" << result.stable_real
            << ", neutral_real=" << result.neutral_real
            << ", unstable=(" << result.unstable.real
            << "," << result.unstable.complex_pairs << ")"
            << ", neutral_complex_pairs="
            << result.neutral_complex_pairs
            << ", eigenvalues=[";
        message << std::scientific << std::setprecision(6);
        for(std::size_t index = 0;
            index < result.eigenpairs.size();
            ++index)
        {
            if(index != 0)
                message << ", ";
            message
                << "(" << result.eigenpairs[index].value.real()
                << "," << result.eigenpairs[index].value.imag()
                << ")";
        }
        message << "]";
        return message.str();
    }

    static std::string failed_run_suffix(std::size_t failed_runs)
    {
        if(failed_runs == 0)
            return {};
        return
            " after " + std::to_string(failed_runs) +
            " failed run(s) consumed the retry budget";
    }

    static std::string confirmation_failure_summary(
        std::size_t run,
        const result_type& result)
    {
        constexpr std::size_t maximum_detail_length = 2048;
        std::ostringstream summary;
        summary
            << "run[" << run << "] eigensolver="
            << eigensolvers::eigensolver_status_name(
                   result.eigensolver_status)
            << ", classification="
            << spectrum_classification_status_name(
                   result.classification_status)
            << ", attempts=" << result.classification_attempts;
        if(!result.diagnostic.empty())
        {
            summary << ", detail={";
            if(result.diagnostic.size() <= maximum_detail_length)
            {
                summary << result.diagnostic;
            }
            else
            {
                summary
                    << result.diagnostic.substr(
                           0,
                           maximum_detail_length)
                    << "... [truncated "
                    << result.diagnostic.size() -
                           maximum_detail_length
                    << " characters]";
            }
            summary << '}';
        }
        return summary.str();
    }

    static void append_failed_run_diagnostics(
        std::string& destination,
        const std::vector<std::string>& failures)
    {
        if(failures.empty())
            return;
        destination += "; failed confirmation runs: ";
        for(std::size_t index = 0; index < failures.size(); ++index)
        {
            if(index != 0)
                destination += " | ";
            destination += failures[index];
        }
    }

    static void append_failed_run_diagnostics(
        std::ostringstream& destination,
        const std::vector<std::string>& failures)
    {
        if(failures.empty())
            return;
        destination << "; failed confirmation runs: ";
        for(std::size_t index = 0; index < failures.size(); ++index)
        {
            if(index != 0)
                destination << " | ";
            destination << failures[index];
        }
    }

    template<class CandidateRange>
    static void append_observations(
        result_type& result,
        const CandidateRange& candidates)
    {
        result.observed_unstable_dimensions.clear();
        result.observed_unstable_dimensions.reserve(
            candidates.size());
        for(const auto& candidate : candidates)
        {
            result.observed_unstable_dimensions.push_back({
                candidate.signature,
                candidate.occurrences});
        }
    }

    result_type prepare(
        const vector_type& state,
        scalar_type parameter,
        bool initialize_vector)
    {
        try
        {
            linearization_provider_->set_linearization_point(
                state,
                parameter);
            if(initialize_vector)
                initial_vector_policy_(initial_vector_.get());
        }
        catch(const std::exception& error)
        {
            return failure(
                eigensolvers::eigensolver_status::operator_failure,
                error.what());
        }
        catch(...)
        {
            return failure(
                eigensolvers::eigensolver_status::operator_failure,
                "failed to prepare the stability linearization");
        }
        return {};
    }

    result_type analyze_prepared(const vector_type& initial_vector)
    {
        result_type result;
        std::string first_diagnostic;
        try
        {
            result = classifier_.classify(
                eigensolver_->execute(initial_vector));
            result.classification_attempts = 1;
            std::size_t total_coverage_recoveries =
                result.coverage_recoveries;
            if(
                result.succeeded() ||
                result.classification_status !=
                    spectrum_classification_status::incomplete)
                return result;
            first_diagnostic = result.diagnostic;
            std::size_t attempts = 1;

            if constexpr(
                detail::has_classification_fallback<
                    EigensolverAdapter,
                    vector_type>::value)
            {
                if(eigensolver_->classification_fallback_available())
                {
                    result = classifier_.classify(
                        eigensolver_->
                            execute_classification_fallback(
                                initial_vector));
                    total_coverage_recoveries +=
                        result.coverage_recoveries;
                    result.coverage_recoveries =
                        total_coverage_recoveries;
                    result.classification_attempts = ++attempts;
                    if(result.succeeded())
                    {
                        result.diagnostic =
                            "classification recovered by fallback on "
                            "attempt " +
                            std::to_string(attempts) +
                            " after {" + first_diagnostic + "}";
                        return result;
                    }
                }
            }

            if(classification_retry_count_ == 0)
                return result;

            for(std::size_t retry = 0;
                retry < classification_retry_count_;
                ++retry)
            {
                initial_vector_policy_(initial_vector_.get());
                result = classifier_.classify(
                    eigensolver_->execute(initial_vector_.get()));
                total_coverage_recoveries +=
                    result.coverage_recoveries;
                result.coverage_recoveries =
                    total_coverage_recoveries;
                result.classification_attempts = ++attempts;
                if(result.succeeded())
                {
                    result.diagnostic =
                        "classification recovered on attempt " +
                        std::to_string(result.classification_attempts) +
                        " after {" + first_diagnostic + "}";
                    return result;
                }
                if(
                    result.classification_status !=
                    spectrum_classification_status::incomplete)
                    return result;
            }
            result.diagnostic =
                "classification retries exhausted after " +
                std::to_string(result.classification_attempts) +
                " attempts; first failure: {" + first_diagnostic +
                "}; last failure: {" + result.diagnostic + "}";
            return result;
        }
        catch(const std::exception& error)
        {
            return failure(
                eigensolvers::eigensolver_status::operator_failure,
                error.what());
        }
        catch(...)
        {
            return failure(
                eigensolvers::eigensolver_status::operator_failure,
                "eigensolver raised an unknown exception");
        }
    }

    static result_type failure(
        eigensolvers::eigensolver_status status,
        std::string diagnostic)
    {
        result_type result;
        result.eigensolver_status = status;
        result.classification_status =
            spectrum_classification_status::eigensolver_failure;
        result.diagnostic = std::move(diagnostic);
        return result;
    }
};

} // namespace analysis
} // namespace stability

#endif
