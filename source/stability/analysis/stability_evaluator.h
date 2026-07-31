#ifndef __STABILITY_ANALYSIS_STABILITY_EVALUATOR_H__
#define __STABILITY_ANALYSIS_STABILITY_EVALUATOR_H__

#include <exception>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

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

        return analyze_prepared(initial_vector_.get());
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

        return analyze_prepared(initial_vector);
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
                    result.diagnostic =
                        result.diagnostic.empty()
                        ? "classification confirmed by dedicated "
                          "eigensolver"
                        : "classification confirmed by dedicated "
                          "eigensolver: " + result.diagnostic;
                }
                return result;
            }
        }

        result_type confirmed =
            analyze_prepared(initial_vector_.get());
        if(!confirmed.succeeded())
            return confirmed;

        const unstable_dimension expected = confirmed.unstable;
        std::size_t total_attempts =
            confirmed.classification_attempts;
        for(std::size_t confirmation = 1;
            confirmation < classification_confirmation_count_;
            ++confirmation)
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

            result_type repeated =
                analyze_prepared(initial_vector_.get());
            total_attempts += repeated.classification_attempts;
            if(!repeated.succeeded())
            {
                repeated.diagnostic =
                    "transition classification confirmation failed: " +
                    repeated.diagnostic;
                repeated.classification_attempts = total_attempts;
                return repeated;
            }
            if(repeated.unstable != expected)
            {
                repeated.classification_status =
                    spectrum_classification_status::incomplete;
                repeated.classification_attempts = total_attempts;
                repeated.diagnostic =
                    "transition classification is inconsistent across "
                    "independent eigensolver runs: expected unstable "
                    "signature (" +
                    std::to_string(expected.real) + "," +
                    std::to_string(expected.complex_pairs) +
                    "), obtained (" +
                    std::to_string(repeated.unstable.real) + "," +
                    std::to_string(
                        repeated.unstable.complex_pairs) + ")";
                return repeated;
            }
        }

        confirmed.classification_attempts = total_attempts;
        confirmed.diagnostic =
            "classification confirmed by " +
            std::to_string(classification_confirmation_count_) +
            " independent eigensolver run(s)";
        return confirmed;
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

private:
    VectorOperations* vector_operations_;
    LinearizationProvider* linearization_provider_;
    EigensolverAdapter* eigensolver_;
    SpectrumClassifier classifier_;
    InitialVectorPolicy initial_vector_policy_;
    detail::vector_workspace<VectorOperations> initial_vector_;
    std::size_t classification_retry_count_ = 0;
    std::size_t classification_confirmation_count_ = 2;

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
