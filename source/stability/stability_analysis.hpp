#ifndef __STABILITY_STABILITY_ANALYSIS_HPP__
#define __STABILITY_STABILITY_ANALYSIS_HPP__

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <stability/analysis/eigensolver_adapter.h>
#include <stability/analysis/initial_vector_policy.h>
#include <stability/analysis/spectrum_classifier.h>
#include <stability/analysis/stability_evaluator.h>
#include <stability/analysis/stability_transition_refiner.h>

namespace stability
{

namespace detail
{

template<class Eigensolver, class Vector, class = void>
struct has_stability_probe_generator : std::false_type
{
};

template<class Eigensolver, class Vector>
struct has_stability_probe_generator<
    Eigensolver,
    Vector,
    std::void_t<decltype(
        std::declval<Eigensolver&>().set_probe_generator(
            std::declval<std::function<void(
                std::size_t,
                const Vector&,
                Vector&)>>()))>>
    : std::true_type
{
};

template<class Eigensolver, class = void>
struct has_stability_probe_generator_query : std::false_type
{
};

template<class Eigensolver>
struct has_stability_probe_generator_query<
    Eigensolver,
    std::void_t<decltype(
        std::declval<const Eigensolver&>().
            has_probe_generator())>>
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

} // namespace detail

/**
 * Bifurcation-diagram stability facade.
 *
 * The eigensolver is injected through the structured adapter interface:
 * execute(initial_vector) -> eigensolver_result. This keeps spectrum
 * transformations, complex arithmetic, and solver ownership outside the
 * nonlinear operator and outside the bifurcation-diagram driver.
 */
template<
    class VectorOperations,
    class NonlinearOperations,
    class Log,
    class Newton,
    class EigensolverAdapter,
    class LinearizationProvider = NonlinearOperations>
class stability_analysis
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using eigensolver_adapter_type = EigensolverAdapter;
    using linearization_provider_type = LinearizationProvider;
    using analysis_result_type =
        analysis::stability_point_result<T>;
    using transition_result_type =
        analysis::stability_transition_result<T>;
    using classifier_options_type =
        analysis::spectrum_classifier_options<T>;

private:
    using classifier_type =
        analysis::spectrum_classifier<T>;
    using initial_vector_policy_type =
        analysis::nonlinear_operator_random_initial_vector<
            NonlinearOperations>;
    using evaluator_type =
        analysis::stability_evaluator<
            VectorOperations,
            linearization_provider_type,
            eigensolver_adapter_type,
            classifier_type,
            initial_vector_policy_type>;
    using transition_refiner_type =
        analysis::stability_transition_refiner<
            VectorOperations,
            NonlinearOperations,
            Newton,
            evaluator_type>;

public:
    stability_analysis(
        VectorOperations* vec_ops,
        Log* log,
        NonlinearOperations* nonlin_op,
        Newton* newton,
        eigensolver_adapter_type* eigensolver_adapter,
        linearization_provider_type* linearization_provider = nullptr)
        : vec_ops_(vec_ops),
          log_(log),
          nonlin_op_(nonlin_op),
          newton_(newton),
          eigensolver_adapter_(eigensolver_adapter),
          linearization_provider_(
              resolve_linearization_provider(
                  nonlin_op,
                  linearization_provider)),
          initial_vector_policy_(nonlin_op),
          evaluator_(
              vec_ops,
              linearization_provider_,
              eigensolver_adapter_,
              classifier_type{},
              initial_vector_policy_),
          transition_refiner_(
              vec_ops,
              nonlin_op,
              newton,
              &evaluator_)
    {
        if(vec_ops_ == nullptr)
            throw std::invalid_argument(
                "stability_analysis: vector operations are null");
        if(log_ == nullptr)
            throw std::invalid_argument(
                "stability_analysis: log is null");
        if(nonlin_op_ == nullptr)
            throw std::invalid_argument(
                "stability_analysis: nonlinear operator is null");
        if(newton_ == nullptr)
            throw std::invalid_argument(
                "stability_analysis: Newton solver is null");
        if(eigensolver_adapter_ == nullptr)
            throw std::invalid_argument(
                "stability_analysis: eigensolver adapter is null");
        configure_stability_probe_generator();
    }

    void set_linear_operator_stable_eigenvalues_halfplane(T sign)
    {
        classifier_options_type options =
            evaluator_.classifier().options();
        options.stable = sign < T(0)
            ? analysis::stable_halfplane::left
            : analysis::stable_halfplane::right;
        evaluator_.classifier().set_options(options);
    }

    void set_classifier_options(
        const classifier_options_type& options)
    {
        evaluator_.classifier().set_options(options);
    }

    const classifier_options_type& classifier_options() const
    {
        return evaluator_.classifier().options();
    }

    void set_classification_retry_count(std::size_t retry_count)
    {
        evaluator_.set_classification_retry_count(retry_count);
    }

    std::size_t classification_retry_count() const
    {
        return evaluator_.classification_retry_count();
    }

    void set_classification_confirmation_count(
        std::size_t confirmation_count)
    {
        evaluator_.set_classification_confirmation_count(
            confirmation_count);
    }

    std::size_t classification_confirmation_count() const
    {
        return evaluator_.classification_confirmation_count();
    }

    void set_transition_refinement_uses_fixed_parameter_newton(
        bool value)
    {
        transition_options_.
            correct_with_fixed_parameter_newton = value;
    }

    bool transition_refinement_uses_fixed_parameter_newton() const
    {
        return transition_options_.
            correct_with_fixed_parameter_newton;
    }

    void set_transition_refinement_maximum_iterations(
        unsigned int value)
    {
        if(value == 0)
            throw std::invalid_argument(
                "stability_analysis: transition refinement requires "
                "at least one iteration");
        transition_options_.maximum_iterations = value;
    }

    unsigned int transition_refinement_maximum_iterations() const
    {
        return transition_options_.maximum_iterations;
    }

    void set_transition_refinement_parameter_tolerance(T value)
    {
        if(value < T(0))
            throw std::invalid_argument(
                "stability_analysis: transition parameter tolerance "
                "must be non-negative");
        transition_options_.parameter_tolerance = value;
    }

    T transition_refinement_parameter_tolerance() const
    {
        return transition_options_.parameter_tolerance;
    }

    void set_transition_refinement_maximum_subdivisions(
        unsigned int value)
    {
        if(value == 0)
            throw std::invalid_argument(
                "stability_analysis: transition sequence requires at "
                "least one subdivision level");
        transition_maximum_subdivisions_ = value;
    }

    unsigned int transition_refinement_maximum_subdivisions() const
    {
        return transition_maximum_subdivisions_;
    }

    template<class Aligner>
    void set_transition_state_aligner(Aligner* aligner)
    {
        transition_refiner_.set_transition_state_aligner(
            aligner);
    }

    void reset_transition_state_aligner()
    {
        transition_refiner_.reset_transition_state_aligner();
    }

    void reset_recycled_subspace()
    {
        if constexpr(
            detail::has_recycled_subspace_reset<
                eigensolver_adapter_type>::value)
        {
            eigensolver_adapter_->reset_recycled_subspace();
        }
    }

    analysis_result_type analyze(
        const T_vec& state,
        T parameter)
    {
        analysis_result_type result =
            evaluator_.analyze(state, parameter);
        log_result(parameter, result);
        return result;
    }

    analysis_result_type analyze_confirmed(
        const T_vec& state,
        T parameter)
    {
        analysis_result_type result =
            evaluator_.analyze_confirmed(state, parameter);
        if(result.succeeded())
        {
            log_->info_f(
                "stability transition confirmation at lambda = %lf: "
                "dim(U) = (%i,%i), attempts = %zu",
                double(parameter),
                result.unstable.real,
                result.unstable.complex_pairs,
                result.classification_attempts);
        }
        else
        {
            log_->warning_f(
                "stability transition confirmation failed at lambda "
                "= %lf: %s",
                double(parameter),
                result.diagnostic.c_str());
        }
        return result;
    }

    analysis_result_type analyze_independent(
        const T_vec& state,
        T parameter)
    {
        reset_recycled_subspace();
        return analyze(state, parameter);
    }

    analysis_result_type analyze_confirmed_independent(
        const T_vec& state,
        T parameter)
    {
        reset_recycled_subspace();
        return analyze_confirmed(state, parameter);
    }

    transition_result_type refine_transition(
        const T_vec& state_1,
        T parameter_1,
        const T_vec& state_2,
        T parameter_2,
        T_vec& refined_state,
        unsigned int maximum_iterations = 0)
    {
        const analysis_result_type result_1 =
            analyze_independent(state_1, parameter_1);
        const analysis_result_type result_2 =
            analyze_independent(state_2, parameter_2);

        typename transition_refiner_type::options_type options =
            transition_options_;
        if(maximum_iterations != 0)
            options.maximum_iterations = maximum_iterations;
        return transition_refiner_.refine(
            state_1,
            parameter_1,
            result_1,
            state_2,
            parameter_2,
            result_2,
            refined_state,
            options);
    }

    transition_result_type refine_transition_confirmed(
        const T_vec& state_1,
        T parameter_1,
        const T_vec& state_2,
        T parameter_2,
        T_vec& refined_state,
        unsigned int maximum_iterations = 0)
    {
        const analysis_result_type result_1 =
            analyze_confirmed_independent(state_1, parameter_1);
        const analysis_result_type result_2 =
            analyze_confirmed_independent(state_2, parameter_2);
        ensure_classified(
            result_1,
            "stability_analysis::refine_transition_confirmed");
        ensure_classified(
            result_2,
            "stability_analysis::refine_transition_confirmed");

        typename transition_refiner_type::options_type options =
            transition_options_;
        options.confirm_stability_classification = true;
        if(maximum_iterations != 0)
            options.maximum_iterations = maximum_iterations;
        return transition_refiner_.refine(
            state_1,
            parameter_1,
            result_1,
            state_2,
            parameter_2,
            result_2,
            refined_state,
            options);
    }

    std::pair<int, int> execute(
        const T_vec& state,
        T parameter)
    {
        const analysis_result_type result =
            analyze(state, parameter);
        ensure_classified(result, "stability_analysis::execute");
        return result.unstable_dimension_pair();
    }

    transition_result_type bisect_bifurcation_point_known(
        const T_vec& state_1,
        T parameter_1,
        std::pair<int, int> dimension_1,
        const T_vec& state_2,
        T parameter_2,
        std::pair<int, int> dimension_2,
        T_vec& refined_state,
        T& refined_parameter,
        unsigned int maximum_iterations = 0)
    {
        const analysis_result_type result_1 =
            endpoint_result(dimension_1);
        const analysis_result_type result_2 =
            endpoint_result(dimension_2);

        typename transition_refiner_type::options_type options =
            transition_options_;
        if(maximum_iterations != 0)
            options.maximum_iterations = maximum_iterations;
        const auto transition = transition_refiner_.refine(
            state_1,
            parameter_1,
            result_1,
            state_2,
            parameter_2,
            result_2,
            refined_state,
            options);
        refined_parameter = transition.parameter;
        ensure_refined(
            transition,
            "stability_analysis::bisect_bifurcation_point_known");

        log_->info_f(
            "stability transition refined to lambda = %lf, "
            "dim(U): before = (%i,%i), after = (%i,%i), "
            "iterations = %i, consistency restarts = %i",
            double(refined_parameter),
            transition.before_stability.unstable.real,
            transition.before_stability.unstable.complex_pairs,
            transition.after_stability.unstable.real,
            transition.after_stability.unstable.complex_pairs,
            int(transition.iterations),
            int(transition.consistency_restarts));
        return transition;
    }

    template<class EventCallback>
    std::size_t refine_transition_sequence_known(
        const T_vec& state_1,
        T parameter_1,
        std::pair<int, int> dimension_1,
        const T_vec& state_2,
        T parameter_2,
        std::pair<int, int> dimension_2,
        EventCallback&& on_event)
    {
        const analysis_result_type result_1 =
            endpoint_result(dimension_1);
        const analysis_result_type result_2 =
            endpoint_result(dimension_2);
        auto callback = std::forward<EventCallback>(on_event);
        return refine_transition_sequence(
            state_1,
            parameter_1,
            result_1,
            state_2,
            parameter_2,
            result_2,
            callback,
            0,
            transition_options_);
    }

    template<class EventCallback>
    std::size_t refine_transition_sequence_confirmed(
        const T_vec& state_1,
        T parameter_1,
        const analysis_result_type& result_1,
        const T_vec& state_2,
        T parameter_2,
        const analysis_result_type& result_2,
        EventCallback&& on_event)
    {
        ensure_classified(
            result_1,
            "stability_analysis::"
            "refine_transition_sequence_confirmed");
        ensure_classified(
            result_2,
            "stability_analysis::"
            "refine_transition_sequence_confirmed");
        auto callback = std::forward<EventCallback>(on_event);
        typename transition_refiner_type::options_type options =
            transition_options_;
        options.confirm_stability_classification = true;
        return refine_transition_sequence(
            state_1,
            parameter_1,
            result_1,
            state_2,
            parameter_2,
            result_2,
            callback,
            0,
            options);
    }

    void bisect_bifurcaiton_point(
        const T_vec& state_1,
        const T& parameter_1,
        const T_vec& state_2,
        const T& parameter_2,
        T_vec& refined_state,
        T& refined_parameter,
        unsigned int maximum_iterations = 0)
    {
        const analysis_result_type result_1 =
            analyze_independent(state_1, parameter_1);
        const analysis_result_type result_2 =
            analyze_independent(state_2, parameter_2);
        ensure_classified(
            result_1,
            "stability_analysis::bisect_bifurcaiton_point");
        ensure_classified(
            result_2,
            "stability_analysis::bisect_bifurcaiton_point");

        if(
            result_1.unstable.real_subspace_dimension() ==
            result_2.unstable.real_subspace_dimension())
        {
            log_->info_f(
                "stability transition is absent: at %lf dim(U) = "
                "(%i,%i), at %lf dim(U) = (%i,%i); both real "
                "unstable-subspace dimensions are %i",
                double(parameter_1),
                result_1.unstable.real,
                result_1.unstable.complex_pairs,
                double(parameter_2),
                result_2.unstable.real,
                result_2.unstable.complex_pairs,
                result_1.unstable.real_subspace_dimension());
            return;
        }

        bisect_bifurcation_point_known(
            state_1,
            parameter_1,
            result_1.unstable_dimension_pair(),
            state_2,
            parameter_2,
            result_2.unstable_dimension_pair(),
            refined_state,
            refined_parameter,
            maximum_iterations);
    }

private:
    VectorOperations* vec_ops_;
    Log* log_;
    NonlinearOperations* nonlin_op_;
    Newton* newton_;
    eigensolver_adapter_type* eigensolver_adapter_;
    linearization_provider_type* linearization_provider_;
    initial_vector_policy_type initial_vector_policy_;
    evaluator_type evaluator_;
    transition_refiner_type transition_refiner_;
    typename transition_refiner_type::options_type
        transition_options_;
    unsigned int transition_maximum_subdivisions_ = 8;

    void configure_stability_probe_generator()
    {
        if constexpr(
            detail::has_stability_probe_generator<
                eigensolver_adapter_type,
                T_vec>::value)
        {
            if constexpr(
                detail::has_stability_probe_generator_query<
                    eigensolver_adapter_type>::value)
            {
                if(eigensolver_adapter_->has_probe_generator())
                    return;
            }
            NonlinearOperations* nonlinear_operations =
                nonlin_op_;
            eigensolver_adapter_->set_probe_generator(
                [nonlinear_operations](
                    std::size_t,
                    const T_vec&,
                    T_vec& vector)
                {
                    analysis::initialize_stability_probe(
                        *nonlinear_operations,
                        vector);
                });
        }
    }

    static linearization_provider_type*
    resolve_linearization_provider(
        NonlinearOperations* nonlinear_operations,
        linearization_provider_type* linearization_provider)
    {
        if(linearization_provider != nullptr)
            return linearization_provider;
        if constexpr(
            std::is_same<
                linearization_provider_type,
                NonlinearOperations>::value)
        {
            return nonlinear_operations;
        }
        throw std::invalid_argument(
            "stability_analysis: custom linearization provider is null");
    }

    static analysis_result_type endpoint_result(
        std::pair<int, int> dimension)
    {
        analysis_result_type result;
        result.eigensolver_status =
            eigensolvers::eigensolver_status::success;
        result.classification_status =
            analysis::spectrum_classification_status::complete;
        result.unstable.real = dimension.first;
        result.unstable.complex_pairs = dimension.second;
        return result;
    }

    static void ensure_classified(
        const analysis_result_type& result,
        const char* context)
    {
        if(result.succeeded())
            return;
        throw std::runtime_error(
            std::string(context) + ": " +
            (result.diagnostic.empty()
                 ? analysis::spectrum_classification_status_name(
                       result.classification_status)
                 : result.diagnostic));
    }

    template<class TransitionResult>
    static void ensure_refined(
        const TransitionResult& result,
        const char* context)
    {
        if(result.succeeded())
            return;
        throw std::runtime_error(
            std::string(context) + ": " +
            (result.diagnostic.empty()
                 ? analysis::stability_transition_status_name(
                       result.status)
                 : result.diagnostic));
    }

    template<class EventCallback>
    std::size_t refine_transition_sequence(
        const T_vec& lower_state,
        T lower_parameter,
        const analysis_result_type& lower_stability,
        const T_vec& upper_state,
        T upper_parameter,
        const analysis_result_type& upper_stability,
        EventCallback& on_event,
        unsigned int subdivision_depth,
        const typename transition_refiner_type::options_type& options)
    {
        const int lower_dimension =
            lower_stability.unstable.real_subspace_dimension();
        const int upper_dimension =
            upper_stability.unstable.real_subspace_dimension();
        if(lower_dimension == upper_dimension)
            return 0;

        analysis::detail::vector_workspace<VectorOperations>
            refined_state(vec_ops_);
        const transition_result_type transition =
            transition_refiner_.refine(
                lower_state,
                lower_parameter,
                lower_stability,
                upper_state,
                upper_parameter,
                upper_stability,
                refined_state.get(),
                options);
        if(transition.succeeded())
        {
            log_transition(transition);
            on_event(transition, refined_state.get());
            return 1;
        }

        if(
            transition.status !=
                analysis::stability_transition_status::
                    unexpected_signature ||
            !transition.stability.succeeded())
        {
            ensure_refined(
                transition,
                "stability_analysis::refine_transition_sequence_known");
        }
        if(subdivision_depth >= transition_maximum_subdivisions_)
        {
            throw std::runtime_error(
                "stability_analysis::refine_transition_sequence_known: "
                "maximum transition subdivisions reached after " +
                transition.diagnostic);
        }

        const T minimum_parameter =
            std::min(lower_parameter, upper_parameter);
        const T maximum_parameter =
            std::max(lower_parameter, upper_parameter);
        if(
            !(transition.parameter > minimum_parameter) ||
            !(transition.parameter < maximum_parameter))
        {
            throw std::runtime_error(
                "stability_analysis::refine_transition_sequence_known: "
                "the intermediate transition state does not split the "
                "parameter interval");
        }

        const int middle_dimension =
            transition.stability.unstable.
                real_subspace_dimension();
        log_->info_f(
            "stability transition interval contains multiple events: "
            "lambda = %.16le has dim(U) = (%i,%i), splitting endpoint "
            "dimensions %i and %i at subdivision depth %u",
            double(transition.parameter),
            transition.stability.unstable.real,
            transition.stability.unstable.complex_pairs,
            lower_dimension,
            upper_dimension,
            subdivision_depth + 1);

        std::size_t event_count = 0;
        if(lower_dimension != middle_dimension)
        {
            event_count += refine_transition_sequence(
                lower_state,
                lower_parameter,
                lower_stability,
                refined_state.get(),
                transition.parameter,
                transition.stability,
                on_event,
                subdivision_depth + 1,
                options);
        }
        if(middle_dimension != upper_dimension)
        {
            event_count += refine_transition_sequence(
                refined_state.get(),
                transition.parameter,
                transition.stability,
                upper_state,
                upper_parameter,
                upper_stability,
                on_event,
                subdivision_depth + 1,
                options);
        }
        if(event_count == 0)
        {
            throw std::runtime_error(
                "stability_analysis::refine_transition_sequence_known: "
                "subdivision did not produce a transition bracket");
        }
        return event_count;
    }

    void log_transition(
        const transition_result_type& transition) const
    {
        log_->info_f(
            "stability transition refined to lambda = %lf, "
            "dim(U): before = (%i,%i), after = (%i,%i), "
            "iterations = %i, consistency restarts = %i",
            double(transition.parameter),
            transition.before_stability.unstable.real,
            transition.before_stability.unstable.complex_pairs,
            transition.after_stability.unstable.real,
            transition.after_stability.unstable.complex_pairs,
            int(transition.iterations),
            int(transition.consistency_restarts));
    }

    void log_result(
        T parameter,
        const analysis_result_type& result) const
    {
        if(result.classification_attempts > 1)
        {
            log_->info_f(
                "stability.execute: spectrum classification at lambda "
                "= %lf required %zu attempts",
                double(parameter),
                result.classification_attempts);
        }
        for(const auto& estimate : result.eigenpairs)
        {
            const T real = estimate.value.real();
            const T imag = estimate.value.imag();
            if(imag >= T(0))
                log_->info_f(
                    "   %.3lf+%.3lfi",
                    double(real),
                    double(imag));
            else
                log_->info_f(
                    "   %.3lf%.3lfi",
                    double(real),
                    double(imag));
        }

        if(result.succeeded())
        {
            log_->info_f(
                "stability.execute: unstable manifold dimension at "
                "lambda = %lf: real = %i, complex pairs = %i",
                double(parameter),
                result.unstable.real,
                result.unstable.complex_pairs);
        }
        else
        {
            log_->warning_f(
                "stability.execute: classification failed at lambda "
                "= %lf: %s",
                double(parameter),
                result.diagnostic.c_str());
        }
    }
};

} // namespace stability

#endif
