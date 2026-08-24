#ifndef __STABILITY_ANALYSIS_STABILITY_TRANSITION_REFINER_H__
#define __STABILITY_ANALYSIS_STABILITY_TRANSITION_REFINER_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "detail/vector_workspace.h"
#include "stability_point_result.h"
#include "transition_state_alignment.h"

namespace stability
{
namespace analysis
{

enum class stability_transition_status
{
    success,
    no_transition,
    nonlinear_solver_failure,
    stability_solver_failure,
    unexpected_signature,
    invalid_input
};

inline const char* stability_transition_status_name(
    stability_transition_status status)
{
    switch(status)
    {
    case stability_transition_status::success:
        return "success";
    case stability_transition_status::no_transition:
        return "no_transition";
    case stability_transition_status::nonlinear_solver_failure:
        return "nonlinear_solver_failure";
    case stability_transition_status::stability_solver_failure:
        return "stability_solver_failure";
    case stability_transition_status::unexpected_signature:
        return "unexpected_signature";
    case stability_transition_status::invalid_input:
        return "invalid_input";
    }
    return "unknown";
}

template<class Real>
struct stability_transition_options
{
    unsigned int maximum_iterations = 15;
    Real parameter_tolerance = Real(0);
    bool correct_with_fixed_parameter_newton = true;
    bool recover_failed_classification_with_fixed_parameter_newton =
        true;
    bool recover_failed_newton_with_parameter_homotopy = true;
    unsigned int parameter_homotopy_maximum_subdivisions = 64;
    bool confirm_stability_classification = false;
    bool use_recycled_stability_subspace = false;
};

template<class Real>
struct stability_transition_result
{
    stability_transition_status status =
        stability_transition_status::invalid_input;
    Real parameter = Real{};
    stability_point_result<Real> stability;
    stability_point_result<Real> before_stability;
    stability_point_result<Real> after_stability;
    unsigned int iterations = 0;
    unsigned int consistency_restarts = 0;
    unsigned int parameter_homotopy_recoveries = 0;
    unsigned int parameter_homotopy_steps = 0;
    unsigned int fallback_newton_recoveries = 0;
    std::string diagnostic;

    bool succeeded() const
    {
        return status == stability_transition_status::success;
    }
};

template<class Real>
class stability_transition_error : public std::runtime_error
{
public:
    using result_type = stability_transition_result<Real>;

    stability_transition_error(
        std::string message,
        result_type result)
        : std::runtime_error(std::move(message)),
          result_(std::move(result))
    {
    }

    const result_type& result() const
    {
        return result_;
    }

private:
    result_type result_;
};

template<
    class VectorOperations,
    class NonlinearOperations,
    class Newton,
    class StabilityEvaluator>
class stability_transition_refiner
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using point_result_type = stability_point_result<scalar_type>;
    using options_type = stability_transition_options<scalar_type>;
    using result_type = stability_transition_result<scalar_type>;
    struct fixed_parameter_correction_result
    {
        bool succeeded = false;
        bool used_fallback = false;
        std::string diagnostic;
    };

    stability_transition_refiner(
        VectorOperations* vector_operations,
        NonlinearOperations* nonlinear_operations,
        Newton* newton,
        StabilityEvaluator* evaluator)
        : vector_operations_(vector_operations),
          nonlinear_operations_(nonlinear_operations),
          newton_(newton),
          evaluator_(evaluator),
          state_alignment_(vector_operations),
          lower_state_(vector_operations),
          upper_state_(vector_operations),
          midpoint_reference_(vector_operations),
          newton_initial_state_(vector_operations),
          homotopy_state_(vector_operations),
          homotopy_trial_(vector_operations)
    {
        if(vector_operations_ == nullptr)
            throw std::invalid_argument(
                "stability_transition_refiner: "
                "vector operations are null");
        if(nonlinear_operations_ == nullptr)
            throw std::invalid_argument(
                "stability_transition_refiner: "
                "nonlinear operator is null");
        if(newton_ == nullptr)
            throw std::invalid_argument(
                "stability_transition_refiner: Newton solver is null");
        if(evaluator_ == nullptr)
            throw std::invalid_argument(
                "stability_transition_refiner: evaluator is null");
    }

    template<class Aligner>
    void set_transition_state_aligner(Aligner* aligner)
    {
        state_alignment_.set(aligner);
    }

    void reset_transition_state_aligner()
    {
        state_alignment_.reset();
    }

    void align_state(
        const vector_type& reference,
        const vector_type& source,
        vector_type& destination) const
    {
        state_alignment_.align(reference, source, destination);
    }

    template<class FallbackNewton>
    void set_fallback_newton(FallbackNewton* fallback_newton)
    {
        if(fallback_newton == nullptr)
        {
            reset_fallback_newton();
            return;
        }
        fallback_newton_ =
            [this, fallback_newton](
                vector_type& state,
                scalar_type parameter)
            {
                return fallback_newton->solve(
                    nonlinear_operations_,
                    state,
                    parameter);
            };
    }

    void reset_fallback_newton()
    {
        fallback_newton_ = {};
    }

    fixed_parameter_correction_result correct_fixed_parameter_state(
        vector_type& state,
        scalar_type parameter)
    {
        fixed_parameter_correction_result result;
        result.succeeded = run_newton(
            state,
            parameter,
            result.diagnostic,
            result.used_fallback);
        return result;
    }

    result_type refine(
        const vector_type& lower_state,
        scalar_type lower_parameter,
        const point_result_type& lower_stability,
        const vector_type& upper_state,
        scalar_type upper_parameter,
        const point_result_type& upper_stability,
        vector_type& refined_state,
        options_type options = {})
    {
        result_type result;
        result.before_stability = lower_stability;
        result.after_stability = upper_stability;
        result.parameter =
            lower_parameter +
            scalar_type(0.5)*(upper_parameter - lower_parameter);
        const scalar_type original_lower_parameter =
            lower_parameter;
        const scalar_type original_upper_parameter =
            upper_parameter;

        if(options.maximum_iterations == 0 ||
           !lower_stability.succeeded() ||
           !upper_stability.succeeded())
        {
            result.status = stability_transition_status::invalid_input;
            result.diagnostic =
                "transition refinement requires two classified "
                "endpoints and at least one iteration";
            return result;
        }
        const int lower_subspace_dimension =
            lower_stability.unstable.real_subspace_dimension();
        const int upper_subspace_dimension =
            upper_stability.unstable.real_subspace_dimension();
        if(lower_subspace_dimension == upper_subspace_dimension)
        {
            result.status = stability_transition_status::no_transition;
            result.diagnostic =
                "endpoint real unstable-subspace dimensions are equal";
            return result;
        }

        vector_operations_->assign(
            lower_state,
            lower_state_.get());
        state_alignment_.align(
            lower_state_.get(),
            upper_state,
            upper_state_.get());
        for(unsigned int iteration = 0;
            iteration < options.maximum_iterations;
            ++iteration)
        {
            result.parameter =
                lower_parameter +
                scalar_type(0.5)*
                    (upper_parameter - lower_parameter);
            vector_operations_->assign_mul(
                scalar_type(0.5),
                lower_state_.get(),
                scalar_type(0.5),
                upper_state_.get(),
                refined_state);
            vector_operations_->assign(
                refined_state,
                midpoint_reference_.get());

            bool state_corrected = false;
            if(options.correct_with_fixed_parameter_newton)
            {
                const state_correction_result correction =
                    correct_state(
                        refined_state,
                        result.parameter,
                        lower_state_.get(),
                        lower_parameter,
                        upper_state_.get(),
                        upper_parameter,
                        options);
                if(!correction.succeeded)
                {
                    result.status =
                        stability_transition_status::
                            nonlinear_solver_failure;
                    result.diagnostic = correction.diagnostic;
                    return result;
                }
                record_state_correction(result, correction);
                if(!correction.used_parameter_homotopy)
                {
                    state_alignment_.align(
                        midpoint_reference_.get(),
                        refined_state,
                        refined_state);
                }
                state_corrected = true;
            }

            if(!options.use_recycled_stability_subspace)
                evaluator_->reset_recycled_ritz_subspace();
            result.stability = evaluator_->analyze(
                refined_state,
                result.parameter);
            result.iterations = iteration + 1;

            using std::abs;
            const bool final_midpoint =
                iteration + 1 == options.maximum_iterations ||
                (
                    options.parameter_tolerance > scalar_type(0) &&
                    scalar_type(0.5)*
                        abs(upper_parameter - lower_parameter) <=
                        options.parameter_tolerance);
            bool unexpected_signature = false;
            if(result.stability.succeeded())
            {
                const int provisional_dimension =
                    result.stability.unstable.
                        real_subspace_dimension();
                unexpected_signature =
                    provisional_dimension !=
                        lower_subspace_dimension &&
                    provisional_dimension !=
                        upper_subspace_dimension;
            }
            if(
                options.confirm_stability_classification &&
                (
                    !result.stability.succeeded() ||
                    unexpected_signature ||
                    final_midpoint))
            {
                if(!options.use_recycled_stability_subspace)
                    evaluator_->reset_recycled_ritz_subspace();
                result.stability = evaluator_->analyze_confirmed(
                    refined_state,
                    result.parameter);
            }
            if(
                !result.stability.succeeded() &&
                !state_corrected &&
                options.
                    recover_failed_classification_with_fixed_parameter_newton)
            {
                const std::string uncorrected_diagnostic =
                    result.stability.diagnostic;
                vector_operations_->assign(
                    midpoint_reference_.get(),
                    refined_state);
                const state_correction_result correction =
                    correct_state(
                        refined_state,
                        result.parameter,
                        lower_state_.get(),
                        lower_parameter,
                        upper_state_.get(),
                        upper_parameter,
                        options);
                if(!correction.succeeded)
                {
                    result.status =
                        stability_transition_status::
                            nonlinear_solver_failure;
                    result.diagnostic =
                        "stability classification failed on the "
                        "secant state and fixed-parameter Newton "
                        "recovery failed: " +
                        correction.diagnostic +
                        "; secant classification: " +
                        uncorrected_diagnostic;
                    return result;
                }
                record_state_correction(result, correction);
                if(!correction.used_parameter_homotopy)
                {
                    state_alignment_.align(
                        midpoint_reference_.get(),
                        refined_state,
                        refined_state);
                }
                state_corrected = true;
                ++result.consistency_restarts;

                if(!options.use_recycled_stability_subspace)
                    evaluator_->reset_recycled_ritz_subspace();
                result.stability =
                    options.confirm_stability_classification
                    ? evaluator_->analyze_confirmed(
                          refined_state,
                          result.parameter)
                    : evaluator_->analyze(
                          refined_state,
                          result.parameter);
                if(result.stability.succeeded())
                {
                    result.diagnostic =
                        "recovered failed secant-state stability "
                        "classification with fixed-parameter Newton";
                }
                else
                {
                    result.stability.diagnostic =
                        "secant classification failed: {" +
                        uncorrected_diagnostic +
                        "}; corrected-state classification failed: {" +
                        result.stability.diagnostic + "}";
                }
            }
            if(!result.stability.succeeded())
            {
                result.status =
                    stability_transition_status::
                        stability_solver_failure;
                std::ostringstream diagnostic;
                diagnostic
                    << "stability classification failed at lambda = "
                    << std::setprecision(16)
                    << result.parameter << ": "
                    << result.stability.diagnostic;
                result.diagnostic = diagnostic.str();
                return result;
            }

            int intermediate_subspace_dimension =
                result.stability.unstable.real_subspace_dimension();
            if(
                intermediate_subspace_dimension !=
                    lower_subspace_dimension &&
                intermediate_subspace_dimension !=
                    upper_subspace_dimension &&
                !state_corrected)
            {
                const std::string secant_diagnostic =
                    unexpected_signature_diagnostic(
                        result.stability,
                        lower_subspace_dimension,
                        upper_subspace_dimension);
                options_type corrected_options = options;
                corrected_options.
                    correct_with_fixed_parameter_newton = true;
                result_type corrected = refine(
                    lower_state,
                    original_lower_parameter,
                    lower_stability,
                    upper_state,
                    original_upper_parameter,
                    upper_stability,
                    refined_state,
                    corrected_options);
                ++corrected.consistency_restarts;
                if(corrected.succeeded())
                {
                    corrected.diagnostic =
                        "restarted transition refinement with "
                        "fixed-parameter Newton after inconsistent "
                        "secant classification: " +
                        secant_diagnostic;
                }
                else
                {
                    corrected.diagnostic =
                        secant_diagnostic +
                        "; corrected refinement also failed: " +
                        corrected.diagnostic;
                }
                return corrected;
            }

            if(
                intermediate_subspace_dimension ==
                upper_subspace_dimension)
            {
                upper_parameter = result.parameter;
                vector_operations_->assign(
                    refined_state,
                    upper_state_.get());
                result.after_stability = result.stability;
            }
            else if(
                intermediate_subspace_dimension ==
                lower_subspace_dimension)
            {
                lower_parameter = result.parameter;
                vector_operations_->assign(
                    refined_state,
                    lower_state_.get());
                result.before_stability = result.stability;
            }
            else
            {
                result.status =
                    stability_transition_status::
                        unexpected_signature;
                result.diagnostic =
                    unexpected_signature_diagnostic(
                        result.stability,
                        lower_subspace_dimension,
                        upper_subspace_dimension);
                return result;
            }

            if(options.parameter_tolerance > scalar_type(0) &&
               abs(upper_parameter - lower_parameter) <=
                   options.parameter_tolerance)
                break;
        }

        result.status = stability_transition_status::success;
        return result;
    }

private:
    struct state_correction_result
    {
        bool succeeded = false;
        bool used_parameter_homotopy = false;
        unsigned int subdivisions = 0;
        unsigned int fallback_newton_recoveries = 0;
        std::string diagnostic;
    };

    static void record_state_correction(
        result_type& transition,
        const state_correction_result& correction)
    {
        transition.fallback_newton_recoveries +=
            correction.fallback_newton_recoveries;
        if(correction.used_parameter_homotopy)
        {
            ++transition.parameter_homotopy_recoveries;
            transition.parameter_homotopy_steps +=
                correction.subdivisions;
        }
    }

    bool run_newton(
        vector_type& state,
        scalar_type parameter,
        std::string& diagnostic,
        bool& used_fallback)
    {
        used_fallback = false;
        vector_operations_->assign(
            state,
            newton_initial_state_.get());
        bool converged = false;
        std::string primary_diagnostic;
        try
        {
            converged = newton_->solve(
                nonlinear_operations_,
                state,
                parameter);
        }
        catch(const std::exception& error)
        {
            primary_diagnostic = error.what();
        }
        catch(...)
        {
            primary_diagnostic =
                "Newton solver raised an unknown exception";
        }
        if(converged)
            return true;
        if(primary_diagnostic.empty())
        {
            primary_diagnostic =
                "Newton solver failed during transition refinement";
        }
        if(!fallback_newton_)
        {
            diagnostic = primary_diagnostic;
            return false;
        }

        vector_operations_->assign(
            newton_initial_state_.get(),
            state);
        std::string fallback_diagnostic;
        try
        {
            converged = fallback_newton_(state, parameter);
        }
        catch(const std::exception& error)
        {
            fallback_diagnostic = error.what();
        }
        catch(...)
        {
            fallback_diagnostic =
                "fallback Newton solver raised an unknown exception";
        }
        if(converged)
        {
            used_fallback = true;
            diagnostic =
                "primary Newton failed: {" +
                primary_diagnostic +
                "}; fallback Newton recovered the state";
            return true;
        }
        vector_operations_->assign(
            newton_initial_state_.get(),
            state);
        if(fallback_diagnostic.empty())
        {
            fallback_diagnostic =
                "fallback Newton solver failed during transition "
                "refinement";
        }
        diagnostic =
            "primary Newton failed: {" + primary_diagnostic +
            "}; fallback Newton failed: {" +
            fallback_diagnostic + "}";
        return false;
    }

    bool march_from_anchor(
        const vector_type& anchor_state,
        scalar_type anchor_parameter,
        scalar_type target_parameter,
        unsigned int maximum_subdivisions,
        vector_type& destination,
        state_correction_result& result)
    {
        if(maximum_subdivisions < 2)
            return false;

        std::string last_failure;
        unsigned int subdivisions = 2;
        while(true)
        {
            vector_operations_->assign(
                anchor_state,
                homotopy_state_.get());
            bool succeeded = true;
            for(unsigned int step = 1;
                step <= subdivisions;
                ++step)
            {
                const scalar_type fraction =
                    scalar_type(step)/scalar_type(subdivisions);
                const scalar_type parameter =
                    anchor_parameter +
                    fraction*(target_parameter - anchor_parameter);
                vector_operations_->assign(
                    homotopy_state_.get(),
                    homotopy_trial_.get());
                std::string step_diagnostic;
                bool used_fallback = false;
                if(!run_newton(
                       homotopy_trial_.get(),
                       parameter,
                       step_diagnostic,
                       used_fallback))
                {
                    std::ostringstream diagnostic;
                    diagnostic
                        << "subdivisions=" << subdivisions
                        << ", step=" << step
                        << ", parameter=" << std::setprecision(16)
                        << parameter << ": " << step_diagnostic;
                    last_failure = diagnostic.str();
                    succeeded = false;
                    break;
                }
                if(used_fallback)
                    ++result.fallback_newton_recoveries;
                state_alignment_.align(
                    homotopy_state_.get(),
                    homotopy_trial_.get(),
                    homotopy_trial_.get());
                vector_operations_->assign(
                    homotopy_trial_.get(),
                    homotopy_state_.get());
            }
            if(succeeded)
            {
                vector_operations_->assign(
                    homotopy_state_.get(),
                    destination);
                result.succeeded = true;
                result.used_parameter_homotopy = true;
                result.subdivisions = subdivisions;
                std::ostringstream diagnostic;
                diagnostic
                    << "fixed-parameter Newton recovered by "
                    << subdivisions
                    << " homotopy step(s) from parameter "
                    << std::setprecision(16) << anchor_parameter
                    << " to " << target_parameter;
                result.diagnostic = diagnostic.str();
                return true;
            }
            if(subdivisions == maximum_subdivisions)
                break;
            subdivisions = std::min(
                maximum_subdivisions,
                2U*subdivisions);
        }
        result.diagnostic = last_failure;
        return false;
    }

    state_correction_result correct_state(
        vector_type& state,
        scalar_type parameter,
        const vector_type& lower_state,
        scalar_type lower_parameter,
        const vector_type& upper_state,
        scalar_type upper_parameter,
        const options_type& options)
    {
        state_correction_result result;
        std::string direct_diagnostic;
        bool used_fallback = false;
        if(run_newton(
               state,
               parameter,
               direct_diagnostic,
               used_fallback))
        {
            result.succeeded = true;
            result.fallback_newton_recoveries =
                used_fallback ? 1U : 0U;
            if(used_fallback)
                result.diagnostic = direct_diagnostic;
            return result;
        }
        if(
            !options.recover_failed_newton_with_parameter_homotopy ||
            options.parameter_homotopy_maximum_subdivisions < 2)
        {
            result.diagnostic = direct_diagnostic;
            return result;
        }

        using std::abs;
        const bool lower_is_closer =
            abs(parameter - lower_parameter) <=
            abs(upper_parameter - parameter);
        const vector_type* first_state =
            lower_is_closer ? &lower_state : &upper_state;
        const scalar_type first_parameter =
            lower_is_closer ? lower_parameter : upper_parameter;
        const vector_type* second_state =
            lower_is_closer ? &upper_state : &lower_state;
        const scalar_type second_parameter =
            lower_is_closer ? upper_parameter : lower_parameter;

        state_correction_result first_result;
        if(march_from_anchor(
               *first_state,
               first_parameter,
               parameter,
               options.parameter_homotopy_maximum_subdivisions,
               state,
               first_result))
            return first_result;

        state_correction_result second_result;
        if(march_from_anchor(
               *second_state,
               second_parameter,
               parameter,
               options.parameter_homotopy_maximum_subdivisions,
               state,
               second_result))
            return second_result;

        result.diagnostic =
            "direct fixed-parameter Newton failed: {" +
            direct_diagnostic +
            "}; parameter homotopy failed from parameter " +
            std::to_string(first_parameter) + ": {" +
            first_result.diagnostic +
            "}; parameter homotopy failed from parameter " +
            std::to_string(second_parameter) + ": {" +
            second_result.diagnostic + "}";
        return result;
    }

    static std::string unexpected_signature_diagnostic(
        const point_result_type& stability,
        int lower_subspace_dimension,
        int upper_subspace_dimension)
    {
        const auto signature = stability.unstable.as_pair();
        return
            "an intermediate point has unstable signature (" +
            std::to_string(signature.first) + "," +
            std::to_string(signature.second) +
            ") with real unstable-subspace dimension " +
            std::to_string(
                stability.unstable.real_subspace_dimension()) +
            ", different from endpoint dimensions " +
            std::to_string(lower_subspace_dimension) +
            " and " +
            std::to_string(upper_subspace_dimension) +
            "; the interval may contain multiple stability "
            "transitions";
    }

    VectorOperations* vector_operations_;
    NonlinearOperations* nonlinear_operations_;
    Newton* newton_;
    StabilityEvaluator* evaluator_;
    transition_state_alignment<VectorOperations>
        state_alignment_;
    detail::vector_workspace<VectorOperations> lower_state_;
    detail::vector_workspace<VectorOperations> upper_state_;
    detail::vector_workspace<VectorOperations>
        midpoint_reference_;
    detail::vector_workspace<VectorOperations>
        newton_initial_state_;
    detail::vector_workspace<VectorOperations> homotopy_state_;
    detail::vector_workspace<VectorOperations> homotopy_trial_;
    std::function<bool(vector_type&, scalar_type)>
        fallback_newton_;
};

} // namespace analysis
} // namespace stability

#endif
