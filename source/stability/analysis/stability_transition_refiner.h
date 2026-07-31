#ifndef __STABILITY_ANALYSIS_STABILITY_TRANSITION_REFINER_H__
#define __STABILITY_ANALYSIS_STABILITY_TRANSITION_REFINER_H__

#include <cmath>
#include <cstddef>
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
    bool confirm_stability_classification = false;
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
    std::string diagnostic;

    bool succeeded() const
    {
        return status == stability_transition_status::success;
    }
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
          midpoint_reference_(vector_operations)
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
                std::string correction_diagnostic;
                if(!correct_state(
                       refined_state,
                       result.parameter,
                       correction_diagnostic))
                {
                    result.status =
                        stability_transition_status::
                            nonlinear_solver_failure;
                    result.diagnostic = correction_diagnostic;
                    return result;
                }
                state_alignment_.align(
                    midpoint_reference_.get(),
                    refined_state,
                    refined_state);
                state_corrected = true;
            }

            result.stability =
                options.confirm_stability_classification
                ? evaluator_->analyze_confirmed(
                      refined_state,
                      result.parameter)
                : evaluator_->analyze(
                      refined_state,
                      result.parameter);
            result.iterations = iteration + 1;
            if(!result.stability.succeeded())
            {
                result.status =
                    stability_transition_status::
                        stability_solver_failure;
                result.diagnostic = result.stability.diagnostic;
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

            using std::abs;
            if(options.parameter_tolerance > scalar_type(0) &&
               abs(upper_parameter - lower_parameter) <=
                   options.parameter_tolerance)
                break;
        }

        result.status = stability_transition_status::success;
        return result;
    }

private:
    bool correct_state(
        vector_type& state,
        scalar_type parameter,
        std::string& diagnostic)
    {
        bool converged = false;
        try
        {
            converged = newton_->solve(
                nonlinear_operations_,
                state,
                parameter);
        }
        catch(const std::exception& error)
        {
            diagnostic = error.what();
            return false;
        }
        catch(...)
        {
            diagnostic =
                "Newton solver raised an unknown exception";
            return false;
        }
        if(!converged)
        {
            diagnostic =
                "Newton solver failed during transition refinement";
            return false;
        }
        return true;
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
};

} // namespace analysis
} // namespace stability

#endif
