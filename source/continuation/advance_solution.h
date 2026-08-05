#ifndef __CONTINUATION__ADVANCE_SOLUTION_H__
#define __CONTINUATION__ADVANCE_SOLUTION_H__

#include <string>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <common/scalar_math.h>
#include <continuation/chart_helpers.h>
#include <continuation/continuation_step_state.h>
#include <continuation/corrector_retry_policy.h>
#include <continuation/predictor_chart_diagnostics.h>
#include <continuation/predictor_chart_probe.h>
#include <continuation/tangent_normalization.h>
#include <nonlinear_operators/projected_operator_helpers.h>
#include <numerical_algos/detail/str_source_helper.h>
/**
  continuation of a single solution forward or backward on a single step
  execute SOLVE method to continue solution in one step
*/

namespace continuation
{

template<class VectorOperations, class Loggin, class NewtonMethodExtended, class NewtonMethod, class NonlinearOperator, class SystemOperator, class Predictoror, class ConvergenceNewtonExtended>
class advance_solution
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    
    advance_solution(VectorOperations* vec_ops_, Loggin* log_, SystemOperator* sys_op_, NewtonMethodExtended* newton_extended_, NewtonMethod* newton_, Predictoror* predictor_, ConvergenceNewtonExtended* convergence_newton_extended_, char continuation_type_ = 'S'):
    vec_ops(vec_ops_),
    log(log_),
    sys_op(sys_op_),
    newton_extended(newton_extended_),
    newton(newton_),
    predictor(predictor_),
    continuation_type(continuation_type_),
    convergence_newton_extended(convergence_newton_extended_)
    {
        vec_ops->init_vector(x_p); vec_ops->start_use_vector(x_p);
        vec_ops->init_vector(x1_l); vec_ops->start_use_vector(x1_l);
        vec_ops->init_vector(dx10);  vec_ops->start_use_vector(dx10);
        if constexpr(chart::has_isotropy_transition<NonlinearOperator, T_vec, T>::value)
        {
            vec_ops->init_vector(x_isotropy_event);
            vec_ops->start_use_vector(x_isotropy_event);
        }
    }

    ~advance_solution()
    {
        if constexpr(chart::has_isotropy_transition<NonlinearOperator, T_vec, T>::value)
        {
            vec_ops->stop_use_vector(x_isotropy_event);
            vec_ops->free_vector(x_isotropy_event);
        }
        vec_ops->stop_use_vector(dx10); vec_ops->free_vector(dx10);
        vec_ops->stop_use_vector(x1_l); vec_ops->free_vector(x1_l);
        vec_ops->stop_use_vector(x_p); vec_ops->free_vector(x_p);
    }
    

    void reset() //can be used to set everything in default state
    {
        predictor->reset_all();
        last_isotropy_transition_ = {};
    }

    void set_verbose(const bool value)
    {
        verbose = value;
        set_predictor_verbose(predictor, value, 0);
    }

    void set_predictor_chart_policy(const predictor_chart_policy<T>& policy)
    {
        validate_predictor_chart_policy(policy);
        chart_policy = policy;
    }

    void set_isotropy_transition_policy(
        const symmetry::continuation::isotropy_transition_policy<T>& policy)
    {
        policy.validate();
        isotropy_policy = policy;
    }

    bool has_isotropy_transition() const
    {
        return last_isotropy_transition_.detected;
    }

    const symmetry::continuation::isotropy_transition_result<T>&
    last_isotropy_transition() const
    {
        return last_isotropy_transition_;
    }

    const continuation_step_attempt_state<T>& last_attempt_state() const
    {
        return last_attempt_state_;
    }

    step_retry_result reduce_next_step(const T factor)
    {
        return predictor->reduce_next_step(factor);
    }

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x0, const T& lambda0, const T_vec& x0_s, const T& lambda0_s, T_vec& x1, T& lambda1, T_vec& x1_s, T& lambda1_s)
    {
        last_isotropy_transition_ = {};
        bool converged = false;
        bool failed = false;
        bool terminal_isotropy_transition = false;
        continuation_step_attempt_state<T> attempt;
        last_attempt_state_ = attempt;
        isotropy_refinement_bracket<T> isotropy_bracket;
        T lambda_p;
        predictor_chart_policy<T> effective_chart_policy = chart_policy;
        if constexpr(!chart::has_continuation_chart<NonlinearOperator, T_vec, T>::value)
        {
            effective_chart_policy.enabled = false;
        }
        if(verbose)
        {
            log->info("continuation::advance_solution::starting point:");
            log->info_f("   ||x0|| = %le, lambda0 = %le, ||x0_s|| = %le, lambda0_s = %le", (double)vec_ops->norm(x0), (double)lambda0, (double)vec_ops->norm(x0_s), (double)lambda0_s);
        }
        chart::begin_continuation_chart(vec_ops, log, nonlin_op, x0, lambda0, x0_s, lambda0_s);
        if(verbose)
        {
            chart::log_continuation_chart(log, nonlin_op, "continuation::advance_solution::begin_chart");
        }
        predictor->reset_tangent_space(x0, lambda0, x0_s, lambda0_s);
        while((!converged)&&(!failed))
        {
            predictor->apply(x_p, lambda_p, x1, lambda1);
            chart::stabilize_predictor_for_continuation(
                vec_ops,
                log,
                nonlin_op,
                x0,
                lambda0,
                x0_s,
                lambda0_s,
                x_p,
                lambda_p,
                x1,
                lambda1);
            T ds_l = predictor->get_ds();
            T ds_max = predictor->get_ds_max();
            T tangent_norm = vec_ops->norm_rank1(x0_s, lambda0_s);
            const auto predictor_probe = evaluate_predictor_chart(
                vec_ops,
                x0,
                lambda0,
                x0_s,
                lambda0_s,
                x_p,
                lambda_p,
                x1,
                lambda1,
                dx10,
                ds_l,
                effective_chart_policy);
            const auto& predictor_validation = predictor_probe.validation;
            if(verbose)
            {
                log->info_f("continuation::predict: dS = %le, max dS = %le, tangent norm = %le, ||x_p|| = %le, lambda_p = %le, ||x1|| = %le, lambda1 = %le, ||x1 - x_p|| = %le", (double)ds_l, (double)ds_max, (double)tangent_norm, (double)vec_ops->norm(x_p), (double)lambda_p, (double)vec_ops->norm(x1), (double)lambda1, (double)predictor_probe.chart_displacement);
                log->info_f(
                    "continuation::predict: tangent progress diagnostics: raw = %le (x = %le, lambda = %le), charted = %le (x = %le, lambda = %le), charted/raw = %le",
                    (double)predictor_probe.raw_tangent_progress,
                    (double)predictor_probe.raw_x_progress,
                    (double)predictor_probe.raw_lambda_progress,
                    (double)predictor_probe.charted_tangent_progress,
                    (double)predictor_probe.charted_x_progress,
                    (double)predictor_probe.charted_lambda_progress,
                    (double)(predictor_probe.raw_tangent_progress == T(0) ? T(0) : predictor_probe.charted_tangent_progress/predictor_probe.raw_tangent_progress));
            }
            log_predictor_chart_validation_warnings(
                log,
                ds_l,
                predictor_probe.raw_tangent_progress,
                predictor_probe.charted_tangent_progress,
                predictor_probe.chart_displacement,
                predictor_validation);
            if(verbose)
            {
                chart::log_continuation_chart(log, nonlin_op, "continuation::advance_solution::predictor");
            }
            if(predictor_validation.decision == predictor_chart_decision::reject_tangent)
            {
                if(isotropy_bracket.active())
                {
                    terminal_isotropy_transition = true;
                    converged = true;
                    break;
                }
                attempt.failure = continuation_failure_kind::tangent;
                attempt.failure_reason =
                    "predictor validation rejected the current tangent";
                attempt.attempted_step = predictor->get_ds();
                attempt.retry_count = predictor_retry_count(predictor, 0);
                last_attempt_state_ = attempt;
                throw std::runtime_error(
                    "continuation::advance_solution: predictor validation rejected the current tangent");
            }
            if(predictor_validation.decision == predictor_chart_decision::reject_chart)
            {
                if(attempt.chart_retries >= chart_policy.maximum_retries)
                {
                    if(isotropy_bracket.active())
                    {
                        terminal_isotropy_transition = true;
                        converged = true;
                        break;
                    }
                    failed = true;
                    attempt.failure = continuation_failure_kind::predictor_chart;
                    attempt.failure_reason = "predictor chart validation exhausted its retry budget";
                    break;
                }
                chart::restore_continuation_chart(
                    vec_ops,
                    log,
                    nonlin_op,
                    x0,
                    lambda0,
                    x0_s,
                    lambda0_s);
                const auto retry_result =
                    predictor->retry_after_chart_rejection(chart_policy.step_reduction_factor);
                ++attempt.chart_retries;
                attempt.any_chart_retries = true;
                if(retry_result != step_retry_result::retry)
                {
                    if(isotropy_bracket.active())
                    {
                        terminal_isotropy_transition = true;
                        converged = true;
                        break;
                    }
                    failed = true;
                    attempt.failure = continuation_failure_kind::predictor_chart;
                    attempt.failure_reason = retry_result == step_retry_result::retry_limit
                        ? "continuation retry limit reached after chart rejection"
                        : "minimum continuation step reached after chart rejection";
                    break;
                }
                log->warning_f(
                    "continuation::advance_solution: rejected charted predictor; retry %u of %u with dS = %le.",
                    attempt.chart_retries,
                    chart_policy.maximum_retries,
                    (double)predictor->get_ds());
                continue;
            }
            if(continuation_type == 'S')
            {
                sys_op->set_tangent_space((T_vec&)x0, (T&)lambda0, (T_vec&)x0_s, (T&)lambda0_s, ds_l, continuation_type, nonlin_op);
            }
            else if(continuation_type == 'O')
            {
                sys_op->set_tangent_space(x_p, lambda_p, (T_vec&)x0_s, (T&)lambda0_s, ds_l, continuation_type, nonlin_op);
            }
            else
            {
                throw std::runtime_error(std::string("continuation::advance_solution (corrector) " __FILE__ " " __STR(__LINE__) " incorrect continuation_type parameter. Only 'S'pherical or 'O'rthogonal can be used") );
            }
            converged = newton_extended->solve(nonlin_op, x1, lambda1);
            if(!converged)
            {
                if(verbose)
                {
                    log->info("continuation::advance_solution failed to converge; reducing dS.");
                }
                chart::restore_continuation_chart(
                    vec_ops,
                    log,
                    nonlin_op,
                    x0,
                    lambda0,
                    x0_s,
                    lambda0_s);
                const auto retry_result = predictor->retry_after_failure();
                attempt.any_corrector_retries = true;
                if(retry_result != step_retry_result::retry)
                {
                    if(isotropy_bracket.active())
                    {
                        terminal_isotropy_transition = true;
                        converged = true;
                        break;
                    }
                    failed = true;
                    attempt.failure = retry_result == step_retry_result::retry_limit
                        ? continuation_failure_kind::corrector_retry_limit
                        : continuation_failure_kind::minimum_step;
                    attempt.failure_reason = retry_result == step_retry_result::retry_limit
                        ? "continuation corrector retry limit reached"
                        : "minimum continuation step reached after corrector failure";
                }
            }
            else
            {
                if(verbose)
                {
                    if(attempt.any_corrector_retries)
                        log->info("continuation::advance_solution converged after corrector retries.");
                    else
                        log->info("continuation::advance_solution converged on the first corrector attempt.");
                }
            }
            if(converged && isotropy_policy.enabled &&
               chart::has_isotropy_transition<NonlinearOperator, T_vec, T>::value)
            {
                auto transition = chart::detect_isotropy_transition(
                    nonlin_op,
                    x0,
                    x1,
                    isotropy_policy);
                if(transition.detected)
                {
                    if(isotropy_bracket.latch_event(predictor->get_ds(), lambda1))
                    {
                        vec_ops->assign(x1, x_isotropy_event);
                        last_isotropy_transition_ = transition;
                    }
                    if(isotropy_policy.verbose)
                    {
                        log->warning_f(
                            "continuation::advance_solution: corrected trial increases isotropy C_%lu -> C_%lu; transverse ratio changed from %le to %le at lambda = %le, dS = %le.",
                            static_cast<unsigned long>(transition.previous_order),
                            static_cast<unsigned long>(transition.candidate_order),
                            double(transition.previous_transverse_ratio),
                            double(transition.candidate_transverse_ratio),
                            double(lambda1),
                            double(predictor->get_ds()));
                    }
                }
                else if(isotropy_bracket.active())
                {
                    isotropy_bracket.observe_non_event(predictor->get_ds());
                }

                if(isotropy_bracket.active())
                {
                    chart::restore_continuation_chart(
                        vec_ops,
                        log,
                        nonlin_op,
                        x0,
                        lambda0,
                        x0_s,
                        lambda0_s);

                    T next_step = T(0);
                    if(isotropy_bracket.next_step(isotropy_policy, next_step))
                    {
                        const auto retry_result = predictor->retry_at_step(next_step);
                        if(retry_result == step_retry_result::retry)
                        {
                            attempt.any_isotropy_retries = true;
                            converged = false;
                            if(isotropy_policy.verbose)
                            {
                                log->warning_f(
                                    "continuation::advance_solution: refining latched isotropy transition, attempt %u of %u in dS bracket [%le, %le], trial dS = %le.",
                                    isotropy_bracket.refinements(),
                                    isotropy_policy.maximum_refinements,
                                    double(isotropy_bracket.non_event_step()),
                                    double(isotropy_bracket.event_step()),
                                    double(next_step));
                            }
                            continue;
                        }
                    }

                    vec_ops->assign(x_isotropy_event, x1);
                    lambda1 = isotropy_bracket.event_lambda();
                    last_isotropy_transition_.refinements = isotropy_bracket.refinements();
                    terminal_isotropy_transition = true;
                    break;
                }
            }
        }
        if(converged && verbose)
        {
            log->info("continuation::advance_solution::corrector Newton step norms:");
            for(auto& x: *newton_extended->get_convergence_strategy_handle()->get_norms_history_handle())
            {
                
                log->info_f("%le",(double)x);
            }                
        }
        if(failed)
        {
            attempt.attempted_step = predictor->get_ds();
            attempt.retry_count = predictor_retry_count(predictor, 0);
            last_attempt_state_ = attempt;
            throw std::runtime_error(
                std::string("continuation::advance_solution (corrector) " __FILE__ " " __STR(__LINE__) " failed: ") +
                (attempt.failure_reason.empty() ? "unknown retry failure" : attempt.failure_reason));
        }
        if(terminal_isotropy_transition)
        {
            last_isotropy_transition_.refinements = isotropy_bracket.refinements();
            vec_ops->assign(x_isotropy_event, x1);
            lambda1 = isotropy_bracket.event_lambda();
            vec_ops->assign(x0_s, x1_s);
            lambda1_s = lambda0_s;
            return true;
        }
        bool tangent_obtained = false;

	    if(converged)
        {
            T arclength_res = sys_op->arclength_residual(x1, lambda1);
            T tangent_norm = vec_ops->norm_rank1(x0_s, lambda0_s);
            if(verbose)
            {
                log->info_f("continuation::advance_solution::corrected state: dS = %le, tangent norm = %le, arclength residual = %le", (double)predictor->get_ds(), (double)tangent_norm, (double)arclength_res);
            }
            tangent_obtained = sys_op->update_tangent_space(nonlin_op, x1, lambda1, x1_s, lambda1_s);
        }
        if((converged)&&(!tangent_obtained))
        {
            // throw std::runtime_error(std::string("advance_solution::advance_solution (tangent) " __FILE__ " " __STR(__LINE__) " linear system failed to converge.") );
            log->warning("continuation::advance_solution::tangent system failed to converge; using a finite-difference estimate.");
            T ds = predictor->get_ds();
            T d_ds = T(10.0*std::sqrt(2.0)*1.0e-6);
            //T ds_p = ds + d_ds;
            T ds_m = ds - d_ds;
            //T ds_factor_p = ds_p/ds;
            T ds_factor_m = ds_m/ds;
            //x1-x0=dx10
            vec_ops->assign_mul(T(1.0), x1, T(-1.0), x0, dx10);
            //minus_point
            vec_ops->assign_mul(ds_factor_m, dx10, T(1.0), x0, x1_l);
            T lambda1_l = ds_factor_m*(lambda1 - lambda0) + lambda0;

            bool converged_p, converged_m;
            if(continuation_type == 'S')
            {
                sys_op->set_tangent_space((T_vec&)x0, (T&)lambda0, (T_vec&)x0_s, (T&)lambda0_s, ds_m, continuation_type, nonlin_op);
            }
            else if(continuation_type == 'O')
            {
                sys_op->set_tangent_space(x1_l, lambda1_l, (T_vec&)x0_s, (T&)lambda0_s, ds_m, continuation_type, nonlin_op);
            }
            else
            {
                throw std::runtime_error(std::string("continuation::advance_solution (tnagent) " __FILE__ " " __STR(__LINE__) " incorrect continuation_type parameter. Only 'S'pherical or 'O'rthogonal can be used") );
            }
            converged_m = newton_extended->solve(nonlin_op, x1_l, lambda1_l);
            if(converged_m)
            {
                lambda1_s = (lambda1 - lambda1_l)/d_ds;
                vec_ops->assign_mul(T(-1)/d_ds, x1_l, T(1)/d_ds, x1, x1_s);    
                tangent_obtained = normalize_rank1_tangent(
                    vec_ops,
                    x1_s,
                    lambda1_s);
                if(tangent_obtained && verbose)
                {
                    log->info_f("continuation::advance_solution::||(x_s, l_s)|| = %le", (double)(lambda1_s*lambda1_s + vec_ops->scalar_prod(x1_s, x1_s)) );
                }
                if(!tangent_obtained)
                {
                    log->warning(
                        "continuation::advance_solution: finite-difference tangent was zero or non-finite.");
                }
            }
            if(!tangent_obtained)
            {
                
                log->warning("continuation::advance_solution could not produce a usable additional tangent point");
                if(verbose)
                {
                    log->info("continuation::advance_solution using Newton-Raphson estimation.");
                }
                const T x_norm = vec_ops->norm(x1);
                const T delta_lambda = lambda1 - lambda0;
                T sign = delta_lambda < T(0) ? T(-1) : T(1);
                if(common::scalar_math::abs(delta_lambda) <=
                   T(16)*std::numeric_limits<T>::epsilon() && lambda0_s < T(0))
                {
                    sign = T(-1);
                }
                const T d_lambda_magnitude =
                    common::scalar_math::isfinite(x_norm) &&
                    x_norm > T(16)*std::numeric_limits<T>::epsilon()
                        ? T(1)/x_norm
                        : T(1.0e-4)*(T(1) + common::scalar_math::abs(lambda1));
                const T d_lambda = sign*d_lambda_magnitude;
                lambda1_l = lambda1 + d_lambda;
                vec_ops->assign(x1, x1_l); //guess for x1 
                bool converged = newton->solve(nonlin_op, x1_l, lambda1_l);
                if(!converged)
                {
                    //newton method failed to converge!
                    //throw std::runtime_error(std::string("continuation::initial_tangent " __FILE__ " " __STR(__LINE__) " tangent space couldn't be obtained - Newton method failed to converge.") );
                    // reset(); //resets predictor step!
                    log->error("continuation::advance_solution: Newton-Raphson failed to converged. Nothing can be done so far, setting estimation equal to the previous step.");
                    vec_ops->assign(x0_s, x1_s);
                    lambda1_s = lambda0_s;
                    tangent_obtained = normalize_rank1_tangent(
                        vec_ops,
                        x1_s,
                        lambda1_s);
                }
                else
                {
                    lambda1_s = lambda1_l - lambda1; //lambda_s = ds*d(lambda)/ds
                    //x_s = x1 - x;      
                    vec_ops->assign_mul(T(1.0), x1_l, T(-1.0), x1, x1_s);  //x_s = ds*d(x)/ds
                    T ds_l = vec_ops->norm_rank1(x1_s, lambda1_s); 
                    tangent_obtained = normalize_rank1_tangent(
                        vec_ops,
                        x1_s,
                        lambda1_s);
                    if(tangent_obtained && verbose)
                    {
                        log->info_f("continuation::advance_solution: estimated local ds = %le", (double) ds_l);
                        log->info("continuation::advance_solution: Newton-Raphson estimate ends successfully.");
                    }
                }



            }
            if(tangent_obtained)
            {
                nonlinear_operators::detail::project_current_tangent(
                    vec_ops,
                    nonlin_op,
                    x1_s,
                    x1_s);
                tangent_obtained = normalize_rank1_tangent(
                    vec_ops,
                    x1_s,
                    lambda1_s);
            }
            
        }
        if(converged && tangent_obtained)
        {
            chart::accept_continuation_step(
                vec_ops,
                log,
                nonlin_op,
                x1,
                lambda1,
                x1_s,
                lambda1_s);
            if(verbose)
            {
                chart::log_continuation_chart(
                    log,
                    nonlin_op,
                    "continuation::advance_solution::accepted_chart_transition");
            }
        }
        const bool recovered = attempt.recovered();
        attempt.attempted_step = predictor->get_ds();
        attempt.retry_count = predictor_retry_count(predictor, 0);
        last_attempt_state_ = attempt;
        predictor->accept_step(recovered);
        if(verbose)
        {
            if(attempt.any_chart_retries)
            {
                log->info_f(
                    "continuation::advance_solution: accepted predictor after %u chart retries at dS = %le.",
                    attempt.chart_retries,
                    (double)predictor->get_ds());
            }
            if(recovered)
            {
                log->info_f(
                    "continuation::advance_solution: preserving recovered dS = %le for the next step.",
                    (double)predictor->get_ds());
            }
        }
        return tangent_obtained;
    }


private:
    template<class Predictor>
    static auto predictor_retry_count(Predictor* predictor_, int)
        -> decltype(predictor_->get_retry_count())
    {
        return predictor_->get_retry_count();
    }

    static unsigned int predictor_retry_count(...)
    {
        return 0;
    }

    template<class Predictor>
    static auto set_predictor_verbose(Predictor* predictor_, const bool value, int)
        -> decltype(predictor_->set_verbose(value), void())
    {
        predictor_->set_verbose(value);
    }

    static void set_predictor_verbose(...)
    {
    }

    VectorOperations* vec_ops;
    SystemOperator* sys_op;
    NewtonMethodExtended* newton_extended;
    NewtonMethod* newton;
    Predictoror* predictor;
    ConvergenceNewtonExtended* convergence_newton_extended;
    T_vec x_p, x1_l, dx10, x_isotropy_event;
    Loggin* log;
    char continuation_type;
    predictor_chart_policy<T> chart_policy;
    symmetry::continuation::isotropy_transition_policy<T> isotropy_policy;
    symmetry::continuation::isotropy_transition_result<T> last_isotropy_transition_;
    continuation_step_attempt_state<T> last_attempt_state_;
    bool verbose = true;

};




}

#endif
