#ifndef __CONTINUATION__INITIAL_TANGENT_H__
#define __CONTINUATION__INITIAL_TANGENT_H__

/**
    Class to get the initial tangent space

*/

#include <string>
#include <stdexcept>
#include <cmath>
#include <type_traits>
#include <limits>

#include <common/scalar_math.h>
#include <continuation/chart_helpers.h>
#include <continuation/initial_tangent_chart_validator.h>
#include <continuation/initial_tangent_candidates.h>
#include <continuation/initial_tangent_secant_builder.h>
#include <nonlinear_operators/projected_operator_helpers.h>
#include <numerical_algos/detail/str_source_helper.h>

namespace continuation
{

template<class VectorOperations, class Loggin, class NewtonMethod, class NonlinearOperator, class LinearOperator, class LinearSystemSolver>
class initial_tangent
{

public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    using shifted_newton_pair = continuation::shifted_newton_pair<T>;
    using tangent_candidate_quality = continuation::tangent_candidate_quality<T>;
    using tangent_equation_quality = continuation::tangent_equation_quality<T>;


    initial_tangent(VectorOperations*& vec_ops_, Loggin* log_, NewtonMethod* newton_, LinearOperator*& lin_op_, LinearSystemSolver*& lin_solv_, bool = true):
    vec_ops(vec_ops_),
    log(log_),
    lin_op(lin_op_),
    lin_solv(lin_solv_),
    secant_builder(vec_ops_, log_, newton_),
    chart_validator(vec_ops_, log_)
    {
        vec_ops->init_vector(f); vec_ops->start_use_vector(f);
        vec_ops->init_vector(f1); vec_ops->start_use_vector(f1);
        vec_ops->init_vector(row_x); vec_ops->start_use_vector(row_x);
        vec_ops->init_vector(Jlambda); vec_ops->start_use_vector(Jlambda);
        vec_ops->init_vector(candidate_x_s); vec_ops->start_use_vector(candidate_x_s);
        vec_ops->init_vector(best_x_s); vec_ops->start_use_vector(best_x_s);
    }
    ~initial_tangent()
    {
        vec_ops->stop_use_vector(f); vec_ops->free_vector(f);
        vec_ops->stop_use_vector(f1); vec_ops->free_vector(f1);
        vec_ops->stop_use_vector(row_x); vec_ops->free_vector(row_x);
        vec_ops->stop_use_vector(Jlambda); vec_ops->free_vector(Jlambda);
        vec_ops->stop_use_vector(candidate_x_s); vec_ops->free_vector(candidate_x_s);
        vec_ops->stop_use_vector(best_x_s); vec_ops->free_vector(best_x_s);
    }

    void set_predictor_chart_policy(const predictor_chart_policy<T>& policy)
    {
        chart_validator.set_policy(policy);
    }

    void set_projected_tangent_quality_policy(
        const projected_tangent_quality_policy<T>& policy)
    {
        policy.validate();
        tangent_quality_policy = policy;
    }

    void set_tangent_equation_quality_policy(
        const tangent_equation_quality_policy<T>& policy)
    {
        policy.validate();
        tangent_equation_policy = policy;
    }

    bool validate_tangent_candidate(
        NonlinearOperator* nonlin_op,
        const T_vec& x,
        const T& lambda,
        const T_vec& x_s,
        const T& lambda_s,
        const T& predictor_ds,
        const char* method)
    {
        const auto quality = log_tangent_diagnostics(
            nonlin_op,
            x,
            lambda,
            method,
            x_s,
            lambda_s);
        if(!tangent_equation_quality_is_acceptable(
               quality,
               tangent_equation_policy))
        {
            return false;
        }
        return chart_validator.accepts(
            nonlin_op,
            x,
            lambda,
            x_s,
            lambda_s,
            predictor_ds,
            method,
            nullptr);
    }

    bool execute(
        NonlinearOperator*& nonlin_op,
        const T sign,
        const T_vec& x,
        const T& lambda,
        T_vec& x_s,
        T& lambda_s,
        const T& predictor_ds = T(0.1))
    {
        log->info("continuation::initial_tangent: execute starts.");

        bool linear_system_converged = false;

        if constexpr(nonlinear_operators::detail::has_projected_tangent_system<NonlinearOperator, T_vec, T>::value)
        {
            if(execute_projected_bordered_tangent(
                   nonlin_op,
                   sign,
                   x,
                   lambda,
                   x_s,
                   lambda_s,
                   predictor_ds))
            {
                log->info("continuation::initial_tangent: execute ends successfully with projected bordered tangent.");
                return true;
            }
            log->warning("continuation::initial_tangent: projected bordered tangent was not accepted; falling back to the legacy tangent path.");
        }
    
        nonlinear_operators::detail::set_linearization_point(nonlin_op, x, lambda);
        if constexpr(NonlinearOperator::is_periodic_orbit_reprojected::value)
        {
            nonlin_op->F_and_jacobian_alpha(x_s, f); //here x_s is a mute variable! It will be zeroed later.
        }
        else
        {
            nonlinear_operators::detail::jacobian_alpha(nonlin_op, f);
        }
        
        //This is important!!!
        vec_ops->assign_scalar(T(0.0), x_s);
        //

        vec_ops->add_mul_scalar(T(0.0), T(-1.0), f);

        T tolerance_local = T(1.0e-8)*vec_ops->get_l2_size();

        lin_solv->get_linsolver_handle_original()->monitor().set_temp_tolerance(tolerance_local);
        lin_solv->get_linsolver_handle_original()->monitor().set_temp_max_iterations(10000);
        linear_system_converged = lin_solv->solve((*lin_op), f, x_s);
        nonlinear_operators::detail::project_current_tangent(vec_ops, nonlin_op, x_s, x_s);
        // if constexpr(NonlinearOperator::is_periodic_orbit_reprojected::value)
        // {
        //     nonlin_op->reproject(x_s);
        // }  

        T minimum_resid = lin_solv->get_linsolver_handle_original()->monitor().resid_norm_out();
        int iters_performed = lin_solv->get_linsolver_handle_original()->monitor().iters_performed();
        log->info_f("desired residual = %le, minimum attained residual = %le with %i iterations.", (double)tolerance_local, (double)minimum_resid, iters_performed);
        lin_solv->get_linsolver_handle_original()->monitor().restore_max_iterations();
        lin_solv->get_linsolver_handle_original()->monitor().restore_tolerance();

        if(linear_system_converged)
        {
            T z_sq = vec_ops->scalar_prod(x_s, x_s); //(dx,x_0_s)
            lambda_s = sign/common::scalar_math::sqrt(z_sq+T(1.0));
            vec_ops->add_mul_scalar(T(0.0), lambda_s, x_s); 
	    
            T norm = vec_ops->norm_rank1(x_s, lambda_s);
            lambda_s/=norm;
            vec_ops->scale(T(1)/norm, x_s);
            //vec_ops->scale(T(1)/T(vec_ops->get_l2_size()), x_s);
            const auto tangent_quality = log_tangent_diagnostics(
                nonlin_op,
                x,
                lambda,
                "linear solve",
                x_s,
                lambda_s);
            bool accept_linear_tangent =
                tangent_equation_quality_is_acceptable(
                    tangent_quality,
                    tangent_equation_policy);
            if constexpr(nonlinear_operators::detail::has_projected_tangent_system<NonlinearOperator, T_vec, T>::value)
            {
                accept_linear_tangent =
                    accept_linear_tangent &&
                    tangent_quality.absolute_residual <=
                        projected_direct_tangent_residual_tol();
                if(!accept_linear_tangent)
                {
                    log->warning_f(
                        "continuation::initial_tangent: rejected projected direct tangent candidate because tangent residual = %le exceeds tolerance = %le or its relative residual is invalid.",
                        (double)tangent_quality.absolute_residual,
                        (double)projected_direct_tangent_residual_tol());
                }
            }
            if(!accept_linear_tangent)
            {
                log->warning_f(
                    "continuation::initial_tangent: rejected direct tangent candidate: absolute residual = %le, relative residual = %le, maximum relative residual = %le.",
                    (double)tangent_quality.absolute_residual,
                    (double)tangent_quality.relative_residual,
                    (double)tangent_equation_policy.maximum_relative_residual);
                linear_system_converged = false;
            }
            if(accept_linear_tangent)
            {
                accept_linear_tangent = chart_validator.accepts(
                    nonlin_op,
                    x,
                    lambda,
                    x_s,
                    lambda_s,
                    predictor_ds,
                    "linear solve",
                    nullptr);
                if(accept_linear_tangent)
                {
                    log->info("continuation::initial_tangent: execute ends successfully.");
                }
                else
                {
                    linear_system_converged = false;
                }
            }
        }

        if(!linear_system_converged)
        {
            log->warning("continuation::initial_tangent: execute failed to converge. Attempting to use an approximate tangent from Newton-Raphson secants.");
            const bool converged = estimate_tangent_with_secant_fallback(
                nonlin_op,
                sign,
                x,
                lambda,
                x_s,
                lambda_s,
                predictor_ds);
            if(!converged)
            {
                throw std::runtime_error(std::string("continuation::initial_tangent " __FILE__ " " __STR(__LINE__) " tangent space couldn't be obtained - Newton method failed to converge.") );
            }
            linear_system_converged = true;

        }
        return linear_system_converged;

    }


private:
    bool execute_projected_bordered_tangent(
        NonlinearOperator* nonlin_op,
        const T sign,
        const T_vec& x,
        const T& lambda,
        T_vec& x_s,
        T& lambda_s,
        const T& predictor_ds)
    {
        shifted_newton_pair shifted = secant_builder.solve_pair(nonlin_op, sign, x, lambda);

        nonlinear_operators::detail::set_linearization_point(nonlin_op, x, lambda);
        nonlinear_operators::detail::jacobian_alpha(nonlin_op, Jlambda);

        tangent_candidate_selector<T> selector;

        if(shifted.plus_converged && shifted.minus_converged)
        {
            try_projected_bordered_tangent_candidate(
                nonlin_op,
                sign,
                x,
                lambda,
                shifted,
                secant_candidate_kind::two_sided,
                selector,
                predictor_ds);
        }
        if(shifted.plus_converged)
        {
            try_projected_bordered_tangent_candidate(
                nonlin_op,
                sign,
                x,
                lambda,
                shifted,
                secant_candidate_kind::plus_one_sided,
                selector,
                predictor_ds);
        }
        if(shifted.minus_converged)
        {
            try_projected_bordered_tangent_candidate(
                nonlin_op,
                sign,
                x,
                lambda,
                shifted,
                secant_candidate_kind::minus_one_sided,
                selector,
                predictor_ds);
        }

        if(!selector.has_candidate())
        {
            log->warning("continuation::initial_tangent: no projected bordered tangent candidate was accepted.");
            return false;
        }

        const tangent_candidate_quality& best_quality = selector.best();
        vec_ops->assign(best_x_s, x_s);
        lambda_s = best_quality.lambda_s;
        if(!chart_validator.accepts(
               nonlin_op,
               x,
               lambda,
               x_s,
               lambda_s,
               predictor_ds,
               best_quality.method,
               nullptr))
        {
            log->warning("continuation::initial_tangent: selected projected bordered tangent failed its final chart validation.");
            return false;
        }

        log->info_f(
            "continuation::initial_tangent: selected projected bordered tangent candidate: method = %s, score = %le, tangent residual = %le, row residual = %le, pre-normalization norm = %le, orientation = %le, lambda_s = %le.",
            best_quality.method,
            (double)best_quality.score,
            (double)best_quality.tangent_residual,
            (double)best_quality.row_residual_abs,
            (double)best_quality.pre_norm,
            (double)best_quality.orientation,
            (double)lambda_s);
        continuation::chart::log_continuation_chart(log, nonlin_op, "continuation::initial_tangent::projected_bordered_tangent");
        log_tangent_diagnostics(nonlin_op, x, lambda, best_quality.method, x_s, lambda_s);
        return true;
    }

    void try_projected_bordered_tangent_candidate(
        NonlinearOperator* nonlin_op,
        const T sign,
        const T_vec& x,
        const T& lambda,
        const shifted_newton_pair& shifted,
        const secant_candidate_kind kind,
        tangent_candidate_selector<T>& selector,
        const T& predictor_ds)
    {
        const char* method = nullptr;
        T row_lambda = T(0);
        if(!secant_builder.build_row(sign, x, shifted, kind, row_x, row_lambda, method))
        {
            return;
        }

        T beta = T(1);
        nonlinear_operators::detail::project_current_tangent(vec_ops, nonlin_op, row_x, row_x);
        if(!secant_builder.normalize(method, row_x, row_lambda))
        {
            log->warning_f(
                "continuation::initial_tangent: rejected projected bordered tangent candidate %s because its orientation row is degenerate after projection.",
                method);
            return;
        }

        if(row_lambda < T(0))
        {
            vec_ops->scale(T(-1), row_x);
            row_lambda = -row_lambda;
            beta = T(-1);
        }

        const T row_lambda_abs = common::scalar_math::abs(row_lambda);
        if(!(row_lambda_abs > T(0)))
        {
            log->warning_f(
                "continuation::initial_tangent: rejected projected bordered tangent candidate %s because the bordered row has zero lambda component.",
                method);
            return;
        }

        tangent_candidate_quality quality = solve_projected_bordered_tangent_candidate(
            nonlin_op,
            method,
            row_lambda,
            beta,
            candidate_x_s);

        if(!quality.valid)
        {
            log->warning_f(
                "continuation::initial_tangent: rejected projected bordered tangent candidate: method = %s, solved = %i, tangent residual = %le (tol %le), row residual = %le (tol %le), pre-normalization norm = %le, orientation = %le, lambda_s = %le.",
                quality.method,
                quality.solved ? 1 : 0,
                (double)quality.tangent_residual,
                (double)quality.residual_tol,
                (double)quality.row_residual_abs,
                (double)quality.row_residual_tol,
                (double)quality.pre_norm,
                (double)quality.orientation,
                (double)quality.lambda_s);
            return;
        }

        if(!chart_validator.accepts(
               nonlin_op,
               x,
               lambda,
               candidate_x_s,
               quality.lambda_s,
               predictor_ds,
               quality.method,
               &quality))
        {
            return;
        }

        log->info_f(
            "continuation::initial_tangent: accepted projected bordered tangent candidate: method = %s, score = %le, tangent residual = %le, row residual = %le, pre-normalization norm = %le, orientation = %le, lambda_s = %le.",
            quality.method,
            (double)quality.score,
            (double)quality.tangent_residual,
            (double)quality.row_residual_abs,
            (double)quality.pre_norm,
            (double)quality.orientation,
            (double)quality.lambda_s);

        if(selector.consider(quality))
        {
            vec_ops->assign(candidate_x_s, best_x_s);
        }
    }

    tangent_candidate_quality solve_projected_bordered_tangent_candidate(
        NonlinearOperator* nonlin_op,
        const char* method,
        const T& row_lambda,
        const T& beta,
        T_vec& candidate)
    {
        tangent_candidate_quality quality;
        quality.method = method;
        quality.row_residual_tol = T(1.0e-5)*(T(1) + common::scalar_math::abs(beta));
        const T jlambda_norm = vec_ops->norm_l2(Jlambda);
        quality.residual_tol = T(1.0e-4)*(T(1) + jlambda_norm);
        if(quality.residual_tol < tangent_quality_policy.residual_tolerance_floor)
        {
            quality.residual_tol = tangent_quality_policy.residual_tolerance_floor;
        }
        if(quality.residual_tol > tangent_quality_policy.residual_tolerance_ceiling)
        {
            quality.residual_tol = tangent_quality_policy.residual_tolerance_ceiling;
        }

        vec_ops->assign_scalar(T(0), f);
        vec_ops->assign_scalar(T(0), candidate);
        quality.lambda_s = T(0);

        const T tolerance_local = T(1.0e-8)*vec_ops->get_l2_size();
        lin_solv->get_linsolver_handle()->monitor().set_temp_tolerance(tolerance_local);
        lin_solv->get_linsolver_handle()->monitor().set_temp_max_iterations(10000);
        quality.solved = lin_solv->solve((*lin_op), row_x, Jlambda, row_lambda, f, beta, candidate, quality.lambda_s);
        nonlinear_operators::detail::project_current_tangent(vec_ops, nonlin_op, candidate, candidate);

        const T minimum_resid = lin_solv->get_linsolver_handle()->monitor().resid_norm_out();
        const int iters_performed = lin_solv->get_linsolver_handle()->monitor().iters_performed();
        log->info_f(
            "continuation::initial_tangent: projected bordered tangent solve candidate: row = %s, desired residual = %le, minimum attained residual = %le with %i iterations.",
            method,
            (double)tolerance_local,
            (double)minimum_resid,
            iters_performed);
        lin_solv->get_linsolver_handle()->monitor().restore_max_iterations();
        lin_solv->get_linsolver_handle()->monitor().restore_tolerance();

        if(!quality.solved)
        {
            quality.score = std::numeric_limits<T>::max();
            return quality;
        }

        lin_op->apply(candidate, f1);
        vec_ops->add_mul(quality.lambda_s, Jlambda, f1);
        quality.tangent_residual = vec_ops->norm_l2(f1);
        const T row_residual = vec_ops->scalar_prod(row_x, candidate) + row_lambda*quality.lambda_s - beta;
        quality.row_residual_abs = common::scalar_math::abs(row_residual);
        quality.pre_norm = vec_ops->norm_rank1(candidate, quality.lambda_s);

        if(quality.pre_norm > T(0))
        {
            quality.lambda_s /= quality.pre_norm;
            vec_ops->scale(T(1)/quality.pre_norm, candidate);
            quality.orientation = vec_ops->scalar_prod(row_x, candidate) + row_lambda*quality.lambda_s;
        }

        quality.valid = continuation::projected_tangent_quality_is_acceptable(
            quality,
            tangent_quality_policy);
        quality.score = continuation::projected_tangent_candidate_score(
            quality,
            tangent_quality_policy);
        return quality;
    }

    T projected_direct_tangent_residual_tol() const
    {
        return tangent_quality_policy.direct_tangent_residual_tolerance;
    }

    bool estimate_tangent_with_secant_fallback(
        NonlinearOperator* nonlin_op,
        const T sign,
        const T_vec& x,
        const T& lambda,
        T_vec& x_s,
        T& lambda_s,
        const T& predictor_ds)
    {
        const char* method = nullptr;
        if(build_secant_tangent(
               nonlin_op,
               sign,
               x,
               lambda,
               x_s,
               lambda_s,
               method,
               predictor_ds))
        {
            log->info("continuation::initial_tangent: Newton-Raphson secant estimate ends successfully.");
            return true;
        }
        return false;
    }

    bool build_secant_tangent(
        NonlinearOperator* nonlin_op,
        const T sign,
        const T_vec& x,
        const T& lambda,
        T_vec& x_s,
        T& lambda_s,
        const char*& method,
        const T& predictor_ds)
    {
        const shifted_newton_pair shifted = secant_builder.solve_pair(nonlin_op, sign, x, lambda);
        if(shifted.plus_converged && shifted.minus_converged)
        {
            secant_builder.build_row(sign, x, shifted, secant_candidate_kind::two_sided, x_s, lambda_s, method);
            if(secant_builder.normalize(method, x_s, lambda_s) &&
               tangent_candidate_is_acceptable(
                   nonlin_op,
                   x,
                   lambda,
                   method,
                   x_s,
                   lambda_s) &&
               chart_validator.accepts(
                   nonlin_op,
                   x,
                   lambda,
                   x_s,
                   lambda_s,
                   predictor_ds,
                   method,
                   nullptr))
            {
                return true;
            }
            log->warning("continuation::initial_tangent: two-sided Newton-Raphson estimate produced a degenerate tangent.");
        }

        if(shifted.plus_converged)
        {
            secant_builder.build_row(sign, x, shifted, secant_candidate_kind::plus_one_sided, x_s, lambda_s, method);
            if(secant_builder.normalize(method, x_s, lambda_s) &&
               tangent_candidate_is_acceptable(
                   nonlin_op,
                   x,
                   lambda,
                   method,
                   x_s,
                   lambda_s) &&
               chart_validator.accepts(
                   nonlin_op,
                   x,
                   lambda,
                   x_s,
                   lambda_s,
                   predictor_ds,
                   method,
                   nullptr))
            {
                return true;
            }
        }

        if(shifted.minus_converged)
        {
            secant_builder.build_row(sign, x, shifted, secant_candidate_kind::minus_one_sided, x_s, lambda_s, method);
            if(secant_builder.normalize(method, x_s, lambda_s) &&
               tangent_candidate_is_acceptable(
                   nonlin_op,
                   x,
                   lambda,
                   method,
                   x_s,
                   lambda_s) &&
               chart_validator.accepts(
                   nonlin_op,
                   x,
                   lambda,
                   x_s,
                   lambda_s,
                   predictor_ds,
                   method,
                   nullptr))
            {
                return true;
            }
        }

        return false;
    }

    bool tangent_candidate_is_acceptable(
        NonlinearOperator* nonlin_op,
        const T_vec& x,
        const T& lambda,
        const char* method,
        const T_vec& x_s,
        const T& lambda_s)
    {
        const auto quality = log_tangent_diagnostics(
            nonlin_op,
            x,
            lambda,
            method,
            x_s,
            lambda_s);
        return tangent_equation_quality_is_acceptable(
            quality,
            tangent_equation_policy);
    }

    tangent_equation_quality log_tangent_diagnostics(
        NonlinearOperator* nonlin_op,
        const T_vec& x,
        const T& lambda,
        const char* method,
        const T_vec& x_s,
        const T& lambda_s)
    {
        nonlinear_operators::detail::set_linearization_point(nonlin_op, x, lambda);
        const T x_s_norm = vec_ops->norm_l2(x_s);
        const T rank1_norm = vec_ops->norm_rank1(x_s, lambda_s);

        nonlinear_operators::detail::jacobian_alpha(nonlin_op, f);
        vec_ops->add_mul_scalar(T(0), T(-1), f);

        // f stores -J_lambda, so J*x_s - lambda_s*f is
        // J*x_s + J_lambda*lambda_s, the tangent equation residual.
        const T parameter_term_norm =
            common::scalar_math::abs(lambda_s)*vec_ops->norm_l2(f);
        lin_op->apply(x_s, f1);
        const T jacobian_term_norm = vec_ops->norm_l2(f1);
        vec_ops->add_mul(-lambda_s, f, f1);
        const T tangent_residual = vec_ops->norm_l2(f1);
        const auto quality = make_tangent_equation_quality(
            tangent_residual,
            jacobian_term_norm,
            parameter_term_norm);

        log->info_f(
            "continuation::initial_tangent: diagnostics: method = %s, ||x_s|| = %le, lambda_s = %le, ||(x_s,lambda_s)|| = %le, tangent equation residual = %le, equation scale = %le, relative residual = %le",
            method,
            (double)x_s_norm,
            (double)lambda_s,
            (double)rank1_norm,
            (double)quality.absolute_residual,
            (double)quality.equation_scale,
            (double)quality.relative_residual);
        if(!tangent_equation_quality_is_acceptable(
               quality,
               tangent_equation_policy))
        {
            log->warning_f(
                "continuation::initial_tangent: rejected tangent candidate by equation validation: method = %s, absolute residual = %le, relative residual = %le, maximum relative residual = %le, lambda_s = %le.",
                method,
                (double)quality.absolute_residual,
                (double)quality.relative_residual,
                (double)tangent_equation_policy.maximum_relative_residual,
                (double)lambda_s);
        }
        return quality;
    }

    VectorOperations* vec_ops;
    Loggin* log;
    LinearOperator* lin_op;
    LinearSystemSolver* lin_solv;
    T_vec f, f1;
    T_vec row_x, Jlambda;
    T_vec candidate_x_s, best_x_s;
    initial_tangent_secant_builder<
        VectorOperations,
        Loggin,
        NewtonMethod,
        NonlinearOperator> secant_builder;
    initial_tangent_chart_validator<
        VectorOperations,
        Loggin,
        NonlinearOperator> chart_validator;
    projected_tangent_quality_policy<T> tangent_quality_policy;
    tangent_equation_quality_policy<T> tangent_equation_policy;
    
};

}


#endif
