#ifndef __CONTINUATION__INITIAL_TANGENT_H__
#define __CONTINUATION__INITIAL_TANGENT_H__

/**
    Class to get the initial tangent space

*/

#include <string>
#include <stdexcept>
#include <cmath>

#include <common/scalar_math.h>
#include <nonlinear_operators/projected_operator_helpers.h>
#include <iostream>

namespace continuation
{

template<class VectorOperations, class Loggin, class NewtonMethod, class NonlinearOperator, class LinearOperator, class LinearSystemSolver>
class initial_tangent
{

public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;


    initial_tangent(VectorOperations*& vec_ops_,  Loggin* log_, NewtonMethod* newton_, LinearOperator*& lin_op_, LinearSystemSolver*& lin_solv_, bool verbose_=true):
    vec_ops(vec_ops_),
    log(log_),
    newton(newton_),
    lin_op(lin_op_),
    lin_solv(lin_solv_),
    verbose(verbose_)
    {
        vec_ops->init_vector(f); vec_ops->start_use_vector(f);
        vec_ops->init_vector(f1); vec_ops->start_use_vector(f1);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(x2); vec_ops->start_use_vector(x2);
    }
    ~initial_tangent()
    {
        vec_ops->stop_use_vector(f); vec_ops->free_vector(f);
        vec_ops->stop_use_vector(f1); vec_ops->free_vector(f1);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(x2); vec_ops->free_vector(x2);
    }

    bool execute(NonlinearOperator*& nonlin_op, const T sign, const T_vec& x, const T& lambda, T_vec& x_s, T& lambda_s)
    {
        log->info("continuation::initial_tangent: execute starts.");

        bool linear_system_converged = false;
    
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
	    
        //TODO: do smth with the norm
	    //norm differs greatly for large
	    //and small systems

            T norm = vec_ops->norm_rank1(x_s, lambda_s);
            lambda_s/=norm;
            vec_ops->scale(T(1)/norm, x_s);
            //vec_ops->scale(T(1)/T(vec_ops->get_l2_size()), x_s);
            log_tangent_diagnostics(nonlin_op, x, lambda, "linear solve", x_s, lambda_s);
            log->info("continuation::initial_tangent: execute ends successfully.");

        }
        else
        {
            log->warning("continuation::initial_tangent: execute falied to converge. Attempting to use approximate tangent solution via the Newton-Raphson method.");
            const bool converged = estimate_tangent_with_secant_fallback(nonlin_op, sign, x, lambda, x_s, lambda_s);
            if(!converged)
            {
                throw std::runtime_error(std::string("continuation::initial_tangent " __FILE__ " " __STR(__LINE__) " tangent space couldn't be obtained - Newton method failed to converge.") );
            }
            linear_system_converged = true;

        }
        return linear_system_converged;

    }


private:
    bool estimate_tangent_with_secant_fallback(
        NonlinearOperator* nonlin_op,
        const T sign,
        const T_vec& x,
        const T& lambda,
        T_vec& x_s,
        T& lambda_s)
    {
        const T x_norm = vec_ops->norm(x);
        const T d_lambda = x_norm > T(0) ? T(1.0)/x_norm : T(1.0);
        const T lambda_plus = lambda + d_lambda;
        const T lambda_minus = lambda - d_lambda;

        const bool plus_converged = solve_shifted_newton(nonlin_op, x, lambda_plus, x1);
        const bool minus_converged = solve_shifted_newton(nonlin_op, x, lambda_minus, x2);

        log->info_f(
            "continuation::initial_tangent: two-sided secant fallback shifted solves: d_lambda = %le, plus_converged = %i, minus_converged = %i.",
            (double)d_lambda,
            plus_converged ? 1 : 0,
            minus_converged ? 1 : 0);

        if(plus_converged && minus_converged)
        {
            vec_ops->assign_mul(sign, x1, -sign, x2, x_s);
            lambda_s = sign*(lambda_plus - lambda_minus);
            if(normalize_secant_tangent("two-sided Newton-Raphson secant fallback", x_s, lambda_s))
            {
                log_tangent_diagnostics(nonlin_op, x, lambda, "two-sided Newton-Raphson secant fallback", x_s, lambda_s);
                log->info("continuation::initial_tangent: two-sided Newton-Raphson estimate ends successfully.");
                return true;
            }
            log->warning("continuation::initial_tangent: two-sided Newton-Raphson estimate produced a degenerate tangent.");
        }

        if(plus_converged)
        {
            vec_ops->assign_mul(sign, x1, -sign, x, x_s);
            lambda_s = sign*d_lambda;
            if(normalize_secant_tangent("one-sided Newton-Raphson secant fallback from plus side", x_s, lambda_s))
            {
                log_tangent_diagnostics(nonlin_op, x, lambda, "one-sided Newton-Raphson secant fallback from plus side", x_s, lambda_s);
                log->info("continuation::initial_tangent: one-sided plus Newton-Raphson estimate ends successfully.");
                return true;
            }
        }

        if(minus_converged)
        {
            vec_ops->assign_mul(sign, x, -sign, x2, x_s);
            lambda_s = sign*d_lambda;
            if(normalize_secant_tangent("one-sided Newton-Raphson secant fallback from minus side", x_s, lambda_s))
            {
                log_tangent_diagnostics(nonlin_op, x, lambda, "one-sided Newton-Raphson secant fallback from minus side", x_s, lambda_s);
                log->info("continuation::initial_tangent: one-sided minus Newton-Raphson estimate ends successfully.");
                return true;
            }
        }

        return false;
    }

    bool solve_shifted_newton(NonlinearOperator* nonlin_op, const T_vec& x, const T& lambda_shifted, T_vec& x_shifted)
    {
        vec_ops->assign(x, x_shifted);
        const bool converged = newton->solve(nonlin_op, x_shifted, lambda_shifted);
        if(converged)
        {
            nonlinear_operators::detail::project_state_relative_to(vec_ops, nonlin_op, x, x_shifted);
        }
        return converged;
    }

    bool normalize_secant_tangent(const char* method, T_vec& x_s, T& lambda_s)
    {
        const T ds_l = vec_ops->norm_rank1(x_s, lambda_s);
        if(!(ds_l > T(0)))
        {
            return false;
        }
        lambda_s /= ds_l;
        vec_ops->scale(T(1.0)/ds_l, x_s);
        log->info_f("continuation::initial_tangent: estimated local ds = %le using %s", (double)ds_l, method);
        return true;
    }

    void log_tangent_diagnostics(
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

        // f stores -J_lambda at this point, so J*x_s - lambda_s*f is
        // J*x_s + J_lambda*lambda_s, the tangent equation residual.
        lin_op->apply(x_s, f1);
        vec_ops->add_mul(-lambda_s, f, f1);
        const T tangent_residual = vec_ops->norm_l2(f1);

        log->info_f(
            "continuation::initial_tangent: diagnostics: method = %s, ||x_s|| = %le, lambda_s = %le, ||(x_s,lambda_s)|| = %le, tangent equation residual = %le",
            method,
            (double)x_s_norm,
            (double)lambda_s,
            (double)rank1_norm,
            (double)tangent_residual);
        if(tangent_residual > T(1))
        {
            log->warning_f(
                "continuation::initial_tangent: validation warning: tangent equation residual is large: method = %s, residual = %le, lambda_s = %le.",
                method,
                (double)tangent_residual,
                (double)lambda_s);
        }
    }

    VectorOperations* vec_ops;
    Loggin* log;
    NewtonMethod* newton;
    LinearOperator* lin_op;
    LinearSystemSolver* lin_solv;
    bool verbose;
    T_vec f, f1;
    T_vec x1, x2;
    
};

}


#endif
