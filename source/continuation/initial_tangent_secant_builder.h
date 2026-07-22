#ifndef __CONTINUATION_INITIAL_TANGENT_SECANT_BUILDER_H__
#define __CONTINUATION_INITIAL_TANGENT_SECANT_BUILDER_H__

#include <exception>

#include <continuation/initial_tangent_candidates.h>
#include <nonlinear_operators/projected_operator_helpers.h>

namespace continuation
{

template<class VectorOperations, class Log, class NewtonMethod, class NonlinearOperator>
class initial_tangent_secant_builder
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using shifted_pair_type = shifted_newton_pair<scalar_type>;

    initial_tangent_secant_builder(
        VectorOperations* vec_ops_,
        Log* log_,
        NewtonMethod* newton_):
        vec_ops(vec_ops_),
        log(log_),
        newton(newton_)
    {
        vec_ops->init_vector(plus_state);
        vec_ops->start_use_vector(plus_state);
        vec_ops->init_vector(minus_state);
        vec_ops->start_use_vector(minus_state);
    }

    ~initial_tangent_secant_builder()
    {
        vec_ops->stop_use_vector(minus_state);
        vec_ops->free_vector(minus_state);
        vec_ops->stop_use_vector(plus_state);
        vec_ops->free_vector(plus_state);
    }

    initial_tangent_secant_builder(const initial_tangent_secant_builder&) = delete;
    initial_tangent_secant_builder& operator=(const initial_tangent_secant_builder&) = delete;

    shifted_pair_type solve_pair(
        NonlinearOperator* nonlin_op,
        const scalar_type sign,
        const vector_type& x,
        const scalar_type& lambda)
    {
        shifted_pair_type shifted;
        const scalar_type x_norm = vec_ops->norm(x);
        shifted.d_lambda = x_norm > scalar_type(0)
            ? scalar_type(1)/x_norm
            : scalar_type(1);
        shifted.lambda_plus = lambda + shifted.d_lambda;
        shifted.lambda_minus = lambda - shifted.d_lambda;
        shifted.plus_converged = solve_shifted_newton(
            nonlin_op,
            x,
            shifted.lambda_plus,
            plus_state);
        shifted.minus_converged = solve_shifted_newton(
            nonlin_op,
            x,
            shifted.lambda_minus,
            minus_state);

        log->info_f(
            "continuation::initial_tangent: shifted Newton solves for tangent candidates: d_lambda = %le, plus_converged = %i, minus_converged = %i, sign = %le.",
            double(shifted.d_lambda),
            shifted.plus_converged ? 1 : 0,
            shifted.minus_converged ? 1 : 0,
            double(sign));
        return shifted;
    }

    bool build_row(
        const scalar_type sign,
        const vector_type& x,
        const shifted_pair_type& shifted,
        const secant_candidate_kind kind,
        vector_type& x_s,
        scalar_type& lambda_s,
        const char*& method)
    {
        switch(kind)
        {
        case secant_candidate_kind::two_sided:
            if(!(shifted.plus_converged && shifted.minus_converged))
            {
                return false;
            }
            vec_ops->assign_mul(sign, plus_state, -sign, minus_state, x_s);
            lambda_s = sign*(shifted.lambda_plus - shifted.lambda_minus);
            method = "two-sided Newton-Raphson secant";
            return true;
        case secant_candidate_kind::plus_one_sided:
            if(!shifted.plus_converged)
            {
                return false;
            }
            vec_ops->assign_mul(sign, plus_state, -sign, x, x_s);
            lambda_s = sign*shifted.d_lambda;
            method = "one-sided Newton-Raphson secant from plus side";
            return true;
        case secant_candidate_kind::minus_one_sided:
            if(!shifted.minus_converged)
            {
                return false;
            }
            vec_ops->assign_mul(sign, x, -sign, minus_state, x_s);
            lambda_s = sign*shifted.d_lambda;
            method = "one-sided Newton-Raphson secant from minus side";
            return true;
        }
        return false;
    }

    bool normalize(const char* method, vector_type& x_s, scalar_type& lambda_s)
    {
        const scalar_type ds = vec_ops->norm_rank1(x_s, lambda_s);
        if(!(ds > scalar_type(0)))
        {
            return false;
        }
        lambda_s /= ds;
        vec_ops->scale(scalar_type(1)/ds, x_s);
        log->info_f(
            "continuation::initial_tangent: estimated local ds = %le using %s",
            double(ds),
            method);
        return true;
    }

private:
    bool solve_shifted_newton(
        NonlinearOperator* nonlin_op,
        const vector_type& x,
        const scalar_type& lambda_shifted,
        vector_type& x_shifted)
    {
        vec_ops->assign(x, x_shifted);
        bool converged = false;
        try
        {
            converged = newton->solve(nonlin_op, x_shifted, lambda_shifted);
        }
        catch(const std::exception& error)
        {
            log->warning_f(
                "continuation::initial_tangent: shifted Newton solve failed at lambda = %le while building a tangent secant: %s",
                double(lambda_shifted),
                error.what());
            return false;
        }
        if(converged)
        {
            nonlinear_operators::detail::project_state_relative_to(
                vec_ops,
                nonlin_op,
                x,
                x_shifted);
        }
        return converged;
    }

    VectorOperations* vec_ops;
    Log* log;
    NewtonMethod* newton;
    vector_type plus_state;
    vector_type minus_state;
};

} // namespace continuation

#endif // __CONTINUATION_INITIAL_TANGENT_SECANT_BUILDER_H__
