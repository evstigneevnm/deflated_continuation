#ifndef __CONTINUATION_PROJECTED_SYSTEM_OPERATOR_CONTINUATION_H__
#define __CONTINUATION_PROJECTED_SYSTEM_OPERATOR_CONTINUATION_H__

#include <stdexcept>
#include <string>

#include <nonlinear_operators/projected_operator_helpers.h>

namespace continuation
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSystemSolver, class Log>
class projected_system_operator_continuation
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    projected_system_operator_continuation(
        VectorOperations* vec_ops_,
        Log* log_,
        LinearOperator* lin_op_,
        LinearSystemSolver* SM_solver_):
        vec_ops(vec_ops_),
        log(log_),
        lin_op(lin_op_),
        SM_solver(SM_solver_)
    {
        vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
        vec_ops->init_vector(f); vec_ops->start_use_vector(f);
        vec_ops->init_vector(Jlambda); vec_ops->start_use_vector(Jlambda);
    }

    ~projected_system_operator_continuation()
    {
        vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
        vec_ops->stop_use_vector(f); vec_ops->free_vector(f);
        vec_ops->stop_use_vector(Jlambda); vec_ops->free_vector(Jlambda);
    }

    void set_tangent_space(T_vec& x_0_, T& lambda_0_, T_vec& x_0_s_, T& lambda_0_s_, T& ds_l_, char continuation_type_ = 'S')
    {
        x_0 = x_0_;
        lambda_0 = lambda_0_;
        x_0_s = x_0_s_;
        lambda_0_s = lambda_0_s_;
        tangent_set = true;

        if(continuation_type_ == 'S')
        {
            ds_l = ds_l_;
        }
        else if(continuation_type_ == 'O')
        {
            ds_l = T(0);
        }
        else
        {
            throw std::runtime_error("continuation::projected_system_operator_continuation: incorrect continuation_type parameter. Only 'S'pherical or 'O'rthogonal can be used");
        }

        const T tangent_norm = vec_ops->norm_rank1(x_0_s, lambda_0_s);
        log->info_f("continuation::projected_system_operator: tangent space set: dS = %le, tangent norm = %le", (double)ds_l, (double)tangent_norm);
    }

    T arclength_residual(const T_vec& x_1, const T& lambda_1)
    {
        if(!tangent_set)
        {
            throw std::runtime_error("continuation::projected_system_operator: tangent space is not set. Set it with the method set_tangent_space(...).");
        }
        return orthogonal_projection(x_1, lambda_1);
    }

    bool update_tangent_space(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& x_1_s, T& lambda_1_s)
    {
        if(!tangent_set)
        {
            throw std::runtime_error("continuation::projected_system_operator: tangent space is not set. Set it with the method set_tangent_space(...).");
        }

        log->info("continuation::projected_system_operator: update_tangent_space starts.");

        nonlinear_operators::detail::set_linearization_point(nonlin_op, x, lambda);
        nonlinear_operators::detail::jacobian_alpha(nonlin_op, Jlambda);

        vec_ops->assign_scalar(T(0), f);
        T beta = T(1);
        T alpha = lambda_0_s;
        vec_ops->assign(x_0_s, x_1_s);
        lambda_1_s = lambda_0_s;

        T tolerance_local = T(1.0e-5)*vec_ops->get_l2_size();
        SM_solver->get_linsolver_handle()->monitor().set_temp_tolerance(tolerance_local);
        SM_solver->get_linsolver_handle()->monitor().set_temp_max_iterations(1000);
        const bool flag_lin_solver = SM_solver->solve((*lin_op), x_0_s, Jlambda, alpha, f, beta, x_1_s, lambda_1_s);
        nonlinear_operators::detail::project_current_tangent(vec_ops, nonlin_op, x_1_s, x_1_s);

        T minimum_resid = SM_solver->get_linsolver_handle()->monitor().resid_norm_out();
        int iters_performed = SM_solver->get_linsolver_handle()->monitor().iters_performed();
        log->info_f("desired residual = %le, minimum attained residual = %le with %i iterations.", (double)tolerance_local, (double)minimum_resid, iters_performed);

        SM_solver->get_linsolver_handle()->monitor().restore_max_iterations();
        SM_solver->get_linsolver_handle()->monitor().restore_tolerance();

        T norm = vec_ops->norm_rank1(x_1_s, lambda_1_s);
        lambda_1_s /= norm;
        vec_ops->scale(T(1)/norm, x_1_s);

        log->info("continuation::projected_system_operator: update_tangent_space ends.");
        tangent_set = false;
        return flag_lin_solver;
    }

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x, T& d_lambda)
    {
        if(!tangent_set)
        {
            throw std::runtime_error("continuation::projected_system_operator: tangent space is not set. Set it with the method set_tangent_space(...).");
        }

        nonlinear_operators::detail::set_linearization_point(nonlin_op, x, lambda);
        nonlinear_operators::detail::jacobian_alpha(nonlin_op, Jlambda);
        nonlinear_operators::detail::residual_at_linearization(nonlin_op, x, lambda, f);
        vec_ops->add_mul_scalar(T(0), T(-1), f);

        T arclength_res = orthogonal_projection(x, lambda);
        log->info_f("continuation::projected_system_operator: arclength residual = %le", (double)arclength_res);
        T beta = -arclength_res;
        T alpha = lambda_0_s;

        const bool flag_lin_solver = SM_solver->solve((*lin_op), x_0_s, Jlambda, alpha, f, beta, d_x, d_lambda);
        nonlinear_operators::detail::project_current_tangent(vec_ops, nonlin_op, d_x, d_x);
        return flag_lin_solver;
    }

private:
    T orthogonal_projection(const T_vec& x_1, const T& lambda_1)
    {
        vec_ops->assign_mul(T(1), x_1, T(-1), x_0, dx);
        T x_proj = vec_ops->scalar_prod(dx, x_0_s);
        T lambda_proj = (lambda_1 - lambda_0)*lambda_0_s;
        return x_proj + lambda_proj - ds_l;
    }

private:
    VectorOperations* vec_ops;
    Log* log;
    LinearOperator* lin_op;
    LinearSystemSolver* SM_solver;

    bool tangent_set = false;
    T_vec x_0, x_0_s;
    T lambda_0, lambda_0_s;
    T_vec dx, f, Jlambda;
    T ds_l;
};

} // namespace continuation

#endif
