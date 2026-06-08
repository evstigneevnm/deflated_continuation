#ifndef __SYSTEM_OPERATOR_DEFLATION_H__
#define __SYSTEM_OPERATOR_DEFLATION_H__

#include <nonlinear_operators/projected_operator_helpers.h>

namespace deflation
{

namespace detail
{

template<class T>
void log_deflation_distance(void*, const char*, const T&)
{
}

template<class Log, class T>
auto log_deflation_distance(Log* log, const char* label, const T& beta) -> decltype(log->info_f("%s distance = %le", label, static_cast<double>(beta)), void())
{
    if(log)
    {
        log->info_f("%s distance = %le", label, static_cast<double>(beta));
    }
}

} // namespace detail

template<class vector_operations, class nonlinear_operator, class linear_operator, class sherman_morrison_linear_system_solver, class solution_storage, class Log = void>
class system_operator_deflation
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;
    

    system_operator_deflation(vector_operations* vec_ops_, linear_operator* lin_op_, sherman_morrison_linear_system_solver* SM_solver_, solution_storage* sol_storage_, Log* log_ = nullptr):
    vec_ops(vec_ops_),
    lin_op(lin_op_),
    SM_solver(SM_solver_),
    sol_storage(sol_storage_),
    log(log_)
    {
        vec_ops->init_vector(f); vec_ops->start_use_vector(f);
        vec_ops->init_vector(b); vec_ops->start_use_vector(b);
        vec_ops->init_vector(c); vec_ops->start_use_vector(c);
    }
    ~system_operator_deflation()
    {
        vec_ops->stop_use_vector(f); vec_ops->free_vector(f);
        vec_ops->stop_use_vector(b); vec_ops->free_vector(b);
        vec_ops->stop_use_vector(c); vec_ops->free_vector(c);
    }

    bool solve(nonlinear_operator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x, T& d_lambda)
    {
        bool flag_lin_solver;
        
        nonlinear_operators::detail::set_linearization_point(nonlin_op, x, lambda);
        nonlinear_operators::detail::residual_at_linearization(nonlin_op, x, lambda, f); // f = F(x)
        vec_ops->assign(f, b);
        sol_storage->calc_distance(x, beta, c); //beta = 1/||x-x0_j||, c = (x-x0_j)
        vec_ops->add_mul_scalar(T(0), T(-beta), b); //b=-F(x,lambda)
        detail::log_deflation_distance(log, "deflation::system_operator", beta);
        flag_lin_solver = SM_solver->solve(beta, *lin_op, T(1.0), c, f, b, d_x);
        d_lambda = 0;
        return flag_lin_solver;
    }
private:
    vector_operations* vec_ops;
    linear_operator* lin_op;
    sherman_morrison_linear_system_solver* SM_solver;
    solution_storage* sol_storage;
    Log* log;
    T_vec b;
    T_vec f;
    T_vec c;
    T beta;

};

}

#endif
