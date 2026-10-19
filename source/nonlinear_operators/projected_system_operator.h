#ifndef __PROJECTED_SYSTEM_OPERATOR_H__
#define __PROJECTED_SYSTEM_OPERATOR_H__

#include <nonlinear_operators/projected_operator_helpers.h>
#include <numerical_algos/lin_solvers/linear_solve_recovery.h>
#include <nmfd/detail/vector_wrap.h>

namespace nonlinear_operators
{

template <class vector_operations, class nonlinear_operator, class linear_operator, class linear_solver>
class projected_system_operator
{
public:
    typedef typename vector_operations::scalar_type T;
    typedef typename vector_operations::vector_type T_vec;

    projected_system_operator( vector_operations *&vec_ops_, linear_operator *&lin_op_, linear_solver *&lin_solver_ )
        : vec_ops( vec_ops_ ), lin_op( lin_op_ ), lin_solver( lin_solver_ ), rhs_storage_( *vec_ops_ )
    {
        rhs_storage_.init();
        rhs_storage_.start_use();
    }

    bool solve( nonlinear_operator *nonlin_op, const T_vec &x, const T lambda, T_vec &d_x )
    {
        detail::set_linearization_point( nonlin_op, x, lambda );
        detail::residual_at_linearization( nonlin_op, x, lambda, b );
        vec_ops->add_mul_scalar( T( 0 ), T( -1 ), b );
        const bool flag_lin_solver = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            lin_solver, [this, &d_x]() { return lin_solver->solve( *lin_op, b, d_x ); },
            [this, &d_x]() { vec_ops->assign_scalar( T( 0 ), d_x ); }
        );
        detail::project_current_tangent( vec_ops, nonlin_op, d_x, d_x );
        return flag_lin_solver;
    }

private:
    vector_operations                                         *vec_ops;
    linear_operator                                           *lin_op;
    linear_solver                                             *lin_solver;
    nmfd::detail::vector_wrap<vector_operations, false, false> rhs_storage_;
    T_vec                                                     &b = *rhs_storage_;
};

} // namespace nonlinear_operators

#endif
