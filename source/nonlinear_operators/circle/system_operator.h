#ifndef __SYSTEM_OPERATOR_H__
#define __SYSTEM_OPERATOR_H__


/**
*   System operator class used to solve the linear system
*   Executed in the Newton's method
*
*
*/

#include <numerical_algos/lin_solvers/linear_solve_recovery.h>
#include <nmfd/detail/vector_wrap.h>


namespace nonlinear_operators
{


template <class vector_operations, class nonlinear_operator, class linear_operator, class linear_solver>
class system_operator
{
public:
    typedef typename vector_operations::scalar_type T;
    typedef typename vector_operations::vector_type T_vec;


    system_operator( vector_operations *&vec_ops_, linear_operator *&lin_op_, linear_solver *&lin_solver_ )
        : vec_ops( vec_ops_ ), lin_op( lin_op_ ), lin_solver( lin_solver_ ), rhs_storage_( *vec_ops_ )
    {
        rhs_storage_.init();
        rhs_storage_.start_use();
    }

    system_operator( const system_operator & )            = delete;
    system_operator &operator=( const system_operator & ) = delete;

    bool solve( nonlinear_operator *nonlin_op, const T_vec &x, const T lambda, T_vec &d_x )
    {
        bool flag_lin_solver;

        nonlin_op->set_linearization_point( x, lambda );
        nonlin_op->F( x, lambda, b );                  //
        vec_ops->add_mul_scalar( T( 0 ), T( -1 ), b ); //b=-F(x,lambda)
        flag_lin_solver = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            lin_solver, [this, &d_x]() { return lin_solver->solve( *lin_op, b, d_x ); },
            [this, &d_x]() { vec_ops->assign_scalar( T( 0 ), d_x ); }
        );
        return flag_lin_solver;
    }

private:
    vector_operations                                         *vec_ops;
    linear_operator                                           *lin_op;
    linear_solver                                             *lin_solver;
    nmfd::detail::vector_wrap<vector_operations, false, false> rhs_storage_;
    T_vec                                                     &b = *rhs_storage_;
};

}

#endif
