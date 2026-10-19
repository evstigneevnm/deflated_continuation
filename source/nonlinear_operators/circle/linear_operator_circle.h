#ifndef __LINEAR_OPERATOR_CIRCLE_H__
#define __LINEAR_OPERATOR_CIRCLE_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*
*/

#include <memory>
#include <stdexcept>
#include <utility>

namespace nonlinear_operators
{


template <class vector_operations, class nonlinear_operator>
class linear_operator_circle
{
public:
    typedef typename vector_operations::scalar_type T;
    typedef typename vector_operations::vector_type T_vec;

    explicit linear_operator_circle( std::shared_ptr<nonlinear_operator> problem )
        : problem_owner_( std::move( problem ) ), nonlin_op( problem_owner_.get() )
    {
        if ( nonlin_op == nullptr )
            throw std::invalid_argument( "circle linear operator requires a problem" );
    }

    linear_operator_circle( nonlinear_operator *&nonlin_op_ ) : nonlin_op( nonlin_op_ )
    {
    }
    ~linear_operator_circle()
    {
    }

    void apply( const T_vec &x, T_vec &f ) const
    {
        nonlin_op->jacobian_u( x, f );
    }

private:
    std::shared_ptr<nonlinear_operator> problem_owner_;
    nonlinear_operator                 *nonlin_op;
};

}

#endif
