#ifndef __LINEAR_OPERATOR_BRATU_H__
#define __LINEAR_OPERATOR_BRATU_H__

#include <memory>
#include <utility>

namespace nonlinear_operators
{

template <class VectorOperations, class NonlinearOperator>
class linear_operator_bratu
{
public:
    using T     = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    explicit linear_operator_bratu( NonlinearOperator *nonlin_op_ ) : nonlin_op( nonlin_op_ )
    {
    }

    explicit linear_operator_bratu( std::shared_ptr<NonlinearOperator> problem )
        : owner_( std::move( problem ) ), nonlin_op( owner_.get() )
    {
    }

    void apply( const T_vec &x, T_vec &f ) const
    {
        nonlin_op->jacobian_u( x, f );
    }

    void copy_matrix( T *values ) const
    {
        nonlin_op->copy_jacobian_matrix( values );
    }

private:
    std::shared_ptr<NonlinearOperator> owner_;
    NonlinearOperator                 *nonlin_op;
};

} // namespace nonlinear_operators

#endif
