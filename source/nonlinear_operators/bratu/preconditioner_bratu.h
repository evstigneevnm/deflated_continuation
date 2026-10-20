#ifndef __PRECONDITIONER_BRATU_H__
#define __PRECONDITIONER_BRATU_H__

#include <memory>
#include <utility>

namespace nonlinear_operators
{

template <class VectorOperations, class NonlinearOperator, class LinearOperator>
class preconditioner_bratu
{
public:
    using T     = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    explicit preconditioner_bratu( NonlinearOperator *nonlin_op_ ) : nonlin_op( nonlin_op_ )
    {
    }

    explicit preconditioner_bratu( std::shared_ptr<NonlinearOperator> problem )
        : owner_( std::move( problem ) ), nonlin_op( owner_.get() )
    {
    }

    void set_operator( std::shared_ptr<const LinearOperator> op )
    {
        operator_owner_ = std::move( op );
        lin_op          = operator_owner_.get();
    }

    void set_operator( const LinearOperator *op_ ) const
    {
        lin_op = op_;
        (void)lin_op;
    }

    bool apply( T_vec &x ) const
    {
        return nonlin_op->try_solve_jacobian_system( x );
    }

private:
    std::shared_ptr<NonlinearOperator>    owner_;
    std::shared_ptr<const LinearOperator> operator_owner_;
    NonlinearOperator                    *nonlin_op;
    mutable const LinearOperator         *lin_op = nullptr;
};

} // namespace nonlinear_operators

#endif
