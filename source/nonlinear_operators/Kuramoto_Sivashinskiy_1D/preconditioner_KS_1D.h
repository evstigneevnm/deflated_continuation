#ifndef __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_1D_H__
#define __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_1D_H__

#include <memory>
#include <utility>

namespace nonlinear_operators
{

template <class VectorOperations, class NonlinearOperator, class LinearOperator>
class preconditioner_KS_1D
{
public:
    using T     = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    explicit preconditioner_KS_1D( NonlinearOperator *nonlin_op_ ) : nonlin_op( nonlin_op_ )
    {
    }

    explicit preconditioner_KS_1D( std::shared_ptr<NonlinearOperator> nonlin_op_ )
        : owner_( std::move( nonlin_op_ ) ), nonlin_op( owner_.get() )
    {
    }

    void set_operator( std::shared_ptr<const LinearOperator> op ) const
    {
        operator_owner_ = std::move( op );
        lin_op          = operator_owner_.get();
    }

    void set_operator( const LinearOperator *op_ ) const
    {
        operator_owner_.reset();
        lin_op = op_;
        (void)lin_op;
    }

    void apply( T_vec &x ) const
    {
        nonlin_op->preconditioner_jacobian_u( x );
    }

private:
    std::shared_ptr<NonlinearOperator>            owner_;
    mutable std::shared_ptr<const LinearOperator> operator_owner_;
    NonlinearOperator                            *nonlin_op;
    mutable const LinearOperator                 *lin_op = nullptr;
};

} // namespace nonlinear_operators

#endif
