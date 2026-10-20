#ifndef __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_2D_H__
#define __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_2D_H__

#include <memory>
#include <utility>

namespace nonlinear_operators
{

template <class VectorOperations, class NonlinearOperator, class LinearOperator>
class preconditioner_KS_2D
{
public:
    using vector_type = typename VectorOperations::vector_type;

    explicit preconditioner_KS_2D( NonlinearOperator *nonlinear_operator ) : nonlinear_operator_( nonlinear_operator )
    {
    }

    explicit preconditioner_KS_2D( std::shared_ptr<NonlinearOperator> problem )
        : problem_owner_( std::move( problem ) ), nonlinear_operator_( problem_owner_.get() )
    {
    }

    void set_operator( const LinearOperator *linear_operator ) const
    {
        if ( linear_operator != operator_owner_.get() )
            operator_owner_.reset();
        linear_operator_ = linear_operator;
    }

    void set_operator( std::shared_ptr<const LinearOperator> linear_operator ) const
    {
        operator_owner_  = std::move( linear_operator );
        linear_operator_ = operator_owner_.get();
    }

    void apply( vector_type &input_output ) const
    {
        nonlinear_operator_->preconditioner_jacobian_u( input_output );
    }

private:
    std::shared_ptr<NonlinearOperator>            problem_owner_;
    mutable std::shared_ptr<const LinearOperator> operator_owner_;
    NonlinearOperator                            *nonlinear_operator_;
    mutable const LinearOperator                 *linear_operator_ = nullptr;
};

} // namespace nonlinear_operators

#endif
