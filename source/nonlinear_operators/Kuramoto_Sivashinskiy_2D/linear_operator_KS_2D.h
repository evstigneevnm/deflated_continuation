#ifndef __LINEAR_OPERATOR_KURAMOTO_SIVASHINSKIY_2D_H__
#define __LINEAR_OPERATOR_KURAMOTO_SIVASHINSKIY_2D_H__

#include <memory>
#include <utility>

namespace nonlinear_operators
{

template <class VectorOperations, class NonlinearOperator>
class linear_operator_KS_2D
{
public:
    using vector_type = typename VectorOperations::vector_type;

    explicit linear_operator_KS_2D( NonlinearOperator *nonlinear_operator ) : nonlinear_operator_( nonlinear_operator )
    {
    }

    explicit linear_operator_KS_2D( std::shared_ptr<NonlinearOperator> problem )
        : problem_owner_( std::move( problem ) ), nonlinear_operator_( problem_owner_.get() )
    {
    }

    void apply( const vector_type &input, vector_type &output ) const
    {
        nonlinear_operator_->jacobian_u( input, output );
    }

private:
    std::shared_ptr<NonlinearOperator> problem_owner_;
    NonlinearOperator                 *nonlinear_operator_;
};

} // namespace nonlinear_operators

#endif
