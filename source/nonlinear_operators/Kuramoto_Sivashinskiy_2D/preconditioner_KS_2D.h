#ifndef __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_2D_H__
#define __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_2D_H__

#include <memory>

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator>
class preconditioner_KS_2D
{
public:
    using vector_type = typename VectorOperations::vector_type;

    explicit preconditioner_KS_2D(NonlinearOperator* nonlinear_operator):
        nonlinear_operator_(nonlinear_operator)
    {
    }

    void set_operator(const LinearOperator* linear_operator) const
    {
        linear_operator_ = linear_operator;
    }

    void set_operator(
        const std::shared_ptr<const LinearOperator>& linear_operator) const
    {
        linear_operator_ = linear_operator.get();
    }

    void apply(vector_type& input_output) const
    {
        nonlinear_operator_->preconditioner_jacobian_u(input_output);
    }

private:
    NonlinearOperator* nonlinear_operator_;
    mutable const LinearOperator* linear_operator_ = nullptr;
};

} // namespace nonlinear_operators

#endif
