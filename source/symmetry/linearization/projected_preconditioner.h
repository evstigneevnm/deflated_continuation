#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_PRECONDITIONER_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_PRECONDITIONER_H__

namespace symmetry
{
namespace linearization
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator>
class projected_preconditioner
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit projected_preconditioner(NonlinearOperator* nonlin_op_):
        nonlin_op(nonlin_op_),
        vec_ops(nonlin_op_->get_vec_ops_ref())
    {
        vec_ops->init_vector(projected_x);
        vec_ops->start_use_vector(projected_x);
        vec_ops->init_vector(gauge_component);
        vec_ops->start_use_vector(gauge_component);
    }

    ~projected_preconditioner()
    {
        vec_ops->stop_use_vector(gauge_component);
        vec_ops->free_vector(gauge_component);
        vec_ops->stop_use_vector(projected_x);
        vec_ops->free_vector(projected_x);
    }

    projected_preconditioner(const projected_preconditioner&) = delete;
    projected_preconditioner& operator=(const projected_preconditioner&) = delete;

    void set_operator(const LinearOperator* op_) const
    {
        lin_op = op_;
    }

    void apply(vector_type& x) const
    {
        nonlin_op->project_current_tangent(x, projected_x);
        vec_ops->assign_mul(
            scalar_type(1),
            x,
            scalar_type(-1),
            projected_x,
            gauge_component);
        vec_ops->assign(projected_x, x);
        nonlin_op->preconditioner_jacobian_u(x);
        nonlin_op->project_current_tangent(x, x);
        vec_ops->add_mul(scalar_type(1), gauge_component, x);
    }

private:
    NonlinearOperator* nonlin_op;
    VectorOperations* vec_ops;
    mutable vector_type projected_x;
    mutable vector_type gauge_component;
    mutable const LinearOperator* lin_op = nullptr;
};

} // namespace linearization
} // namespace symmetry

#endif // __SYMMETRY_LINEARIZATION_PROJECTED_PRECONDITIONER_H__
