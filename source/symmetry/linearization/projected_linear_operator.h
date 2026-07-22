#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_LINEAR_OPERATOR_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_LINEAR_OPERATOR_H__

namespace symmetry
{
namespace linearization
{

template<class VectorOperations, class NonlinearOperator>
class projected_linear_operator
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit projected_linear_operator(NonlinearOperator* nonlin_op_):
        nonlin_op(nonlin_op_),
        vec_ops(nonlin_op_->get_vec_ops_ref())
    {
        vec_ops->init_vector(projected_x);
        vec_ops->start_use_vector(projected_x);
        vec_ops->init_vector(gauge_component);
        vec_ops->start_use_vector(gauge_component);
    }

    ~projected_linear_operator()
    {
        vec_ops->stop_use_vector(gauge_component);
        vec_ops->free_vector(gauge_component);
        vec_ops->stop_use_vector(projected_x);
        vec_ops->free_vector(projected_x);
    }

    projected_linear_operator(const projected_linear_operator&) = delete;
    projected_linear_operator& operator=(const projected_linear_operator&) = delete;

    void apply(const vector_type& x, vector_type& f) const
    {
        nonlin_op->project_current_tangent(x, projected_x);
        nonlin_op->projected_jacobian_u(projected_x, f);

        // Complete the group-tangent kernel of PJP with identity on the gauge
        // component, leaving PJP unchanged on the slice tangent space.
        vec_ops->assign_mul(
            scalar_type(1),
            x,
            scalar_type(-1),
            projected_x,
            gauge_component);
        vec_ops->add_mul(scalar_type(1), gauge_component, f);
    }

private:
    NonlinearOperator* nonlin_op;
    VectorOperations* vec_ops;
    mutable vector_type projected_x;
    mutable vector_type gauge_component;
};

} // namespace linearization
} // namespace symmetry

#endif // __SYMMETRY_LINEARIZATION_PROJECTED_LINEAR_OPERATOR_H__
