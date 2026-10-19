#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_PRECONDITIONER_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_PRECONDITIONER_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>

namespace symmetry
{
namespace linearization
{

template <class VectorOperations, class NonlinearOperator, class LinearOperator>
class projected_preconditioner
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit projected_preconditioner( NonlinearOperator *nonlin_op_ )
        : nonlin_op(
              nonlin_op_ ? nonlin_op_ : throw std::invalid_argument( "projected preconditioner requires a problem" )
          ),
          vec_ops( nonlin_op->get_vec_ops_ref() ), projected_storage_( *vec_ops ), gauge_storage_( *vec_ops )
    {
        projected_storage_.init();
        projected_storage_.start_use();
        gauge_storage_.init();
        gauge_storage_.start_use();
    }

    explicit projected_preconditioner( std::shared_ptr<NonlinearOperator> problem )
        : projected_preconditioner( problem.get() )
    {
        problem_owner_ = std::move( problem );
    }

    projected_preconditioner( const projected_preconditioner & )            = delete;
    projected_preconditioner &operator=( const projected_preconditioner & ) = delete;

    void set_operator( const LinearOperator *op_ ) const
    {
        if ( op_ != operator_owner_.get() )
            operator_owner_.reset();
        lin_op = op_;
    }

    void set_operator( std::shared_ptr<const LinearOperator> op ) const
    {
        operator_owner_ = std::move( op );
        lin_op          = operator_owner_.get();
    }

    void apply( vector_type &x ) const
    {
        nonlin_op->project_current_tangent( x, projected_x );
        vec_ops->assign_mul( scalar_type( 1 ), x, scalar_type( -1 ), projected_x, gauge_component );
        vec_ops->assign( projected_x, x );
        nonlin_op->preconditioner_jacobian_u( x );
        nonlin_op->project_current_tangent( x, x );
        vec_ops->add_mul( scalar_type( 1 ), gauge_component, x );
    }

private:
    std::shared_ptr<NonlinearOperator>                                problem_owner_;
    mutable std::shared_ptr<const LinearOperator>                     operator_owner_;
    NonlinearOperator                                                *nonlin_op;
    VectorOperations                                                 *vec_ops;
    mutable nmfd::detail::vector_wrap<VectorOperations, false, false> projected_storage_, gauge_storage_;
    vector_type                                                      &projected_x     = *projected_storage_;
    vector_type                                                      &gauge_component = *gauge_storage_;
    mutable const LinearOperator                                     *lin_op          = nullptr;
};

} // namespace linearization
} // namespace symmetry

#endif // __SYMMETRY_LINEARIZATION_PROJECTED_PRECONDITIONER_H__
