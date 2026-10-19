#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_LINEAR_OPERATOR_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_LINEAR_OPERATOR_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>

namespace symmetry
{
namespace linearization
{

template <class VectorOperations, class NonlinearOperator>
class projected_linear_operator
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit projected_linear_operator( NonlinearOperator *nonlin_op_, scalar_type gauge_completion = scalar_type( 1 ) )
        : nonlin_op( nonlin_op_ ? nonlin_op_ : throw std::invalid_argument( "projected operator requires a problem" ) ),
          vec_ops( nonlin_op->get_vec_ops_ref() ), gauge_completion_( gauge_completion ),
          projected_storage_( *vec_ops ), gauge_storage_( *vec_ops )
    {
        projected_storage_.init();
        projected_storage_.start_use();
        gauge_storage_.init();
        gauge_storage_.start_use();
    }

    explicit projected_linear_operator(
        std::shared_ptr<NonlinearOperator> problem, scalar_type gauge_completion = scalar_type( 1 )
    )
        : projected_linear_operator( problem.get(), gauge_completion )
    {
        problem_owner_ = std::move( problem );
    }

    projected_linear_operator( const projected_linear_operator & )            = delete;
    projected_linear_operator &operator=( const projected_linear_operator & ) = delete;

    void apply( const vector_type &x, vector_type &f ) const
    {
        nonlin_op->project_current_tangent( x, projected_x );
        nonlin_op->projected_jacobian_u( projected_x, f );

        if ( gauge_completion_ != scalar_type( 0 ) )
        {
            vec_ops->assign_mul( scalar_type( 1 ), x, scalar_type( -1 ), projected_x, gauge_component );
            vec_ops->add_mul( gauge_completion_, gauge_component, f );
        }
    }

    scalar_type gauge_completion() const
    {
        return gauge_completion_;
    }

private:
    std::shared_ptr<NonlinearOperator>                                problem_owner_;
    NonlinearOperator                                                *nonlin_op;
    VectorOperations                                                 *vec_ops;
    scalar_type                                                       gauge_completion_;
    mutable nmfd::detail::vector_wrap<VectorOperations, false, false> projected_storage_, gauge_storage_;
    vector_type                                                      &projected_x     = *projected_storage_;
    vector_type                                                      &gauge_component = *gauge_storage_;
};

} // namespace linearization
} // namespace symmetry

#endif // __SYMMETRY_LINEARIZATION_PROJECTED_LINEAR_OPERATOR_H__
