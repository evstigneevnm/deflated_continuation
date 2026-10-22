#ifndef __NONLINEAR_OPERATORS_FIXED_PARAMETER_OPERATOR_H__
#define __NONLINEAR_OPERATORS_FIXED_PARAMETER_OPERATOR_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <nonlinear_operators/projected_operator_helpers.h>

namespace nonlinear_operators
{

// Bind F(u, parameter) to NMFD's vector-only nonlinear operator interface.
// The supplied Jacobian must act on the same problem and residual formulation.
template <class VectorOperations, class NonlinearOperator, class LinearOperator>
class fixed_parameter_operator
{
public:
    using scalar_type          = typename VectorOperations::scalar_type;
    using vector_type          = typename VectorOperations::vector_type;
    using residual_vector_type = vector_type;
    using jacobi_operator_type = LinearOperator;

    fixed_parameter_operator(
        std::shared_ptr<NonlinearOperator> problem, std::shared_ptr<const LinearOperator> jacobian,
        scalar_type parameter
    )
        : problem_( problem ? std::move( problem ) : throw std::invalid_argument( "Missing nonlinear problem" ) ),
          jacobian_( jacobian ? std::move( jacobian ) : throw std::invalid_argument( "Missing Jacobian operator" ) ),
          parameter_( parameter )
    {
    }

    // Change the bound parameter between solves; require a fresh linearization.
    void set_parameter( scalar_type parameter )
    {
        parameter_ = parameter;
        prepared_  = false;
    }

    scalar_type parameter() const
    {
        return parameter_;
    }

    // Evaluate the selected residual without changing the prepared Jacobian.
    void apply( const vector_type &state, vector_type &residual )
    {
        detail::residual( problem_.get(), state, parameter_, residual );
    }

    void set_linearization_point( const vector_type &state )
    {
        prepared_ = false;
        detail::set_linearization_point( problem_.get(), state, parameter_ );
        prepared_ = true;
    }

    // Share the existing prepared operator, without copying state or workspace.
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        if ( !prepared_ )
            throw std::logic_error( "Prepare the fixed-parameter Jacobian before requesting it" );
        return jacobian_;
    }

private:
    std::shared_ptr<NonlinearOperator>    problem_;
    std::shared_ptr<const LinearOperator> jacobian_;
    scalar_type                           parameter_;
    bool                                  prepared_ = false;
};

}

#endif
