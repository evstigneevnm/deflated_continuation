#ifndef NONLINEAR_OPERATORS_CH_CONVEX_SPLIT_PROBLEM_H
#define NONLINEAR_OPERATORS_CH_CONVEX_SPLIT_PROBLEM_H

#include "detail/concentration_problem.h"
#include <nmfd/time_steppers/runge_kutta/imex_operator.h>

namespace nonlinear_operators::cahn_hilliard
{
// The explicit expansive rate is -m*Delta(c). No implicit workspace is needed.
template <class Space>
class explicit_operator
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using model_type  = model<scalar_type>;
    explicit_operator( std::shared_ptr<Space> ops, std::shared_ptr<const model_type> physics )
        : ops_( detail::checked_space( std::move( ops ), physics ) ), model_( std::move( physics ) )
    {
    }
    void set_time( scalar_type )
    {
    } // Autonomous partition.
    struct mapping
    {
        model_type          model;
        vector_type         in, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            out( i ) = -model.mobility * model.grid.apply( in, i, 0, 1 );
        }
    };
    bool evaluate_rate( const vector_type &state, vector_type &rate ) const
    {
        typename Space::for_each_type each;
        each( mapping{ *model_, state, rate }, model_->grid.size() );
        each.wait();
        return true;
    }

private:
    std::shared_ptr<Space>            ops_;
    std::shared_ptr<const model_type> model_;
};

// R=a*(c-w)-b*(m*Delta(c^3-epsilon^2*Delta(c))+r). Two backend passes
// retain a five-point stencil rather than forming a biharmonic matrix. Each
// mutable problem/Jacobian is dedicated to one serial evaluation context.
template <class Space>
class implicit_operator : public detail::concentration_problem<Space, true>
{
    using base_type = detail::concentration_problem<Space, true>;

public:
    using base_type::base_type;
};

template <class Space>
class convex_split_problem
{
public:
    using model_type             = model<typename Space::scalar_type>;
    using explicit_operator_type = explicit_operator<Space>;
    using implicit_operator_type = implicit_operator<Space>;
    using imex_operator_type =
        nmfd::time_steppers::runge_kutta::imex_operator<explicit_operator_type, implicit_operator_type>;
    convex_split_problem( std::shared_ptr<Space> ops, std::shared_ptr<const model_type> physics )
        : bundle_(
              std::make_shared<imex_operator_type>(
                  std::make_shared<explicit_operator_type>( ops, physics ),
                  std::make_shared<implicit_operator_type>( ops, physics )
              )
          )
    {
    }
    std::shared_ptr<imex_operator_type> get_imex_operator() const
    {
        return bundle_;
    }

private:
    std::shared_ptr<imex_operator_type> bundle_;
};
}
#endif
