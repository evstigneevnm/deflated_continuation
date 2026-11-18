#ifndef NONLINEAR_OPERATORS_CH_UNSPLIT_PROBLEM_H
#define NONLINEAR_OPERATORS_CH_UNSPLIT_PROBLEM_H

#include "detail/concentration_problem.h"

namespace nonlinear_operators::cahn_hilliard
{
// Complete physical equation with identity mass:
// F(c)=m*Delta_h(c^3-c-epsilon^2*Delta_h(c)), R=a*(c-w)-b*(F(c)+r).
// Its prepared Jacobian includes the concave derivative -1. The problem owns
// no solver, timestep history or method-specific energy acceptance policy.
template <class Space>
class unsplit_problem : public detail::concentration_problem<Space, false>
{
    using base_type = detail::concentration_problem<Space, false>;

public:
    using base_type::base_type;
};
}
#endif
