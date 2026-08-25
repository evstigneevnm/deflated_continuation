#ifndef __NONLINEAR_OPERATORS_DETAIL_LINEAR_NONLINEAR_TERMS_H__
#define __NONLINEAR_OPERATORS_DETAIL_LINEAR_NONLINEAR_TERMS_H__

namespace nonlinear_operators
{
namespace detail
{

enum class linear_nonlinear_terms
{
    linear,
    nonlinear,
    all
};

template <linear_nonlinear_terms Terms>
constexpr bool includes_linear()
{
    return Terms != linear_nonlinear_terms::nonlinear;
}

template <linear_nonlinear_terms Terms>
constexpr bool includes_nonlinear()
{
    return Terms != linear_nonlinear_terms::linear;
}

} // namespace detail
} // namespace nonlinear_operators

#endif
