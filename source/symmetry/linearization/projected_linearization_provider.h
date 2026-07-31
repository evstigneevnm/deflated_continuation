#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_LINEARIZATION_PROVIDER_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_LINEARIZATION_PROVIDER_H__

namespace symmetry
{
namespace linearization
{

namespace detail
{

template<class NonlinearOperator, class Vector, class Scalar>
auto set_stateless_projected_linearization_point(
    NonlinearOperator& nonlinear_operator,
    const Vector& state,
    Scalar parameter,
    int)
    -> decltype(
        nonlinear_operator.
            set_stateless_projected_linearization_point(
                state,
                parameter),
        void())
{
    nonlinear_operator.
        set_stateless_projected_linearization_point(
            state,
            parameter);
}

template<class NonlinearOperator, class Vector, class Scalar>
void set_stateless_projected_linearization_point(
    NonlinearOperator& nonlinear_operator,
    const Vector& state,
    Scalar parameter,
    long)
{
    nonlinear_operator.set_projected_linearization_point(
        state,
        parameter);
}

} // namespace detail

template<class VectorOperations, class NonlinearOperator>
class projected_linearization_provider
{
public:
    using vector_operations_type = VectorOperations;
    using nonlinear_operator_type = NonlinearOperator;
    using scalar_type = typename vector_operations_type::scalar_type;
    using vector_type = typename vector_operations_type::vector_type;

    explicit projected_linearization_provider(
        nonlinear_operator_type& nonlinear_operator)
        : nonlinear_operator_(nonlinear_operator)
    {
    }

    void set_linearization_point(
        const vector_type& state,
        scalar_type parameter)
    {
        detail::set_stateless_projected_linearization_point(
            nonlinear_operator_,
            state,
            parameter,
            0);
    }

    nonlinear_operator_type& nonlinear_operator()
    {
        return nonlinear_operator_;
    }

    const nonlinear_operator_type& nonlinear_operator() const
    {
        return nonlinear_operator_;
    }

private:
    nonlinear_operator_type& nonlinear_operator_;
};

} // namespace linearization
} // namespace symmetry

#endif
