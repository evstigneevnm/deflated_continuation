#ifndef TIME_STEPPER_SEMIDISCRETE_AUTONOMOUS_SPATIAL_PROBLEM_H
#define TIME_STEPPER_SEMIDISCRETE_AUTONOMOUS_SPATIAL_PROBLEM_H

#include <stdexcept>

#include <time_stepper/semidiscrete/problem_traits.h>

namespace time_steppers
{
namespace semidiscrete
{

struct linear_implicit_nonlinear_explicit
{
    template<class SpatialOperator, class Vector, class Parameter>
    static void implicit_residual(
        SpatialOperator& spatial_operator,
        const Vector& state,
        const Parameter& parameter,
        Vector& output)
    {
        spatial_operator.linear_residual(state, parameter, output);
    }

    template<class SpatialOperator, class Vector, class Parameter>
    static void explicit_residual(
        SpatialOperator& spatial_operator,
        const Vector& state,
        const Parameter& parameter,
        Vector& output)
    {
        spatial_operator.nonlinear_residual(state, parameter, output);
    }
};

template<
    class VectorOperations,
    class SpatialOperator,
    class Parameter = typename VectorOperations::scalar_type>
class autonomous_identity_mass_problem
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using parameter_type = Parameter;
    using mass_matrix_type = identity_mass_matrix;

    autonomous_identity_mass_problem(
        VectorOperations* vector_operations,
        SpatialOperator* spatial_operator):
        vector_operations_(vector_operations),
        spatial_operator_(spatial_operator)
    {
        if(vector_operations_ == nullptr || spatial_operator_ == nullptr)
        {
            throw std::invalid_argument("autonomous_identity_mass_problem requires non-null operations and spatial operator pointers.");
        }
    }

    void mass_action(
        const scalar_type,
        const vector_type&,
        const vector_type& state_rate,
        const parameter_type&,
        vector_type& output)
    {
        vector_operations_->assign(state_rate, output);
    }

    void residual(
        const scalar_type,
        const vector_type& state,
        const parameter_type& parameter,
        vector_type& output)
    {
        spatial_operator_->F(state, parameter, output);
    }

protected:
    SpatialOperator& spatial_operator()
    {
        return *spatial_operator_;
    }

private:
    VectorOperations* vector_operations_;
    SpatialOperator* spatial_operator_;
};

template<
    class VectorOperations,
    class SpatialOperator,
    class SplitPolicy = linear_implicit_nonlinear_explicit,
    class Parameter = typename VectorOperations::scalar_type>
class autonomous_identity_mass_split_problem:
    public autonomous_identity_mass_problem<VectorOperations, SpatialOperator, Parameter>
{
public:
    using base_type = autonomous_identity_mass_problem<VectorOperations, SpatialOperator, Parameter>;
    using scalar_type = typename base_type::scalar_type;
    using vector_type = typename base_type::vector_type;
    using parameter_type = typename base_type::parameter_type;
    using mass_matrix_type = typename base_type::mass_matrix_type;

    using base_type::base_type;

    void implicit_residual(
        const scalar_type,
        const vector_type& state,
        const parameter_type& parameter,
        vector_type& output)
    {
        SplitPolicy::implicit_residual(this->spatial_operator(), state, parameter, output);
    }

    void explicit_residual(
        const scalar_type,
        const vector_type& state,
        const parameter_type& parameter,
        vector_type& output)
    {
        SplitPolicy::explicit_residual(this->spatial_operator(), state, parameter, output);
    }
};

} // namespace semidiscrete
} // namespace time_steppers

#endif
