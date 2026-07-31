#ifndef __STABILITY_ANALYSIS_INITIAL_VECTOR_POLICY_H__
#define __STABILITY_ANALYSIS_INITIAL_VECTOR_POLICY_H__

#include <stdexcept>

namespace stability
{
namespace analysis
{

template<class VectorOperations>
class vector_operations_random_initial_vector
{
public:
    using vector_type = typename VectorOperations::vector_type;

    explicit vector_operations_random_initial_vector(
        VectorOperations* vector_operations)
        : vector_operations_(vector_operations)
    {
        if(vector_operations_ == nullptr)
            throw std::invalid_argument(
                "vector_operations_random_initial_vector: "
                "vector operations are null");
    }

    void operator()(vector_type& vector) const
    {
        vector_operations_->assign_random(vector);
    }

private:
    VectorOperations* vector_operations_;
};

template<class NonlinearOperations>
class nonlinear_operator_random_initial_vector
{
public:
    explicit nonlinear_operator_random_initial_vector(
        NonlinearOperations* nonlinear_operations)
        : nonlinear_operations_(nonlinear_operations)
    {
        if(nonlinear_operations_ == nullptr)
            throw std::invalid_argument(
                "nonlinear_operator_random_initial_vector: "
                "nonlinear operator is null");
    }

    template<class Vector>
    void operator()(Vector& vector) const
    {
        nonlinear_operations_->randomize_vector(vector);
    }

private:
    NonlinearOperations* nonlinear_operations_;
};

} // namespace analysis
} // namespace stability

#endif
