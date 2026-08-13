#ifndef __STABILITY_ANALYSIS_INITIAL_VECTOR_POLICY_H__
#define __STABILITY_ANALYSIS_INITIAL_VECTOR_POLICY_H__

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace stability
{
namespace analysis
{

namespace detail
{

template<class NonlinearOperations, class Vector, class = void>
struct has_randomize_stability_vector : std::false_type
{
};

template<class NonlinearOperations, class Vector>
struct has_randomize_stability_vector<
    NonlinearOperations,
    Vector,
    std::void_t<decltype(
        std::declval<NonlinearOperations&>().
            randomize_stability_vector(std::declval<Vector&>()))>>
    : std::true_type
{
};

} // namespace detail

template<class NonlinearOperations, class Vector>
void initialize_stability_probe(
    NonlinearOperations& nonlinear_operations,
    Vector& vector)
{
    if constexpr(
        detail::has_randomize_stability_vector<
            NonlinearOperations,
            Vector>::value)
    {
        nonlinear_operations.randomize_stability_vector(vector);
    }
    else
    {
        nonlinear_operations.randomize_vector(vector);
    }
}

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
        initialize_stability_probe(
            *nonlinear_operations_,
            vector);
    }

private:
    NonlinearOperations* nonlinear_operations_;
};

} // namespace analysis
} // namespace stability

#endif
