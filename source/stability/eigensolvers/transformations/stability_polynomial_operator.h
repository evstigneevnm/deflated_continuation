#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_STABILITY_POLYNOMIAL_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_STABILITY_POLYNOMIAL_OPERATOR_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class Real>
std::vector<Real> explicit_euler_stability_polynomial()
{
    return {Real(1), Real(1)};
}

template<class Real>
std::vector<Real> classical_rk4_stability_polynomial()
{
    return {
        Real(1),
        Real(1),
        Real(1) / Real(2),
        Real(1) / Real(6),
        Real(1) / Real(24)};
}

template<class Scalar, class Coefficient>
Scalar evaluate_stability_polynomial(
    const std::vector<Coefficient>& coefficients,
    const Scalar& argument)
{
    if(coefficients.empty())
        throw std::invalid_argument(
            "stability polynomial coefficients must not be empty");
    Scalar result =
        static_cast<Scalar>(coefficients.back());
    for(std::size_t index = coefficients.size() - 1; index > 0; --index)
    {
        result =
            result * argument +
            static_cast<Scalar>(coefficients[index - 1]);
    }
    return result;
}

template<class Scalar, class Coefficient>
Scalar evaluate_repeated_stability_polynomial(
    const std::vector<Coefficient>& coefficients,
    const Scalar& argument,
    std::size_t repetitions)
{
    if(repetitions == 0)
        throw std::invalid_argument(
            "stability polynomial repetitions must be positive");
    Scalar result = evaluate_stability_polynomial(
        coefficients,
        argument);
    Scalar repeated = Scalar(1);
    for(std::size_t index = 0; index < repetitions; ++index)
        repeated *= result;
    return repeated;
}

template<class VectorSpace, class OriginalOperator>
class stability_polynomial_operator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;

    stability_polynomial_operator(
        const vector_space_type& vector_space,
        const OriginalOperator& original_operator,
        scalar_type step,
        std::size_t repetitions,
        std::vector<scalar_type> coefficients)
        : vector_space_(vector_space),
          original_operator_(original_operator),
          step_(step),
          repetitions_(repetitions),
          coefficients_(std::move(coefficients)),
          current_(vector_space_, true, true),
          accumulator_(vector_space_, true, true),
          stage_(vector_space_, true, true),
          applied_(vector_space_, true, true)
    {
        if(repetitions_ == 0)
            throw std::invalid_argument(
                "stability polynomial repetitions must be positive");
        if(coefficients_.empty())
            throw std::invalid_argument(
                "stability polynomial coefficients must not be empty");
        if(!std::isfinite(step_))
            throw std::invalid_argument(
                "stability polynomial step must be finite");
        for(const auto coefficient : coefficients_)
        {
            if(!std::isfinite(coefficient))
                throw std::invalid_argument(
                    "stability polynomial coefficient must be finite");
        }
    }

    bool apply(const vector_type& source, vector_type& destination) const
    {
        ++operator_calls_;
        vector_space_.assign(source, *current_);
        vector_type* current = &*current_;
        vector_type* accumulator = &*accumulator_;
        vector_type* stage = &*stage_;

        for(std::size_t repetition = 0;
            repetition < repetitions_;
            ++repetition)
        {
            vector_space_.assign_lin_comb(
                coefficients_.back(),
                *current,
                *accumulator);
            for(
                std::size_t index = coefficients_.size() - 1;
                index > 0;
                --index)
            {
                ++original_operator_calls_;
                if(!nmfd::solvers::krylov::apply_operator(
                       original_operator_,
                       *accumulator,
                       *applied_))
                {
                    ++original_operator_failures_;
                    return false;
                }
                vector_space_.assign_lin_comb(
                    coefficients_[index - 1],
                    *current,
                    step_,
                    *applied_,
                    *stage);
                std::swap(accumulator, stage);
            }
            std::swap(current, accumulator);
        }

        vector_space_.assign(*current, destination);
        return true;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    std::size_t original_operator_calls() const
    {
        return original_operator_calls_;
    }

    std::size_t original_operator_failures() const
    {
        return original_operator_failures_;
    }

    std::size_t degree() const
    {
        return coefficients_.size() - 1;
    }

    std::size_t repetitions() const
    {
        return repetitions_;
    }

    scalar_type step() const
    {
        return step_;
    }

    const std::vector<scalar_type>& coefficients() const
    {
        return coefficients_;
    }

private:
    const vector_space_type& vector_space_;
    const OriginalOperator& original_operator_;
    scalar_type step_;
    std::size_t repetitions_;
    std::vector<scalar_type> coefficients_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> current_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true>
        accumulator_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> stage_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> applied_;
    mutable std::size_t operator_calls_ = 0;
    mutable std::size_t original_operator_calls_ = 0;
    mutable std::size_t original_operator_failures_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
