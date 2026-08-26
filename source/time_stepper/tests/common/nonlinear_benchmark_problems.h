#ifndef TIME_STEPPER_TESTS_COMMON_NONLINEAR_BENCHMARK_PROBLEMS_H
#define TIME_STEPPER_TESTS_COMMON_NONLINEAR_BENCHMARK_PROBLEMS_H

#include <array>
#include <cmath>
#include <utility>

namespace time_steppers
{
namespace tests
{

template<class VectorOperations>
struct lorenz_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    void F(
        const scalar_type,
        const vector_type& state,
        const scalar_type,
        vector_type& output) const
    {
        output(0) = sigma*(state(1)-state(0));
        output(1) = state(0)*(rho-state(2))-state(1);
        output(2) = state(0)*state(1)-beta*state(2);
    }

    scalar_type sigma = 10;
    scalar_type rho = 28;
    scalar_type beta = scalar_type(8)/scalar_type(3);
};

template<class VectorOperations>
struct rossler_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    void F(
        const scalar_type,
        const vector_type& state,
        const scalar_type,
        vector_type& output) const
    {
        output(0) = -state(1)-state(2);
        output(1) = state(0)+a*state(1);
        output(2) = b+state(2)*(state(0)-c);
    }

    scalar_type a = scalar_type(0.2);
    scalar_type b = scalar_type(0.2);
    scalar_type c = scalar_type(5.7);
};

template<class VectorOperations>
class van_der_pol_problem
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    void F(
        const scalar_type,
        const vector_type& state,
        const scalar_type mu,
        vector_type& output) const
    {
        output(0) = state(1);
        output(1) = mu*(scalar_type(1)-state(0)*state(0))*state(1)-state(0);
    }

    void set_linearization_point(const vector_type& state, const scalar_type mu)
    {
        state_[0] = state(0);
        state_[1] = state(1);
        mu_ = mu;
    }

    class linear_operator
    {
    public:
        explicit linear_operator(const van_der_pol_problem* problem): problem_(problem)
        {}

        void set_aE_plus_bA(const std::pair<scalar_type, scalar_type>& coefficients)
        {
            coefficients_ = coefficients;
        }

        bool solve(const vector_type& right_hand_side, vector_type& solution) const
        {
            const scalar_type x = problem_->state_[0];
            const scalar_type y = problem_->state_[1];
            const scalar_type mu = problem_->mu_;
            const scalar_type jacobian_10 = -scalar_type(2)*mu*x*y-scalar_type(1);
            const scalar_type jacobian_11 = mu*(scalar_type(1)-x*x);
            const scalar_type a = coefficients_.first;
            const scalar_type b = coefficients_.second;
            const scalar_type matrix_00 = a;
            const scalar_type matrix_01 = b;
            const scalar_type matrix_10 = b*jacobian_10;
            const scalar_type matrix_11 = a+b*jacobian_11;
            const scalar_type determinant =
                matrix_00*matrix_11-matrix_01*matrix_10;
            if(!std::isfinite(determinant) || determinant == scalar_type(0))
            {
                return false;
            }
            solution(0) =
                (matrix_11*right_hand_side(0)-matrix_01*right_hand_side(1))/determinant;
            solution(1) =
                (-matrix_10*right_hand_side(0)+matrix_00*right_hand_side(1))/determinant;
            return std::isfinite(solution(0)) && std::isfinite(solution(1));
        }

    private:
        const van_der_pol_problem* problem_;
        std::pair<scalar_type, scalar_type> coefficients_{1, 0};
    };

private:
    std::array<scalar_type, 2> state_{{0, 0}};
    scalar_type mu_ = 0;
};

} // namespace tests
} // namespace time_steppers

#endif
