#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <common/scfd_vector_operations.h>
#include <nonlinear_operators/adjoint_jacobian_capability.h>
#include <nonlinear_operators/bratu/bratu.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/star_shaped/star_shaped.h>
#include <nonlinear_operators/tests/adjoint_jacobian_test.h>

#if defined(TEST_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#include <scfd/backend/cuda.h>
#else
#include <scfd/backend/omp.h>
#endif

namespace
{

struct operator_without_adjoint
{
    using T_vec = double*;

    void jacobian_u(const T_vec&, T_vec&)
    {
    }
};

static_assert(
    !nonlinear_operators::has_jacobian_u_adjoint_v<operator_without_adjoint>,
    "an operator without jacobian_u_adjoint must not satisfy the capability");
static_assert(
    !nonlinear_operators::has_component_jacobian_u_adjoint_v<operator_without_adjoint>,
    "an operator without component adjoints must not satisfy the capability");
static_assert(
    !nonlinear_operators::has_preconditioner_jacobian_affine_u_adjoint_v<
        operator_without_adjoint>,
    "an operator without an affine adjoint preconditioner must not satisfy the capability");
static_assert(
    !nonlinear_operators::has_affine_preconditioner_adjoint_pair_v<
        operator_without_adjoint>,
    "an operator without affine preconditioners must not satisfy the pair capability");

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

template<class VectorSpace>
double scalar_value(
    VectorSpace& vector_space,
    const typename VectorSpace::vector_type& vector)
{
    double result = 0.0;
    vector_space.get(vector, &result, 1);
    return result;
}

template<class VectorSpace, class NonlinearOperator>
void verify_affine_inverse(
    const std::string& label,
    VectorSpace& vector_space,
    NonlinearOperator& nonlinear_operator,
    double state,
    double parameter,
    double jacobian_scale,
    double identity_shift)
{
    using vector_type = typename VectorSpace::vector_type;
    vector_type state_vector;
    vector_type rhs;
    vector_type image;
    vector_type direction;
    vector_type cotangent;
    vector_space.init_vectors(
        state_vector,
        rhs,
        image,
        direction,
        cotangent);
    vector_space.start_use_vectors(
        state_vector,
        rhs,
        image,
        direction,
        cotangent);

    vector_space.set(&state, state_vector, 1);
    nonlinear_operator.set_linearization_point(
        state_vector,
        parameter);

    const double original_rhs = 1.375;
    const double cotangent_value = -0.825;
    vector_space.set(&original_rhs, rhs, 1);
    vector_space.set(&original_rhs, direction, 1);
    vector_space.set(&cotangent_value, cotangent, 1);
    nonlinear_operator.preconditioner_jacobian_affine_u(
        rhs,
        jacobian_scale,
        identity_shift);
    nonlinear_operator.jacobian_u(rhs, image);
    vector_space.assign_mul(
        jacobian_scale,
        image,
        identity_shift,
        rhs,
        image);

    require(
        std::abs(scalar_value(vector_space, image) -
                 original_rhs) < 1.0e-12,
        label + " affine inverse residual");

    static_assert(
        nonlinear_operators::has_affine_preconditioner_adjoint_pair_v<
            NonlinearOperator>,
        "refactored scalar problem must provide an affine preconditioner adjoint pair");
    nonlinear_operators::tests::check_affine_preconditioner_adjoint(
        vector_space,
        nonlinear_operator,
        state_vector,
        direction,
        cotangent,
        parameter,
        jacobian_scale,
        identity_shift,
        64.0*std::numeric_limits<double>::epsilon(),
        [](const bool condition, const std::string& message)
        {
            require(condition, message);
        },
        label);
    nonlinear_operators::tests::check_affine_adjoint_inverse_residual(
        vector_space,
        nonlinear_operator,
        state_vector,
        cotangent,
        parameter,
        jacobian_scale,
        identity_shift,
        1.0e-12,
        [](const bool condition, const std::string& message)
        {
            require(condition, message);
        },
        label);

    vector_space.stop_use_vectors(
        state_vector,
        rhs,
        image,
        direction,
        cotangent);
    vector_space.free_vectors(
        state_vector,
        rhs,
        image,
        direction,
        cotangent);
}

template<class VectorSpace, class NonlinearOperator>
void verify_adjoint(
    const std::string& label,
    VectorSpace& vector_space,
    NonlinearOperator& nonlinear_operator,
    const double state,
    const double parameter)
{
    static_assert(
        nonlinear_operators::has_jacobian_u_adjoint_v<NonlinearOperator>,
        "refactored scalar problem must provide an adjoint Jacobian action");

    using vector_type = typename VectorSpace::vector_type;
    vector_type state_vector;
    vector_type direction;
    vector_type cotangent;
    vector_space.init_vectors(state_vector, direction, cotangent);
    vector_space.start_use_vectors(state_vector, direction, cotangent);
    const double direction_value = -0.73;
    const double cotangent_value = 1.21;
    vector_space.set(&state, state_vector, 1);
    vector_space.set(&direction_value, direction, 1);
    vector_space.set(&cotangent_value, cotangent, 1);

    nonlinear_operators::tests::check_adjoint_jacobian(
        vector_space,
        nonlinear_operator,
        state_vector,
        direction,
        cotangent,
        parameter,
        64.0*std::numeric_limits<double>::epsilon(),
        [](const bool condition, const std::string& message)
        {
            require(condition, message);
        },
        label);
    vector_space.stop_use_vectors(state_vector, direction, cotangent);
    vector_space.free_vectors(state_vector, direction, cotangent);
}

template<class Backend>
void verify_bratu_adjoint(
    const std::string& label,
    const typename nonlinear_operators::bratu<
        scfd_vector_operations<Backend, double>>::spatial_discretization discretization)
{
    constexpr std::size_t size = 17;
    using vector_space_type = scfd_vector_operations<Backend, double>;
    using problem_type = nonlinear_operators::bratu<vector_space_type>;
    using vector_type = typename vector_space_type::vector_type;
    vector_space_type vector_space(size);
    problem_type problem(size, &vector_space, discretization);
    vector_type state;
    vector_type direction;
    vector_type cotangent;
    vector_space.init_vectors(state, direction, cotangent);
    vector_space.start_use_vectors(state, direction, cotangent);
    std::vector<double> host_state(size);
    std::vector<double> host_direction(size);
    std::vector<double> host_cotangent(size);
    for(std::size_t index = 0; index < size; ++index)
    {
        const double x = static_cast<double>(index + 1)/
            static_cast<double>(size + 1);
        host_state[index] = 0.12*std::sin(std::acos(-1.0)*x);
        host_direction[index] = 0.07*std::cos(2.0*std::acos(-1.0)*x);
        host_cotangent[index] = 0.09*std::sin(3.0*std::acos(-1.0)*x) + 0.01*x;
    }
    vector_space.set(host_state.data(), state, size);
    vector_space.set(host_direction.data(), direction, size);
    vector_space.set(host_cotangent.data(), cotangent, size);
    nonlinear_operators::tests::check_adjoint_jacobian(
        vector_space,
        problem,
        state,
        direction,
        cotangent,
        2.5,
        3.0e-13,
        [](const bool condition, const std::string& message)
        {
            require(condition, message);
        },
        label);
    static_assert(
        nonlinear_operators::has_affine_preconditioner_adjoint_pair_v<
            problem_type>,
        "Bratu must provide an affine preconditioner adjoint pair");
    nonlinear_operators::tests::check_affine_preconditioner_adjoint(
        vector_space,
        problem,
        state,
        direction,
        cotangent,
        2.5,
        0.8,
        -1.3,
        2.0e-11,
        [](const bool condition, const std::string& message)
        {
            require(condition, message);
        },
        label);
    nonlinear_operators::tests::check_affine_adjoint_inverse_residual(
        vector_space,
        problem,
        state,
        cotangent,
        2.5,
        0.8,
        -1.3,
        2.0e-10,
        [](const bool condition, const std::string& message)
        {
            require(condition, message);
        },
        label);
    vector_space.stop_use_vectors(state, direction, cotangent);
    vector_space.free_vectors(state, direction, cotangent);
}

template<class Backend>
void run_backend(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    vector_space_type vector_space(1);

    nonlinear_operators::circle<vector_space_type> circle(
        1.0,
        1,
        &vector_space);
    verify_affine_inverse(
        label + " circle",
        vector_space,
        circle,
        0.6,
        0.8,
        -1.75,
        0.45);
    verify_adjoint(
        label + " circle",
        vector_space,
        circle,
        0.6,
        0.8);

    nonlinear_operators::star_shaped<vector_space_type> star(
        1,
        &vector_space,
        0.35);
    verify_affine_inverse(
        label + " star-shaped",
        vector_space,
        star,
        0.72,
        0.41,
        1.4,
        -0.3);
    verify_adjoint(
        label + " star-shaped",
        vector_space,
        star,
        0.72,
        0.41);

    using bratu_type = nonlinear_operators::bratu<vector_space_type>;
    verify_bratu_adjoint<Backend>(
        label + " Bratu FD3",
        bratu_type::spatial_discretization::fd3);
    verify_bratu_adjoint<Backend>(
        label + " Bratu Chebyshev",
        bratu_type::spatial_discretization::chebyshev);
}

} // namespace

int main(int argc, char** argv)
{
#if defined(TEST_VECTOR_BACKEND_CUDA)
    const std::string selector =
        argc > 1 ? argv[1] : "auto";
    const int device =
        common::init_cuda_from_scfd_selector(selector);
    std::cout << "Using CUDA device " << device << '\n';
    run_backend<scfd::backend::cuda>("CUDA");
#else
    (void)argc;
    (void)argv;
    run_backend<scfd::backend::omp>("OMP");
#endif
    std::cout
        << "Scalar affine inverse checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
