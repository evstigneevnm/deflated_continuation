#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <common/scfd_vector_operations.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/star_shaped/star_shaped.h>

#if defined(TEST_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#include <scfd/backend/cuda.h>
#else
#include <scfd/backend/omp.h>
#endif

namespace
{

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
    vector_space.init_vector(state_vector);
    vector_space.init_vector(rhs);
    vector_space.init_vector(image);
    vector_space.start_use_vector(state_vector);
    vector_space.start_use_vector(rhs);
    vector_space.start_use_vector(image);

    vector_space.set(&state, state_vector, 1);
    nonlinear_operator.set_linearization_point(
        state_vector,
        parameter);

    const double original_rhs = 1.375;
    vector_space.set(&original_rhs, rhs, 1);
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

    vector_space.stop_use_vector(image);
    vector_space.stop_use_vector(rhs);
    vector_space.stop_use_vector(state_vector);
    vector_space.free_vector(image);
    vector_space.free_vector(rhs);
    vector_space.free_vector(state_vector);
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
