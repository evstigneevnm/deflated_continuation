#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(KS1D_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#endif

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/fourier/real_packed_fourier_actions_1d.h>

#include "KS1D_backend_typedefs.h"

namespace
{

using ks1d_t = nonlinear_operators::kuramoto_sivashinskiy_1d<vec_ops_real, fft_backend_t, Blocks_x_>;
using real_vec = typename vec_ops_real::vector_type;
using finite_actions_t = symmetry::finite_action_registry<vec_ops_real>;

int checks = 0;
int failures = 0;

template<class T>
T tolerance()
{
    return std::is_same<T, float>::value ? T(5.0e-3) : T(2.0e-8);
}

template<class T>
T fd_step()
{
    return std::is_same<T, float>::value ? T(1.0e-3) : T(1.0e-6);
}

void record_failure(const std::string& message)
{
    ++failures;
    std::cerr << "FAIL " << message << std::endl;
}

template<class T>
void check_close(T value, T expected, T tol, const std::string& label)
{
    ++checks;
    const T err = common::scalar_math::abs(value - expected);
    if(!(err <= tol))
    {
        record_failure(
            label +
            " value=" + std::to_string(static_cast<double>(value)) +
            " expected=" + std::to_string(static_cast<double>(expected)) +
            " err=" + std::to_string(static_cast<double>(err)) +
            " tol=" + std::to_string(static_cast<double>(tol))
        );
    }
}

void check_condition(bool condition, const std::string& label)
{
    ++checks;
    if(!condition)
    {
        record_failure(label);
    }
}

void set_mode(vec_ops_real& vec_ops, real_vec& x, std::size_t mode, real amplitude)
{
    std::vector<real> host(vec_ops.get_default_size(), real(0));
    if(mode == 0 || mode > host.size())
    {
        throw std::runtime_error("set_mode: mode is outside reduced KS1D vector.");
    }
    host[mode - 1] = amplitude;
    vec_ops.set(host.data(), x, host.size());
}

std::vector<real> host_vector(vec_ops_real& vec_ops, const real_vec& x, std::size_t n)
{
    std::vector<real> host(n, real(0));
    vec_ops.get(x, host.data(), n);
    return host;
}

void check_vector_close(
    vec_ops_real& vec_ops,
    const real_vec& value,
    const real_vec& expected,
    const real tol,
    const std::string& label)
{
    const auto value_host = host_vector(vec_ops, value, vec_ops.get_size(value));
    const auto expected_host = host_vector(vec_ops, expected, vec_ops.get_size(expected));
    check_condition(value_host.size() == expected_host.size(), label + " size");
    if(value_host.size() != expected_host.size())
    {
        return;
    }
    for(std::size_t i = 0; i < value_host.size(); ++i)
    {
        check_close(
            value_host[i],
            expected_host[i],
            tol*(real(1) + common::scalar_math::abs(expected_host[i])),
            label + " component " + std::to_string(i));
    }
}

void test_zero_branch(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec x;
    real_vec f;
    vec_ops.init_vector(x);
    vec_ops.start_use_vector(x);
    vec_ops.init_vector(f);
    vec_ops.start_use_vector(f);

    vec_ops.assign_scalar(real(0), x);
    ks.F(x, real(3.7), f);
    check_close(vec_ops.norm(f), real(0), tolerance<real>(), "zero branch residual");

    vec_ops.stop_use_vector(f);
    vec_ops.free_vector(f);
    vec_ops.stop_use_vector(x);
    vec_ops.free_vector(x);
}

void test_linear_spectrum(vec_ops_real& vec_ops, std::size_t physical_size)
{
    ks1d_t ks_linear(real(0), real(4), physical_size, &vec_ops);

    real_vec x;
    real_vec f;
    vec_ops.init_vector(x);
    vec_ops.start_use_vector(x);
    vec_ops.init_vector(f);
    vec_ops.start_use_vector(f);

    const std::size_t mode = 3;
    const real lambda = real(1.25);
    const real amplitude = real(0.375);
    set_mode(vec_ops, x, mode, amplitude);
    ks_linear.F(x, lambda, f);

    const auto host_f = host_vector(vec_ops, f, vec_ops.get_default_size());
    for(std::size_t i = 0; i < host_f.size(); ++i)
    {
        const real expected = (i + 1 == mode) ? ks_linear.linear_multiplier(mode, lambda)*amplitude : real(0);
        check_close(
            host_f[i],
            expected,
            tolerance<real>()*(real(1) + common::scalar_math::abs(expected)),
            "linear spectrum mode " + std::to_string(i + 1)
        );
    }

    vec_ops.stop_use_vector(f);
    vec_ops.free_vector(f);
    vec_ops.stop_use_vector(x);
    vec_ops.free_vector(x);
}

void test_physical_oddness(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec x;
    real_vec physical;
    vec_ops.init_vector(x);
    vec_ops.start_use_vector(x);
    vec_ops.init_vector(physical);
    vec_ops.start_use_vector(physical, ks.physical_size());

    set_mode(vec_ops, x, 2, real(1));
    ks.physical_solution(x, physical);

    const auto host = host_vector(vec_ops, physical, ks.physical_size());
    check_close(host[0], real(0), tolerance<real>(), "physical oddness x=0");
    check_close(host[ks.physical_size()/2], real(0), tolerance<real>(), "physical oddness x=pi");
    for(std::size_t i = 1; i < ks.physical_size(); ++i)
    {
        const std::size_t mirror = ks.physical_size() - i;
        check_close(
            host[i] + host[mirror],
            real(0),
            tolerance<real>(),
            "physical oddness pair " + std::to_string(i)
        );
    }

    vec_ops.stop_use_vector(physical);
    vec_ops.free_vector(physical);
    vec_ops.stop_use_vector(x);
    vec_ops.free_vector(x);
}

void fill_test_vectors(vec_ops_real& vec_ops, real_vec& u, real_vec& du)
{
    std::vector<real> host_u(vec_ops.get_default_size(), real(0));
    std::vector<real> host_du(vec_ops.get_default_size(), real(0));
    for(std::size_t i = 0; i < host_u.size(); ++i)
    {
        const real k = static_cast<real>(i + 1);
        host_u[i] = real(0.15)*static_cast<real>((i%3) + 1)/(k*k);
        host_du[i] = real(0.08)*static_cast<real>((i%2 == 0) ? 1 : -1)/k;
    }
    vec_ops.set(host_u.data(), u, host_u.size());
    vec_ops.set(host_du.data(), du, host_du.size());
}

void test_jacobian_u(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec u;
    real_vec du;
    real_vec u_plus;
    real_vec u_minus;
    real_vec f_plus;
    real_vec f_minus;
    real_vec finite_difference;
    real_vec jacobian_du;
    vec_ops.init_vectors(u, du, u_plus, u_minus, f_plus, f_minus, finite_difference, jacobian_du);
    vec_ops.start_use_vectors(u, du, u_plus, u_minus, f_plus, f_minus, finite_difference, jacobian_du);

    fill_test_vectors(vec_ops, u, du);
    const real lambda = real(3.3);
    const real eps = fd_step<real>();
    vec_ops.assign_mul(real(1), u, eps, du, u_plus);
    vec_ops.assign_mul(real(1), u, -eps, du, u_minus);
    ks.F(u_plus, lambda, f_plus);
    ks.F(u_minus, lambda, f_minus);
    vec_ops.assign_mul(real(1)/(real(2)*eps), f_plus, -real(1)/(real(2)*eps), f_minus, finite_difference);

    ks.set_linearization_point(u, lambda);
    ks.jacobian_u(du, jacobian_du);

    const auto fd_host = host_vector(vec_ops, finite_difference, vec_ops.get_default_size());
    const auto jac_host = host_vector(vec_ops, jacobian_du, vec_ops.get_default_size());
    for(std::size_t i = 0; i < fd_host.size(); ++i)
    {
        check_close(
            jac_host[i],
            fd_host[i],
            tolerance<real>()*(real(1) + common::scalar_math::abs(fd_host[i])),
            "jacobian_u finite difference mode " + std::to_string(i + 1)
        );
    }

    vec_ops.stop_use_vectors(u, du, u_plus, u_minus, f_plus, f_minus, finite_difference, jacobian_du);
    vec_ops.free_vectors(u, du, u_plus, u_minus, f_plus, f_minus, finite_difference, jacobian_du);
}

void test_jacobian_alpha(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec u;
    real_vec f_plus;
    real_vec f_minus;
    real_vec finite_difference;
    real_vec jacobian_alpha;
    vec_ops.init_vectors(u, f_plus, f_minus, finite_difference, jacobian_alpha);
    vec_ops.start_use_vectors(u, f_plus, f_minus, finite_difference, jacobian_alpha);

    real_vec du_unused;
    vec_ops.init_vector(du_unused);
    vec_ops.start_use_vector(du_unused);
    fill_test_vectors(vec_ops, u, du_unused);

    const real lambda = real(3.3);
    const real eps = fd_step<real>();
    ks.F(u, lambda + eps, f_plus);
    ks.F(u, lambda - eps, f_minus);
    vec_ops.assign_mul(real(1)/(real(2)*eps), f_plus, -real(1)/(real(2)*eps), f_minus, finite_difference);

    ks.set_linearization_point(u, lambda);
    ks.jacobian_alpha(jacobian_alpha);

    const auto fd_host = host_vector(vec_ops, finite_difference, vec_ops.get_default_size());
    const auto jac_host = host_vector(vec_ops, jacobian_alpha, vec_ops.get_default_size());
    for(std::size_t i = 0; i < fd_host.size(); ++i)
    {
        check_close(
            jac_host[i],
            fd_host[i],
            tolerance<real>()*(real(1) + common::scalar_math::abs(fd_host[i])),
            "jacobian_alpha finite difference mode " + std::to_string(i + 1)
        );
    }

    vec_ops.stop_use_vector(du_unused);
    vec_ops.free_vector(du_unused);
    vec_ops.stop_use_vectors(u, f_plus, f_minus, finite_difference, jacobian_alpha);
    vec_ops.free_vectors(u, f_plus, f_minus, finite_difference, jacobian_alpha);
}

void test_preconditioner_at_zero(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec zero;
    real_vec rhs;
    vec_ops.init_vector(zero);
    vec_ops.start_use_vector(zero);
    vec_ops.init_vector(rhs);
    vec_ops.start_use_vector(rhs);

    vec_ops.assign_scalar(real(0), zero);
    set_mode(vec_ops, rhs, 4, real(2.5));
    const real lambda = real(3.25);
    ks.set_linearization_point(zero, lambda);
    ks.preconditioner_jacobian_u(rhs);

    const auto host_rhs = host_vector(vec_ops, rhs, vec_ops.get_default_size());
    for(std::size_t i = 0; i < host_rhs.size(); ++i)
    {
        const real expected = (i == 3) ? real(2.5)/ks.linear_multiplier(i + 1, lambda) : real(0);
        check_close(
            host_rhs[i],
            expected,
            tolerance<real>()*(real(1) + common::scalar_math::abs(expected)),
            "preconditioner mode " + std::to_string(i + 1)
        );
    }

    vec_ops.stop_use_vector(rhs);
    vec_ops.free_vector(rhs);
    vec_ops.stop_use_vector(zero);
    vec_ops.free_vector(zero);
}

void test_preconditioner_bypasses_exact_diagonal_pole(
    vec_ops_real& vec_ops,
    ks1d_t& ks)
{
    real_vec zero;
    real_vec rhs;
    vec_ops.init_vector(zero);
    vec_ops.start_use_vector(zero);
    vec_ops.init_vector(rhs);
    vec_ops.start_use_vector(rhs);

    vec_ops.assign_scalar(real(0), zero);
    set_mode(vec_ops, rhs, 5, real(2.5));
    ks.set_linearization_point(zero, real(100));
    ks.preconditioner_jacobian_u(rhs);

    const auto host_rhs =
        host_vector(vec_ops, rhs, vec_ops.get_default_size());
    check_close(
        host_rhs[4],
        real(2.5),
        tolerance<real>(),
        "preconditioner keeps component at an exact diagonal pole");

    vec_ops.stop_use_vector(rhs);
    vec_ops.free_vector(rhs);
    vec_ops.stop_use_vector(zero);
    vec_ops.free_vector(zero);
}

void test_affine_preconditioner_at_zero(
    vec_ops_real& vec_ops,
    ks1d_t& ks)
{
    using provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                vec_ops_real,
                ks1d_t>;

    real_vec zero;
    real_vec right_hand_side;
    real_vec solution;
    vec_ops.init_vectors(
        zero,
        right_hand_side,
        solution);
    vec_ops.start_use_vectors(
        zero,
        right_hand_side,
        solution);

    constexpr std::size_t mode = 4;
    const real lambda = real(3.25);
    const real jacobian_scale = real(0.075);
    const real identity_shift = real(1.2);
    const real right_hand_side_value = real(2.5);

    vec_ops.assign_scalar(real(0), zero);
    set_mode(
        vec_ops,
        right_hand_side,
        mode,
        right_hand_side_value);
    ks.set_linearization_point(zero, lambda);

    provider_type provider(vec_ops, ks);
    check_condition(
        provider.apply(
            jacobian_scale,
            identity_shift,
            right_hand_side,
            solution),
        "real affine preconditioner provider succeeds");
    check_condition(
        provider.apply_calls() == 1 &&
        provider.failed_applications() == 0,
        "real affine preconditioner provider statistics");

    const auto host_solution = host_vector(
        vec_ops,
        solution,
        vec_ops.get_default_size());
    const real expected =
        right_hand_side_value /
        (
            jacobian_scale *
                ks.linear_multiplier(mode, lambda) +
            identity_shift);
    for(std::size_t index = 0;
        index < host_solution.size();
        ++index)
    {
        const real expected_value =
            index + 1 == mode ? expected : real(0);
        check_close(
            host_solution[index],
            expected_value,
            tolerance<real>() *
                (real(1) +
                 common::scalar_math::abs(expected_value)),
            "real affine preconditioner mode " +
                std::to_string(index + 1));
    }

    vec_ops.stop_use_vectors(
        zero,
        right_hand_side,
        solution);
    vec_ops.free_vectors(
        zero,
        right_hand_side,
        solution);
}

void test_half_period_shift_equivariance(vec_ops_real& vec_ops, ks1d_t& ks)
{
    finite_actions_t finite_actions(&vec_ops);
    ks.configure_finite_symmetry_actions(finite_actions);
    const int action_index = finite_actions.find("sine_half_period_shift");
    check_condition(action_index >= 0, "reduced KS1D registers sine half-period shift action");
    if(action_index < 0)
    {
        return;
    }

    real_vec u;
    real_vec du;
    real_vec shifted_u;
    real_vec shifted_du;
    real_vec f;
    real_vec shifted_f;
    real_vec action_f;
    real_vec jdu;
    real_vec shifted_jdu;
    real_vec action_jdu;
    vec_ops.init_vectors(u, du, shifted_u, shifted_du, f, shifted_f, action_f, jdu, shifted_jdu, action_jdu);
    vec_ops.start_use_vectors(u, du, shifted_u, shifted_du, f, shifted_f, action_f, jdu, shifted_jdu, action_jdu);

    fill_test_vectors(vec_ops, u, du);
    finite_actions.apply(static_cast<std::size_t>(action_index), u, shifted_u);
    finite_actions.apply(static_cast<std::size_t>(action_index), du, shifted_du);

    const real lambda = real(4.25);
    ks.F(u, lambda, f);
    ks.F(shifted_u, lambda, shifted_f);
    finite_actions.apply(static_cast<std::size_t>(action_index), f, action_f);
    check_vector_close(vec_ops, shifted_f, action_f, real(20)*tolerance<real>(), "reduced KS1D half-period F equivariance");

    ks.set_linearization_point(u, lambda);
    ks.jacobian_u(du, jdu);
    ks.set_linearization_point(shifted_u, lambda);
    ks.jacobian_u(shifted_du, shifted_jdu);
    finite_actions.apply(static_cast<std::size_t>(action_index), jdu, action_jdu);
    check_vector_close(vec_ops, shifted_jdu, action_jdu, real(20)*tolerance<real>(), "reduced KS1D half-period J equivariance");

    vec_ops.stop_use_vectors(u, du, shifted_u, shifted_du, f, shifted_f, action_f, jdu, shifted_jdu, action_jdu);
    vec_ops.free_vectors(u, du, shifted_u, shifted_du, f, shifted_f, action_f, jdu, shifted_jdu, action_jdu);
}

} // namespace

int main(int argc, char** argv)
{
#if defined(KS1D_VECTOR_BACKEND_CUDA)
    const std::string selector = argc > 1 ? argv[1] : "auto";
    try
    {
        const int device = common::init_cuda_from_scfd_selector(selector);
        std::cout << "Using CUDA device " << device << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed to initialize CUDA selector '" << selector << "': " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
#else
    (void)argc;
    (void)argv;
#endif

    const std::size_t physical_size = 32;
    const std::size_t reduced_size = physical_size/2 - 1;
    vec_ops_real vec_ops(reduced_size);
    ks1d_t ks(real(2), real(4), physical_size, &vec_ops);

    std::cout << "Testing KS1D operator backend: " << KS1D_BACKEND_NAME << std::endl;
    test_zero_branch(vec_ops, ks);
    test_linear_spectrum(vec_ops, physical_size);
    test_physical_oddness(vec_ops, ks);
    test_jacobian_u(vec_ops, ks);
    test_jacobian_alpha(vec_ops, ks);
    test_preconditioner_at_zero(vec_ops, ks);
    test_preconditioner_bypasses_exact_diagonal_pole(vec_ops, ks);
    test_affine_preconditioner_at_zero(vec_ops, ks);
    test_half_period_shift_equivariance(vec_ops, ks);

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
