#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(KS1D_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#endif

#include <continuation/initial_tangent.h>
#include <continuation/projected_system_operator_continuation.h>
#include <continuation/chart_helpers.h>
#include <symmetry/linearization/projected_linear_operator.h>
#include <symmetry/linearization/projected_preconditioner.h>
#include <symmetry/linearization/projected_stability_linear_operator.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d_full.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <symmetry/linearization/projected_affine_inverse_provider.h>
#include <symmetry/linearization/projected_linearization_provider.h>
#include <symmetry/linearization/projected_stability_gauge.h>
#include <nonlinear_operators/projected_system_operator.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#include <numerical_algos/newton_solvers/newton_solver.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>

#include "KS1D_backend_typedefs.h"

namespace
{

using ks1d_t = nonlinear_operators::kuramoto_sivashinskiy_1d_full<vec_ops_real, fft_backend_t, Blocks_x_>;
using ks1d_reduced_t = nonlinear_operators::kuramoto_sivashinskiy_1d<vec_ops_real, fft_backend_t, Blocks_x_>;
using real_vec = typename vec_ops_real::vector_type;
using symmetry_adapter_t = symmetry::fourier::real_packed_fourier_slice_1d_adapter<vec_ops_real>;
using finite_actions_t = symmetry::finite_action_registry<vec_ops_real>;
using lin_op_t = symmetry::linearization::projected_linear_operator<vec_ops_real, ks1d_t>;
using stability_lin_op_t =
    symmetry::linearization::
        projected_stability_linear_operator<
            vec_ops_real,
            ks1d_t>;
using prec_t = symmetry::linearization::projected_preconditioner<vec_ops_real, ks1d_t, lin_op_t>;
using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>;
using sm_solver_t = numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
    lin_op_t,
    prec_t,
    vec_ops_real,
    monitor_t,
    log_t,
    numerical_algos::lin_solvers::bicgstabl>;
using projected_newton_system_t = nonlinear_operators::projected_system_operator<vec_ops_real, ks1d_t, lin_op_t, sm_solver_t>;
using convergence_strategy_t = nonlinear_operators::newton_method::convergence_strategy<vec_ops_real, ks1d_t, log_t>;
using newton_t = numerical_algos::newton_method::newton_solver<
    vec_ops_real,
    ks1d_t,
    projected_newton_system_t,
    convergence_strategy_t>;
using initial_tangent_t = continuation::initial_tangent<
    vec_ops_real,
    log_t,
    newton_t,
    ks1d_t,
    lin_op_t,
    sm_solver_t>;
using projected_continuation_system_t = continuation::projected_system_operator_continuation<
    vec_ops_real,
    ks1d_t,
    lin_op_t,
    sm_solver_t,
    log_t>;

int checks = 0;
int failures = 0;

template<class T>
T tolerance()
{
    return std::is_same<T, float>::value ? T(1.0e-2) : T(5.0e-8);
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
std::string scientific_string(const T value)
{
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(17) << static_cast<double>(value);
    return stream.str();
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
            " value=" + scientific_string(value) +
            " expected=" + scientific_string(expected) +
            " err=" + scientific_string(err) +
            " tol=" + scientific_string(tol)
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

void check_vector_close(vec_ops_real& vec_ops, const real_vec& value, const real_vec& expected, real tol, const std::string& label)
{
    real_vec diff;
    vec_ops.init_vector(diff);
    vec_ops.start_use_vector(diff);
    vec_ops.assign_mul(real(1), value, real(-1), expected, diff);
    const real err = vec_ops.norm_l2(diff);
    ++checks;
    if(!(err <= tol))
    {
        record_failure(label + " norm err=" + scientific_string(err) +
                       " tol=" + scientific_string(tol));
    }
    vec_ops.stop_use_vector(diff);
    vec_ops.free_vector(diff);
}

void set_mode(vec_ops_real& vec_ops, real_vec& x, std::size_t mode, real real_part, real imag_part)
{
    std::vector<real> host(vec_ops.get_default_size(), real(0));
    if(mode == 0 || 2*mode > host.size())
    {
        throw std::runtime_error("set_mode: mode is outside full KS1D vector.");
    }
    const std::size_t offset = 2*(mode - 1);
    host[offset] = real_part;
    host[offset + 1] = imag_part;
    vec_ops.set(host.data(), x, host.size());
}

std::vector<real> host_vector(vec_ops_real& vec_ops, const real_vec& x)
{
    std::vector<real> host(vec_ops.get_default_size(), real(0));
    vec_ops.get(x, host.data(), host.size());
    return host;
}

void fill_test_vectors(vec_ops_real& vec_ops, real_vec& u, real_vec& du)
{
    std::vector<real> host_u(vec_ops.get_default_size(), real(0));
    std::vector<real> host_du(vec_ops.get_default_size(), real(0));
    const std::size_t mode_count = host_u.size()/2;
    for(std::size_t mode = 1; mode <= mode_count; ++mode)
    {
        const real k = static_cast<real>(mode);
        const std::size_t offset = 2*(mode - 1);
        host_u[offset] = real(0.12)*static_cast<real>((mode%3) + 1)/(k*k);
        host_u[offset + 1] = real(0.09)*static_cast<real>((mode%2 == 0) ? 1 : -1)/(k*(k + real(1)));
        host_du[offset] = real(0.07)*static_cast<real>((mode%2 == 0) ? -1 : 1)/k;
        host_du[offset + 1] = real(0.05)*static_cast<real>((mode%3 == 0) ? -1 : 1)/(k + real(1));
    }
    vec_ops.set(host_u.data(), u, host_u.size());
    vec_ops.set(host_du.data(), du, host_du.size());
}

void fill_reduced_low_mode_vectors(vec_ops_real& vec_ops, real_vec& u, real_vec& du)
{
    std::vector<real> host_u(vec_ops.get_default_size(), real(0));
    std::vector<real> host_du(vec_ops.get_default_size(), real(0));
    for(std::size_t i = 0; i < host_u.size(); ++i)
    {
        if(i >= 3)
        {
            continue;
        }
        const real k = static_cast<real>(i + 1);
        host_u[i] = real(0.15)*static_cast<real>((i%3) + 1)/(k*k);
        host_du[i] = real(0.08)*static_cast<real>((i%2 == 0) ? 1 : -1)/k;
    }
    vec_ops.set(host_u.data(), u, host_u.size());
    vec_ops.set(host_du.data(), du, host_du.size());
}

void reduced_to_full_odd(vec_ops_real& reduced_ops, const real_vec& reduced, vec_ops_real& full_ops, real_vec& full)
{
    std::vector<real> reduced_host(reduced_ops.get_default_size(), real(0));
    std::vector<real> full_host(full_ops.get_default_size(), real(0));
    reduced_ops.get(reduced, reduced_host.data(), reduced_host.size());
    for(std::size_t i = 0; i < reduced_host.size(); ++i)
    {
        const std::size_t offset = 2*i;
        full_host[offset] = real(0);
        full_host[offset + 1] = reduced_host[i];
    }
    full_ops.set(full_host.data(), full, full_host.size());
}

void check_full_odd_matches_reduced(
    vec_ops_real& full_ops,
    vec_ops_real& reduced_ops,
    const real_vec& full_value,
    const real_vec& reduced_value,
    real tol,
    const std::string& label)
{
    const auto full_host = host_vector(full_ops, full_value);
    std::vector<real> reduced_host(reduced_ops.get_default_size(), real(0));
    reduced_ops.get(reduced_value, reduced_host.data(), reduced_host.size());
    for(std::size_t i = 0; i < reduced_host.size(); ++i)
    {
        const std::size_t offset = 2*i;
        check_close(full_host[offset], real(0), tol, label + " real mode " + std::to_string(i + 1));
        check_close(
            full_host[offset + 1],
            reduced_host[i],
            tol*(real(1) + common::scalar_math::abs(reduced_host[i])),
            label + " imag mode " + std::to_string(i + 1));
    }
}

void test_reduced_odd_subspace_consistency(
    vec_ops_real& full_ops,
    vec_ops_real& reduced_ops,
    ks1d_t& full_ks,
    ks1d_reduced_t& reduced_ks)
{
    real_vec reduced_u;
    real_vec reduced_du;
    real_vec reduced_f;
    real_vec reduced_jdu;
    real_vec reduced_jalpha;
    reduced_ops.init_vectors(reduced_u, reduced_du, reduced_f, reduced_jdu, reduced_jalpha);
    reduced_ops.start_use_vectors(reduced_u, reduced_du, reduced_f, reduced_jdu, reduced_jalpha);

    real_vec full_u;
    real_vec full_du;
    real_vec full_f;
    real_vec full_jdu;
    real_vec full_jalpha;
    full_ops.init_vectors(full_u, full_du, full_f, full_jdu, full_jalpha);
    full_ops.start_use_vectors(full_u, full_du, full_f, full_jdu, full_jalpha);

    fill_reduced_low_mode_vectors(reduced_ops, reduced_u, reduced_du);
    reduced_to_full_odd(reduced_ops, reduced_u, full_ops, full_u);
    reduced_to_full_odd(reduced_ops, reduced_du, full_ops, full_du);

    const real lambda = real(4.5);
    reduced_ks.F(reduced_u, lambda, reduced_f);
    full_ks.F(full_u, lambda, full_f);
    check_full_odd_matches_reduced(full_ops, reduced_ops, full_f, reduced_f, real(20)*tolerance<real>(), "full/reduced F odd-subspace");

    reduced_ks.set_linearization_point(reduced_u, lambda);
    full_ks.set_linearization_point(full_u, lambda);
    reduced_ks.jacobian_u(reduced_du, reduced_jdu);
    full_ks.jacobian_u(full_du, full_jdu);
    check_full_odd_matches_reduced(full_ops, reduced_ops, full_jdu, reduced_jdu, real(20)*tolerance<real>(), "full/reduced J odd-subspace");

    reduced_ks.jacobian_alpha(reduced_jalpha);
    full_ks.jacobian_alpha(full_jalpha);
    check_full_odd_matches_reduced(full_ops, reduced_ops, full_jalpha, reduced_jalpha, real(20)*tolerance<real>(), "full/reduced Jalpha odd-subspace");

    full_ops.stop_use_vectors(full_u, full_du, full_f, full_jdu, full_jalpha);
    full_ops.free_vectors(full_u, full_du, full_f, full_jdu, full_jalpha);
    reduced_ops.stop_use_vectors(reduced_u, reduced_du, reduced_f, reduced_jdu, reduced_jalpha);
    reduced_ops.free_vectors(reduced_u, reduced_du, reduced_f, reduced_jdu, reduced_jalpha);
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
    check_close(vec_ops.norm(f), real(0), tolerance<real>(), "full zero branch residual");

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
    const real re = real(0.375);
    const real im = real(-0.625);
    set_mode(vec_ops, x, mode, re, im);
    ks_linear.F(x, lambda, f);

    const auto host_f = host_vector(vec_ops, f);
    for(std::size_t k = 1; k <= host_f.size()/2; ++k)
    {
        const std::size_t offset = 2*(k - 1);
        const real expected_re = (k == mode) ? ks_linear.linear_multiplier(mode, lambda)*re : real(0);
        const real expected_im = (k == mode) ? ks_linear.linear_multiplier(mode, lambda)*im : real(0);
        check_close(host_f[offset], expected_re, tolerance<real>()*(real(1) + common::scalar_math::abs(expected_re)), "full linear real mode " + std::to_string(k));
        check_close(host_f[offset + 1], expected_im, tolerance<real>()*(real(1) + common::scalar_math::abs(expected_im)), "full linear imag mode " + std::to_string(k));
    }

    vec_ops.stop_use_vector(f);
    vec_ops.free_vector(f);
    vec_ops.stop_use_vector(x);
    vec_ops.free_vector(x);
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

    check_vector_close(vec_ops, jacobian_du, finite_difference, tolerance<real>()*(real(1) + vec_ops.norm_l2(finite_difference)), "full jacobian_u finite difference");

    vec_ops.stop_use_vectors(u, du, u_plus, u_minus, f_plus, f_minus, finite_difference, jacobian_du);
    vec_ops.free_vectors(u, du, u_plus, u_minus, f_plus, f_minus, finite_difference, jacobian_du);
}

void test_jacobian_alpha(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec u;
    real_vec du_unused;
    real_vec f_plus;
    real_vec f_minus;
    real_vec finite_difference;
    real_vec jacobian_alpha;
    vec_ops.init_vectors(u, du_unused, f_plus, f_minus, finite_difference, jacobian_alpha);
    vec_ops.start_use_vectors(u, du_unused, f_plus, f_minus, finite_difference, jacobian_alpha);

    fill_test_vectors(vec_ops, u, du_unused);
    const real lambda = real(3.3);
    const real eps = fd_step<real>();
    ks.F(u, lambda + eps, f_plus);
    ks.F(u, lambda - eps, f_minus);
    vec_ops.assign_mul(real(1)/(real(2)*eps), f_plus, -real(1)/(real(2)*eps), f_minus, finite_difference);

    ks.set_linearization_point(u, lambda);
    ks.jacobian_alpha(jacobian_alpha);

    check_vector_close(vec_ops, jacobian_alpha, finite_difference, tolerance<real>()*(real(1) + vec_ops.norm_l2(finite_difference)), "full jacobian_alpha finite difference");

    vec_ops.stop_use_vectors(u, du_unused, f_plus, f_minus, finite_difference, jacobian_alpha);
    vec_ops.free_vectors(u, du_unused, f_plus, f_minus, finite_difference, jacobian_alpha);
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
    set_mode(vec_ops, rhs, 4, real(2.5), real(-0.75));
    const real lambda = real(3.25);
    ks.set_linearization_point(zero, lambda);
    ks.preconditioner_jacobian_u(rhs);

    const auto host_rhs = host_vector(vec_ops, rhs);
    for(std::size_t k = 1; k <= host_rhs.size()/2; ++k)
    {
        const std::size_t offset = 2*(k - 1);
        const real expected_re = (k == 4) ? real(2.5)/ks.linear_multiplier(k, lambda) : real(0);
        const real expected_im = (k == 4) ? real(-0.75)/ks.linear_multiplier(k, lambda) : real(0);
        check_close(host_rhs[offset], expected_re, tolerance<real>()*(real(1) + common::scalar_math::abs(expected_re)), "full preconditioner real mode " + std::to_string(k));
        check_close(host_rhs[offset + 1], expected_im, tolerance<real>()*(real(1) + common::scalar_math::abs(expected_im)), "full preconditioner imag mode " + std::to_string(k));
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
    set_mode(vec_ops, rhs, 5, real(2.5), real(-0.75));
    ks.set_linearization_point(zero, real(100));
    ks.preconditioner_jacobian_u(rhs);

    const auto host_rhs = host_vector(vec_ops, rhs);
    check_close(
        host_rhs[8],
        real(2.5),
        tolerance<real>(),
        "full preconditioner keeps real component at an exact diagonal pole");
    check_close(
        host_rhs[9],
        real(-0.75),
        tolerance<real>(),
        "full preconditioner keeps imaginary component at an exact diagonal pole");

    vec_ops.stop_use_vector(rhs);
    vec_ops.free_vector(rhs);
    vec_ops.stop_use_vector(zero);
    vec_ops.free_vector(zero);
}

void test_deterministic_deflation_seed_profiles(
    vec_ops_real& vec_ops,
    ks1d_t& ks)
{
    real_vec first;
    real_vec repeated;
    real_vec different;
    real_vec difference;
    vec_ops.init_vectors(
        first,
        repeated,
        different,
        difference);
    vec_ops.start_use_vectors(
        first,
        repeated,
        different,
        difference);

    ks.randomize_vector(first, real(24), std::uint64_t(9));
    ks.randomize_vector(repeated, real(24), std::uint64_t(9));
    ks.randomize_vector(different, real(24), std::uint64_t(10));
    check_vector_close(
        vec_ops,
        first,
        repeated,
        tolerance<real>(),
        "deterministic deflation seed is reproducible");
    vec_ops.assign_mul(
        real(1),
        first,
        real(-1),
        different,
        difference);
    check_condition(
        vec_ops.norm_l2(difference) > real(1.0e-8),
        "different deflation seed IDs generate distinct profiles");

    vec_ops.stop_use_vectors(
        first,
        repeated,
        different,
        difference);
    vec_ops.free_vectors(
        first,
        repeated,
        different,
        difference);
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
    const std::size_t offset = 2*(mode - 1);
    const real lambda = real(3.25);
    const real jacobian_scale = real(0.075);
    const real identity_shift = real(1.2);
    std::vector<real> host_right_hand_side(
        vec_ops.get_default_size(),
        real(0));
    host_right_hand_side[offset] = real(2.5);
    host_right_hand_side[offset + 1] = real(-0.75);

    vec_ops.assign_scalar(real(0), zero);
    vec_ops.set(
        host_right_hand_side.data(),
        right_hand_side,
        host_right_hand_side.size());
    ks.set_linearization_point(zero, lambda);

    provider_type provider(vec_ops, ks);
    check_condition(
        provider.apply(
            jacobian_scale,
            identity_shift,
            right_hand_side,
            solution),
        "full real affine preconditioner provider succeeds");
    check_condition(
        provider.apply_calls() == 1 &&
        provider.failed_applications() == 0,
        "full real affine preconditioner provider statistics");

    const auto host_solution =
        host_vector(vec_ops, solution);
    const real denominator =
        jacobian_scale*ks.linear_multiplier(mode, lambda) +
        identity_shift;
    for(std::size_t index = 0; index < host_solution.size(); ++index)
    {
        real expected = real(0);
        if(index == offset || index == offset + 1)
        {
            expected = host_right_hand_side[index]/denominator;
        }
        check_close(
            host_solution[index],
            expected,
            tolerance<real>() *
                (real(1) +
                 common::scalar_math::abs(expected)),
            "full real affine preconditioner component " +
                std::to_string(index));
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

void test_projected_affine_preconditioner_gauge(
    vec_ops_real& vec_ops,
    ks1d_t& ks)
{
    using provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                vec_ops_real,
                ks1d_t>;
    using projected_provider_type =
        symmetry::linearization::
            projected_affine_inverse_provider<
                vec_ops_real,
                ks1d_t,
                provider_type>;

    real_vec state;
    real_vec direction;
    real_vec projected;
    real_vec gauge;
    real_vec solution;
    real_vec expected;
    vec_ops.init_vectors(
        state,
        direction,
        projected,
        gauge,
        solution,
        expected);
    vec_ops.start_use_vectors(
        state,
        direction,
        projected,
        gauge,
        solution,
        expected);

    fill_test_vectors(vec_ops, state, direction);
    ks.set_projected_linearization_point(state, real(3.25));
    ks.project_current_tangent(direction, projected);
    vec_ops.assign_lin_comb(
        real(1),
        direction,
        real(-1),
        projected,
        gauge);

    const real jacobian_scale = real(0.075);
    const real identity_shift = real(1.2);
    const real gauge_factor =
        jacobian_scale + identity_shift;
    vec_ops.assign_lin_comb(
        real(1)/gauge_factor,
        gauge,
        expected);

    auto provider =
        std::make_shared<provider_type>(vec_ops, ks);
    projected_provider_type projected_provider(
        vec_ops,
        ks,
        provider);
    check_condition(
        projected_provider.apply(
            jacobian_scale,
            identity_shift,
            gauge,
            solution),
        "projected real affine gauge solve succeeds");
    const real projected_tolerance =
        real(200)*tolerance<real>();
    check_vector_close(
        vec_ops,
        solution,
        expected,
        projected_tolerance *
            (real(1) + vec_ops.norm_l2(expected)),
        "projected real affine gauge inverse");

    ks.project_current_tangent(solution, projected);
    check_close(
        vec_ops.norm_l2(projected),
        real(0),
        projected_tolerance *
            (real(1) + vec_ops.norm_l2(solution)),
        "projected real affine gauge remains gauge");
    check_condition(
        projected_provider.apply_calls() == 1 &&
        projected_provider.failed_applications() == 0 &&
        provider->apply_calls() == 1,
        "projected real affine provider statistics");
    check_condition(
        !projected_provider.apply(
            real(1),
            real(-1),
            gauge,
            solution),
        "projected real affine rejects singular gauge factor");
    check_condition(
        projected_provider.apply_calls() == 2 &&
        projected_provider.failed_applications() == 1 &&
        provider->apply_calls() == 1,
        "projected real affine failure statistics");

    auto quotient_provider =
        std::make_shared<provider_type>(vec_ops, ks);
    projected_provider_type quotient_affine_provider(
        vec_ops,
        ks,
        quotient_provider,
        real(0));
    vec_ops.assign_lin_comb(
        real(1)/identity_shift,
        gauge,
        expected);
    check_condition(
        quotient_affine_provider.apply(
            jacobian_scale,
            identity_shift,
            gauge,
            solution),
        "quotient affine zero-completion gauge solve succeeds");
    check_vector_close(
        vec_ops,
        solution,
        expected,
        projected_tolerance *
            (real(1) + vec_ops.norm_l2(expected)),
        "quotient affine zero-completion gauge inverse");

    lin_op_t quotient_operator(&ks, real(0));
    quotient_operator.apply(gauge, solution);
    check_close(
        vec_ops.norm_l2(solution),
        real(0),
        projected_tolerance *
            (real(1) + vec_ops.norm_l2(gauge)),
        "quotient stability operator has neutral gauge completion");

    vec_ops.stop_use_vectors(
        state,
        direction,
        projected,
        gauge,
        solution,
        expected);
    vec_ops.free_vectors(
        state,
        direction,
        projected,
        gauge,
        solution,
        expected);
}

void test_projected_stability_gauge()
{
    const auto left_stable =
        symmetry::linearization::
            make_projected_stability_gauge<real>(
                real(-2),
                true,
                real(3));
    check_close(
        left_stable.operator_completion,
        real(1.5),
        tolerance<real>(),
        "left-stable projected gauge operator completion");
    check_close(
        real(-2)*left_stable.operator_completion,
        left_stable.scaled_eigenvalue,
        tolerance<real>(),
        "left-stable projected gauge scaled eigenvalue");
    check_condition(
        left_stable.scaled_eigenvalue < real(0),
        "left-stable projected gauge lies in stable half-plane");

    const auto right_stable =
        symmetry::linearization::
            make_projected_stability_gauge<real>(
                real(0.5),
                false);
    check_close(
        right_stable.operator_completion,
        real(2),
        tolerance<real>(),
        "right-stable projected gauge operator completion");
    check_condition(
        right_stable.scaled_eigenvalue > real(0),
        "right-stable projected gauge lies in stable half-plane");

    bool rejected_zero_scale = false;
    try
    {
        (void)symmetry::linearization::
            make_projected_stability_gauge<real>(
                real(0),
                true);
    }
    catch(const std::invalid_argument&)
    {
        rejected_zero_scale = true;
    }
    check_condition(
        rejected_zero_scale,
        "projected stability gauge rejects zero scale");
}

void test_projected_stability_operator(
    vec_ops_real& vec_ops,
    ks1d_t& ks)
{
    real_vec state;
    real_vec direction;
    real_vec projected_direction;
    real_vec jacobian_image;
    real_vec expected;
    real_vec actual;
    real_vec gauge;
    vec_ops.init_vectors(
        state,
        direction,
        projected_direction,
        jacobian_image,
        expected,
        actual,
        gauge);
    vec_ops.start_use_vectors(
        state,
        direction,
        projected_direction,
        jacobian_image,
        expected,
        actual,
        gauge);

    fill_test_vectors(vec_ops, state, direction);
    ks.set_stateless_projected_linearization_point(
        state,
        real(4.75));
    ks.project_current_tangent(
        direction,
        projected_direction);
    ks.jacobian_u(
        projected_direction,
        jacobian_image);
    ks.project_current_tangent(
        jacobian_image,
        expected);

    stability_lin_op_t quotient_operator(&ks, real(0));
    quotient_operator.apply(direction, actual);
    check_vector_close(
        vec_ops,
        actual,
        expected,
        real(200)*tolerance<real>()*
            (real(1) + vec_ops.norm_l2(expected)),
        "projected stability operator is PJP");

    vec_ops.assign_lin_comb(
        real(1),
        direction,
        real(-1),
        projected_direction,
        gauge);
    vec_ops.add_mul(real(1.5), gauge, expected);
    stability_lin_op_t completed_operator(&ks, real(1.5));
    completed_operator.apply(direction, actual);
    check_vector_close(
        vec_ops,
        actual,
        expected,
        real(200)*tolerance<real>()*
            (real(1) + vec_ops.norm_l2(expected)),
        "projected stability operator gauge completion");

    vec_ops.stop_use_vectors(
        state,
        direction,
        projected_direction,
        jacobian_image,
        expected,
        actual,
        gauge);
    vec_ops.free_vectors(
        state,
        direction,
        projected_direction,
        jacobian_image,
        expected,
        actual,
        gauge);
}

void test_projected_linearization_provider(
    vec_ops_real& vec_ops,
    ks1d_t& ks)
{
    using provider_type =
        symmetry::linearization::
            projected_linearization_provider<
                vec_ops_real,
                ks1d_t>;

    real_vec state;
    real_vec direction;
    real_vec provider_result;
    real_vec direct_result;
    vec_ops.init_vectors(
        state,
        direction,
        provider_result,
        direct_result);
    vec_ops.start_use_vectors(
        state,
        direction,
        provider_result,
        direct_result);

    fill_test_vectors(vec_ops, state, direction);
    const real lambda = real(3.25);
    provider_type provider(ks);
    provider.set_linearization_point(state, lambda);
    ks.projected_jacobian_u(direction, provider_result);

    ks.set_stateless_projected_linearization_point(
        state,
        lambda);
    ks.projected_jacobian_u(direction, direct_result);
    check_vector_close(
        vec_ops,
        provider_result,
        direct_result,
        real(20)*tolerance<real>()*
            (real(1) + vec_ops.norm_l2(direct_result)),
        "projected stability linearization provider");
    check_condition(
        &provider.nonlinear_operator() == &ks,
        "projected stability provider retains nonlinear operator");

    vec_ops.stop_use_vectors(
        state,
        direction,
        provider_result,
        direct_result);
    vec_ops.free_vectors(
        state,
        direction,
        provider_result,
        direct_result);
}

void test_equivariance(vec_ops_real& vec_ops, ks1d_t& ks, symmetry_adapter_t& symmetry)
{
    real_vec u;
    real_vec du;
    real_vec u_shift;
    real_vec du_shift;
    real_vec f;
    real_vec f_shift;
    real_vec shifted_f;
    real_vec jdu;
    real_vec jdu_shift;
    real_vec shifted_jdu;
    vec_ops.init_vectors(u, du, u_shift, du_shift, f, f_shift, shifted_f, jdu, jdu_shift, shifted_jdu);
    vec_ops.start_use_vectors(u, du, u_shift, du_shift, f, f_shift, shifted_f, jdu, jdu_shift, shifted_jdu);

    fill_test_vectors(vec_ops, u, du);
    const real lambda = real(2.7);
    const real shift = real(0.37);
    symmetry.apply_shift(u, u_shift, shift);
    symmetry.apply_shift(du, du_shift, shift);

    ks.F(u, lambda, f);
    ks.F(u_shift, lambda, f_shift);
    symmetry.apply_shift(f, shifted_f, shift);
    check_vector_close(vec_ops, f_shift, shifted_f, tolerance<real>()*(real(1) + vec_ops.norm_l2(shifted_f)), "full F SO2 equivariance");

    ks.set_linearization_point(u, lambda);
    ks.jacobian_u(du, jdu);
    ks.set_linearization_point(u_shift, lambda);
    ks.jacobian_u(du_shift, jdu_shift);
    symmetry.apply_shift(jdu, shifted_jdu, shift);
    check_vector_close(vec_ops, jdu_shift, shifted_jdu, tolerance<real>()*(real(1) + vec_ops.norm_l2(shifted_jdu)), "full J SO2 equivariance");

    vec_ops.stop_use_vectors(u, du, u_shift, du_shift, f, f_shift, shifted_f, jdu, jdu_shift, shifted_jdu);
    vec_ops.free_vectors(u, du, u_shift, du_shift, f, f_shift, shifted_f, jdu, jdu_shift, shifted_jdu);
}

void test_finite_action_equivariance(vec_ops_real& vec_ops, ks1d_t& ks)
{
    finite_actions_t finite_actions(&vec_ops);
    ks.configure_finite_symmetry_actions(finite_actions);
    const int action_index = finite_actions.find("real_packed_negative_reflection");
    ++checks;
    if(action_index < 0)
    {
        record_failure("full KS1D registers real-packed negative reflection action");
        return;
    }

    real_vec u;
    real_vec du;
    real_vec action_u;
    real_vec action_du;
    real_vec f;
    real_vec action_f_expected;
    real_vec action_f;
    real_vec jdu;
    real_vec action_jdu_expected;
    real_vec action_jdu;
    vec_ops.init_vectors(u, du, action_u, action_du, f, action_f_expected, action_f, jdu, action_jdu_expected, action_jdu);
    vec_ops.start_use_vectors(u, du, action_u, action_du, f, action_f_expected, action_f, jdu, action_jdu_expected, action_jdu);

    fill_test_vectors(vec_ops, u, du);
    finite_actions.apply(static_cast<std::size_t>(action_index), u, action_u);
    finite_actions.apply(static_cast<std::size_t>(action_index), du, action_du);

    const real lambda = real(4.15);
    ks.F(u, lambda, f);
    ks.F(action_u, lambda, action_f);
    finite_actions.apply(static_cast<std::size_t>(action_index), f, action_f_expected);
    check_vector_close(
        vec_ops,
        action_f,
        action_f_expected,
        real(30)*tolerance<real>()*(real(1) + vec_ops.norm_l2(action_f_expected)),
        "full KS1D finite-action F equivariance");

    ks.set_linearization_point(u, lambda);
    ks.jacobian_u(du, jdu);
    ks.set_linearization_point(action_u, lambda);
    ks.jacobian_u(action_du, action_jdu);
    finite_actions.apply(static_cast<std::size_t>(action_index), jdu, action_jdu_expected);
    check_vector_close(
        vec_ops,
        action_jdu,
        action_jdu_expected,
        real(30)*tolerance<real>()*(real(1) + vec_ops.norm_l2(action_jdu_expected)),
        "full KS1D finite-action J equivariance");

    vec_ops.stop_use_vectors(u, du, action_u, action_du, f, action_f_expected, action_f, jdu, action_jdu_expected, action_jdu);
    vec_ops.free_vectors(u, du, action_u, action_du, f, action_f_expected, action_f, jdu, action_jdu_expected, action_jdu);
}

void test_project_hook(vec_ops_real& vec_ops, ks1d_t& ks, symmetry_adapter_t& symmetry)
{
    real_vec u;
    real_vec shifted;
    real_vec projected;
    real_vec expected;
    vec_ops.init_vectors(u, shifted, projected, expected);
    vec_ops.start_use_vectors(u, shifted, projected, expected);

    real_vec du_unused;
    vec_ops.init_vector(du_unused);
    vec_ops.start_use_vector(du_unused);
    fill_test_vectors(vec_ops, u, du_unused);
    symmetry.apply_shift(u, shifted, real(0.51));
    vec_ops.assign(shifted, projected);
    ks.project(projected);
    symmetry.stabilize(shifted, expected);
    check_vector_close(vec_ops, projected, expected, tolerance<real>(), "full KS1D project hook stabilizes iterate");

    vec_ops.stop_use_vector(du_unused);
    vec_ops.free_vector(du_unused);
    vec_ops.stop_use_vectors(u, shifted, projected, expected);
    vec_ops.free_vectors(u, shifted, projected, expected);
}

void test_continuation_chart_bridge(vec_ops_real& vec_ops, ks1d_t& ks, symmetry_adapter_t& symmetry)
{
    real_vec u;
    real_vec tangent;
    real_vec predictor;
    real_vec trial;
    real_vec expected;
    real_vec corrector_trial;
    real_vec expected_corrector;
    real_vec arclength_chart;
    real_vec expected_arclength;
    vec_ops.init_vectors(
        u,
        tangent,
        predictor,
        trial,
        expected,
        corrector_trial,
        expected_corrector,
        arclength_chart,
        expected_arclength);
    vec_ops.start_use_vectors(
        u,
        tangent,
        predictor,
        trial,
        expected,
        corrector_trial,
        expected_corrector,
        arclength_chart,
        expected_arclength);

    fill_test_vectors(vec_ops, u, tangent);
    ks.project(u);
    symmetry.apply_shift(u, predictor, real(0.41));

    log_t log;
    log.set_verbosity(0);
    const real lambda0 = real(5.9);
    const real lambda0_s = real(0.2);
    const real lambda_predictor = real(6.1);
    real lambda_trial = real(0);

    continuation::chart::begin_continuation_chart(&vec_ops, &log, &ks, u, lambda0, tangent, lambda0_s);
    continuation::chart::stabilize_predictor_for_continuation(
        &vec_ops,
        &log,
        &ks,
        u,
        lambda0,
        tangent,
        lambda0_s,
        predictor,
        lambda_predictor,
        trial,
        lambda_trial);
    symmetry.stabilize_closest_to_reference(u, predictor, expected);
    check_vector_close(vec_ops, trial, expected, tolerance<real>(), "full KS continuation predictor chart bridge");
    check_close(lambda_trial, lambda_predictor, real(0), "full KS continuation predictor preserves lambda");

    symmetry.apply_shift(u, corrector_trial, real(-0.29));
    vec_ops.assign(corrector_trial, expected_corrector);
    ks.project_relative_to(u, expected_corrector);
    real lambda_corrector = lambda0;
    continuation::chart::stabilize_corrector_trial(&vec_ops, &log, &ks, u, lambda0, corrector_trial, lambda_corrector);
    check_vector_close(
        vec_ops,
        corrector_trial,
        expected_corrector,
        tolerance<real>(),
        "full KS continuation corrector chart bridge");
    check_close(lambda_corrector, lambda0, real(0), "full KS continuation corrector preserves lambda");

    continuation::chart::stabilize_for_arclength(&vec_ops, &ks, u, predictor, arclength_chart);
    symmetry.stabilize_closest_to_reference(u, predictor, expected_arclength);
    check_vector_close(
        vec_ops,
        arclength_chart,
        expected_arclength,
        tolerance<real>(),
        "full KS continuation arclength chart bridge");

    vec_ops.stop_use_vectors(
        u,
        tangent,
        predictor,
        trial,
        expected,
        corrector_trial,
        expected_corrector,
        arclength_chart,
        expected_arclength);
    vec_ops.free_vectors(
        u,
        tangent,
        predictor,
        trial,
        expected,
        corrector_trial,
        expected_corrector,
        arclength_chart,
        expected_arclength);
}

void test_reduced_continuation_chart_fallback(vec_ops_real& vec_ops, ks1d_reduced_t& ks)
{
    real_vec u;
    real_vec tangent;
    real_vec predictor;
    real_vec trial;
    real_vec arclength_chart;
    vec_ops.init_vectors(u, tangent, predictor, trial, arclength_chart);
    vec_ops.start_use_vectors(u, tangent, predictor, trial, arclength_chart);

    fill_reduced_low_mode_vectors(vec_ops, u, tangent);
    std::vector<real> predictor_host(vec_ops.get_default_size(), real(0));
    for(std::size_t i = 0; i < predictor_host.size(); ++i)
    {
        predictor_host[i] = real(0.1)*static_cast<real>(i + 1);
    }
    vec_ops.set(predictor_host.data(), predictor, predictor_host.size());

    log_t log;
    log.set_verbosity(0);
    const real lambda0 = real(5.9);
    const real lambda0_s = real(0.2);
    const real lambda_predictor = real(6.1);
    real lambda_trial = real(0);

    continuation::chart::begin_continuation_chart(&vec_ops, &log, &ks, u, lambda0, tangent, lambda0_s);
    continuation::chart::stabilize_predictor_for_continuation(
        &vec_ops,
        &log,
        &ks,
        u,
        lambda0,
        tangent,
        lambda0_s,
        predictor,
        lambda_predictor,
        trial,
        lambda_trial);
    check_vector_close(vec_ops, trial, predictor, tolerance<real>(), "reduced KS continuation predictor fallback");
    check_close(lambda_trial, lambda_predictor, real(0), "reduced KS continuation predictor fallback lambda");

    continuation::chart::stabilize_corrector_trial(&vec_ops, &log, &ks, u, lambda0, trial, lambda_trial);
    check_vector_close(vec_ops, trial, predictor, tolerance<real>(), "reduced KS continuation corrector fallback");

    continuation::chart::stabilize_for_arclength(&vec_ops, &ks, u, predictor, arclength_chart);
    check_vector_close(vec_ops, arclength_chart, predictor, tolerance<real>(), "reduced KS continuation arclength fallback");

    vec_ops.stop_use_vectors(u, tangent, predictor, trial, arclength_chart);
    vec_ops.free_vectors(u, tangent, predictor, trial, arclength_chart);
}

void test_projected_operator_hooks(vec_ops_real& vec_ops, ks1d_t& ks, symmetry_adapter_t& symmetry)
{
    real_vec u;
    real_vec du;
    real_vec u_shift;
    real_vec projected_residual;
    real_vec shifted_projected_residual;
    real_vec du_projected;
    real_vec du_projected_again;
    real_vec jdu_projected;
    real_vec jdu_projected_again;
    real_vec u_plus;
    real_vec u_minus;
    real_vec projected_plus;
    real_vec projected_minus;
    real_vec projected_finite_difference;
    vec_ops.init_vectors(
        u,
        du,
        u_shift,
        projected_residual,
        shifted_projected_residual,
        du_projected,
        du_projected_again,
        jdu_projected,
        jdu_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference);
    vec_ops.start_use_vectors(
        u,
        du,
        u_shift,
        projected_residual,
        shifted_projected_residual,
        du_projected,
        du_projected_again,
        jdu_projected,
        jdu_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference);

    fill_test_vectors(vec_ops, u, du);
    const real lambda = real(2.85);
    symmetry.apply_shift(u, u_shift, real(0.43));

    ks.projected_F(u, lambda, projected_residual);
    ks.projected_F(u_shift, lambda, shifted_projected_residual);
    check_vector_close(
        vec_ops,
        shifted_projected_residual,
        projected_residual,
        tolerance<real>()*(real(1) + vec_ops.norm_l2(projected_residual)),
        "full projected_F quotient invariance");

    ks.set_projected_linearization_point(u, lambda);
    ks.project_current_tangent(du, du_projected);
    ks.project_current_tangent(du_projected, du_projected_again);
    check_vector_close(
        vec_ops,
        du_projected_again,
        du_projected,
        tolerance<real>()*(real(1) + vec_ops.norm_l2(du_projected)),
        "full tangent projector idempotence");

    ks.projected_jacobian_u(du, jdu_projected);
    ks.project_current_tangent(jdu_projected, jdu_projected_again);
    check_vector_close(
        vec_ops,
        jdu_projected_again,
        jdu_projected,
        tolerance<real>()*(real(1) + vec_ops.norm_l2(jdu_projected)),
        "full projected_jacobian_u output tangent");

    const real eps = fd_step<real>();
    vec_ops.assign_mul(real(1), u, eps, du, u_plus);
    vec_ops.assign_mul(real(1), u, -eps, du, u_minus);
    ks.projected_F(u_plus, lambda, projected_plus);
    ks.projected_F(u_minus, lambda, projected_minus);
    vec_ops.assign_mul(real(1)/(real(2)*eps), projected_plus, -real(1)/(real(2)*eps), projected_minus, projected_finite_difference);
    ks.set_projected_linearization_point(u, lambda);
    ks.projected_jacobian_u(du, jdu_projected);
    check_vector_close(
        vec_ops,
        jdu_projected,
        projected_finite_difference,
        real(400)*tolerance<real>()*(real(1) + vec_ops.norm_l2(projected_finite_difference)),
        "full projected_jacobian_u finite difference");

    vec_ops.stop_use_vectors(
        u,
        du,
        u_shift,
        projected_residual,
        shifted_projected_residual,
        du_projected,
        du_projected_again,
        jdu_projected,
        jdu_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference);
    vec_ops.free_vectors(
        u,
        du,
        u_shift,
        projected_residual,
        shifted_projected_residual,
        du_projected,
        du_projected_again,
        jdu_projected,
        jdu_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference);
}

void test_frozen_lsq_projected_operator_hooks(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec u;
    real_vec du;
    real_vec du_projected;
    real_vec du_projected_again;
    real_vec u_plus;
    real_vec u_minus;
    real_vec projected_plus;
    real_vec projected_minus;
    real_vec projected_finite_difference;
    real_vec projected_jacobian;
    vec_ops.init_vectors(
        u,
        du,
        du_projected,
        du_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference,
        projected_jacobian);
    vec_ops.start_use_vectors(
        u,
        du,
        du_projected,
        du_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference,
        projected_jacobian);

    fill_test_vectors(vec_ops, u, du);
    const real lambda = real(2.85);
    ks.set_projected_linearization_point(u, lambda);
    ks.project_current_tangent(du, du_projected);
    ks.project_current_tangent(du_projected, du_projected_again);
    check_vector_close(
        vec_ops,
        du_projected_again,
        du_projected,
        tolerance<real>()*(real(1) + vec_ops.norm_l2(du_projected)),
        "frozen LSQ tangent projector idempotence");

    const real eps = fd_step<real>();
    vec_ops.assign_mul(real(1), u, eps, du, u_plus);
    vec_ops.assign_mul(real(1), u, -eps, du, u_minus);
    ks.projected_F_in_frozen_chart(u_plus, lambda, projected_plus);
    ks.projected_F_in_frozen_chart(u_minus, lambda, projected_minus);
    vec_ops.assign_mul(
        real(1)/(real(2)*eps),
        projected_plus,
        -real(1)/(real(2)*eps),
        projected_minus,
        projected_finite_difference);
    ks.set_projected_linearization_point(u, lambda);
    ks.projected_jacobian_u(du, projected_jacobian);
    check_vector_close(
        vec_ops,
        projected_jacobian,
        projected_finite_difference,
        real(600)*tolerance<real>()*(real(1) + vec_ops.norm_l2(projected_finite_difference)),
        "frozen LSQ projected Jacobian finite difference");

    vec_ops.stop_use_vectors(
        u,
        du,
        du_projected,
        du_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference,
        projected_jacobian);
    vec_ops.free_vectors(
        u,
        du,
        du_projected,
        du_projected_again,
        u_plus,
        u_minus,
        projected_plus,
        projected_minus,
        projected_finite_difference,
        projected_jacobian);
}

void test_projected_bordered_continuation_correction(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec x0;
    real_vec x0_chart;
    real_vec tangent;
    real_vec tangent_chart;
    real_vec candidate;
    real_vec dx;
    real_vec projected_dx;
    real_vec jlambda;
    real_vec rhs;
    real_vec lhs;
    real_vec linear_residual;
    vec_ops.init_vectors(x0, x0_chart, tangent, tangent_chart, candidate, dx, projected_dx, jlambda, rhs, lhs, linear_residual);
    vec_ops.start_use_vectors(x0, x0_chart, tangent, tangent_chart, candidate, dx, projected_dx, jlambda, rhs, lhs, linear_residual);

    std::vector<real> host_x0(vec_ops.get_default_size(), real(0));
    if(host_x0.size() >= 6)
    {
        host_x0[0] = real(1.10);
        host_x0[1] = real(0.25);
        host_x0[2] = real(0.40);
        host_x0[3] = real(-0.30);
        host_x0[4] = real(0.10);
        host_x0[5] = real(0.05);
    }
    vec_ops.set(host_x0.data(), x0, host_x0.size());
    ks.project(x0);
    host_x0 = host_vector(vec_ops, x0);

    std::vector<real> host_tangent(vec_ops.get_default_size(), real(0));
    const std::size_t mode_count = host_tangent.size()/2;
    for(std::size_t mode = 1; mode <= mode_count; ++mode)
    {
        const real k = static_cast<real>(mode);
        const std::size_t offset = 2*(mode - 1);
        const real re = host_x0[offset];
        const real im = host_x0[offset + 1];
        host_tangent[offset] = real(0.02)/(k + real(1)) - real(2.0)*k*im;
        host_tangent[offset + 1] = -real(0.015)/(k + real(2)) + real(2.0)*k*re;
    }
    vec_ops.set(host_tangent.data(), tangent, host_tangent.size());

    real lambda0 = real(6);
    real lambda_s = real(0.25);
    const real tangent_norm = vec_ops.norm_rank1(tangent, lambda_s);
    vec_ops.scale(real(1)/tangent_norm, tangent);
    lambda_s /= tangent_norm;

    const real ds = real(1.0e-3);
    vec_ops.assign_mul(real(1), x0, ds, tangent, candidate);
    real lambda_candidate = lambda0 + ds*lambda_s;

    log_t log;
    log.set_verbosity(0);
    lin_op_t lin_op(&ks);
    prec_t prec(&ks);
    sm_solver_t sm_solver(&prec, &vec_ops, &log);
    sm_solver.get_linsolver_handle()->monitor().init(
        std::is_same<real, float>::value ? real(1.0e-5) : real(1.0e-10),
        real(1.0e-14),
        2000,
        0,
        false,
        false,
        true);
    sm_solver.get_linsolver_handle()->set_basis_size(16);
    sm_solver.get_linsolver_handle()->set_use_precond_resid(true);
    sm_solver.get_linsolver_handle()->set_resid_recalc_freq(1);

    projected_continuation_system_t system_operator(&vec_ops, &log, &lin_op, &sm_solver);
    real ds_mutable = ds;
    system_operator.set_tangent_space(x0, lambda0, tangent, lambda_s, ds_mutable, 'S', &ks);
    const real beta = -system_operator.arclength_residual(candidate, lambda_candidate);
    real d_lambda = real(0);
    const bool solved = system_operator.solve(&ks, candidate, lambda_candidate, dx, d_lambda);
    ++checks;
    if(!solved)
    {
        record_failure(
            "projected bordered correction linear solver did not converge: residual=" +
            std::to_string(static_cast<double>(
                sm_solver.get_linsolver_handle()->monitor().resid_norm_out())) +
            " iterations=" +
            std::to_string(sm_solver.get_linsolver_handle()->monitor().iters_performed()));
        vec_ops.stop_use_vectors(x0, x0_chart, tangent, tangent_chart, candidate, dx, projected_dx, jlambda, rhs, lhs, linear_residual);
        vec_ops.free_vectors(x0, x0_chart, tangent, tangent_chart, candidate, dx, projected_dx, jlambda, rhs, lhs, linear_residual);
        return;
    }

    ks.stabilize_for_arclength(x0, x0, x0_chart);
    ks.stabilize_tangent_for_arclength(x0_chart, tangent, tangent_chart);
    const real scalar_residual = vec_ops.scalar_prod(tangent_chart, dx) + lambda_s*d_lambda - beta;
    check_close(
        scalar_residual,
        real(0),
        std::is_same<real, float>::value ? real(2.0e-4) : real(1.0e-8),
        "projected bordered correction scalar row");

    ks.project_current_tangent(dx, projected_dx);
    check_vector_close(
        vec_ops,
        projected_dx,
        dx,
        (std::is_same<real, float>::value ? real(2.0e-4) : real(1.0e-5))*(real(1) + vec_ops.norm_l2(dx)),
        "projected bordered correction is tangent");

    ks.set_projected_linearization_point(candidate, lambda_candidate);
    ks.projected_jacobian_alpha(jlambda);
    ks.projected_F_at_linearization(rhs);
    vec_ops.assign_mul(real(-1), rhs, rhs);
    lin_op.apply(dx, lhs);
    vec_ops.add_mul(d_lambda, jlambda, lhs);
    vec_ops.assign_mul(real(1), lhs, real(-1), rhs, linear_residual);
    const real linear_residual_norm = vec_ops.norm_l2(linear_residual);
    ++checks;
    const real linear_tol = std::is_same<real, float>::value ? real(1.0e-2) : real(2.0e-4);
    if(!(linear_residual_norm <= linear_tol*(real(1) + vec_ops.norm_l2(rhs))))
    {
        record_failure(
            "projected bordered correction vector row residual=" +
            std::to_string(static_cast<double>(linear_residual_norm)) +
            " tol=" + std::to_string(static_cast<double>(linear_tol*(real(1) + vec_ops.norm_l2(rhs)))));
    }

    vec_ops.stop_use_vectors(x0, x0_chart, tangent, tangent_chart, candidate, dx, projected_dx, jlambda, rhs, lhs, linear_residual);
    vec_ops.free_vectors(x0, x0_chart, tangent, tangent_chart, candidate, dx, projected_dx, jlambda, rhs, lhs, linear_residual);
}

void test_projected_bordered_initial_tangent_equation(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec x0;
    real_vec row;
    real_vec row_projected;
    real_vec jlambda;
    real_vec rhs_zero;
    real_vec tangent;
    real_vec tangent_projected;
    real_vec lhs;
    vec_ops.init_vectors(x0, row, row_projected, jlambda, rhs_zero, tangent, tangent_projected, lhs);
    vec_ops.start_use_vectors(x0, row, row_projected, jlambda, rhs_zero, tangent, tangent_projected, lhs);

    fill_test_vectors(vec_ops, x0, row);
    ks.project(x0);
    ks.set_projected_linearization_point(x0, real(5.75));
    ks.projected_jacobian_alpha(jlambda);
    ks.project_current_tangent(row, row_projected);

    real row_lambda = real(0.35);
    const real row_norm = vec_ops.norm_rank1(row_projected, row_lambda);
    vec_ops.scale(real(1)/row_norm, row_projected);
    row_lambda /= row_norm;

    log_t log;
    log.set_verbosity(0);
    lin_op_t lin_op(&ks);
    prec_t prec(&ks);
    sm_solver_t sm_solver(&prec, &vec_ops, &log);
    sm_solver.get_linsolver_handle()->monitor().init(
        std::is_same<real, float>::value ? real(1.0e-5) : real(1.0e-10),
        real(1.0e-14),
        2000,
        0,
        false,
        false,
        true);
    sm_solver.get_linsolver_handle()->set_basis_size(16);
    sm_solver.get_linsolver_handle()->set_use_precond_resid(true);
    sm_solver.get_linsolver_handle()->set_resid_recalc_freq(1);

    vec_ops.assign_scalar(real(0), rhs_zero);
    real lambda_s = real(0);
    const real beta = real(1);
    const bool solved = sm_solver.solve(lin_op, row_projected, jlambda, row_lambda, rhs_zero, beta, tangent, lambda_s);
    ++checks;
    if(!solved)
    {
        record_failure("projected bordered initial tangent linear solver did not converge");
    }

    lin_op.apply(tangent, lhs);
    vec_ops.add_mul(lambda_s, jlambda, lhs);
    const real linear_residual_norm = vec_ops.norm_l2(lhs);
    ++checks;
    const real linear_tol = std::is_same<real, float>::value ? real(1.0e-2) : real(2.0e-3);
    if(!(linear_residual_norm <= linear_tol*(real(1) + vec_ops.norm_l2(jlambda))))
    {
        record_failure(
            "projected bordered initial tangent vector row residual=" +
            std::to_string(static_cast<double>(linear_residual_norm)) +
            " tol=" + std::to_string(static_cast<double>(linear_tol*(real(1) + vec_ops.norm_l2(jlambda)))));
    }

    const real row_residual = vec_ops.scalar_prod(row_projected, tangent) + row_lambda*lambda_s - beta;
    check_close(
        row_residual,
        real(0),
        std::is_same<real, float>::value ? real(2.0e-4) : real(1.0e-8),
        "projected bordered initial tangent scalar row");

    ks.project_current_tangent(tangent, tangent_projected);
    check_vector_close(
        vec_ops,
        tangent_projected,
        tangent,
        (std::is_same<real, float>::value ? real(2.0e-4) : real(1.0e-8))*(real(1) + vec_ops.norm_l2(tangent)),
        "projected bordered initial tangent is tangent");

    vec_ops.stop_use_vectors(x0, row, row_projected, jlambda, rhs_zero, tangent, tangent_projected, lhs);
    vec_ops.free_vectors(x0, row, row_projected, jlambda, rhs_zero, tangent, tangent_projected, lhs);
}

void test_initial_tangent_projected_path_smoke(vec_ops_real& vec_ops, ks1d_t& ks)
{
    real_vec x0;
    real_vec tangent;
    vec_ops.init_vectors(x0, tangent);
    vec_ops.start_use_vectors(x0, tangent);
    vec_ops.assign_scalar(real(0), x0);

    log_t log;
    log.set_verbosity(0);
    lin_op_t lin_op(&ks);
    prec_t prec(&ks);
    sm_solver_t sm_solver(&prec, &vec_ops, &log);
    sm_solver.get_linsolver_handle()->monitor().init(
        std::is_same<real, float>::value ? real(1.0e-5) : real(1.0e-10),
        real(1.0e-14),
        1000,
        0,
        false,
        false,
        true);
    sm_solver.get_linsolver_handle()->set_basis_size(16);
    sm_solver.get_linsolver_handle()->set_use_precond_resid(true);
    sm_solver.get_linsolver_handle()->set_resid_recalc_freq(1);
    sm_solver.get_linsolver_handle_original()->monitor().init(
        std::is_same<real, float>::value ? real(1.0e-5) : real(1.0e-10),
        real(1.0e-14),
        1000,
        0,
        false,
        false,
        true);
    sm_solver.get_linsolver_handle_original()->set_basis_size(16);
    sm_solver.get_linsolver_handle_original()->set_use_precond_resid(true);
    sm_solver.get_linsolver_handle_original()->set_resid_recalc_freq(1);

    vec_ops_real* vec_ops_ptr = &vec_ops;
    log_t* log_ptr = &log;
    lin_op_t* lin_op_ptr = &lin_op;
    sm_solver_t* sm_solver_ptr = &sm_solver;
    projected_newton_system_t newton_system(vec_ops_ptr, lin_op_ptr, sm_solver_ptr);
    convergence_strategy_t convergence(vec_ops_ptr, log_ptr, real(1.0e-10), 10, real(1), false, false);
    newton_t newton(&vec_ops, &newton_system, &convergence);
    initial_tangent_t initial_tangent(vec_ops_ptr, &log, &newton, lin_op_ptr, sm_solver_ptr, false);

    ks1d_t* ks_ptr = &ks;
    real lambda_s = real(0);
    const bool solved = initial_tangent.execute(ks_ptr, real(1), x0, real(5.0), tangent, lambda_s);
    ++checks;
    if(!solved)
    {
        record_failure("initial_tangent projected path smoke solve failed");
    }
    check_close(vec_ops.norm_rank1(tangent, lambda_s), real(1), real(20)*tolerance<real>(), "initial tangent projected path normalized tangent");
    check_close(vec_ops.norm_l2(tangent), real(0), real(20)*tolerance<real>(), "initial tangent projected path zero branch spatial tangent");
    check_close(lambda_s, real(1), real(20)*tolerance<real>(), "initial tangent projected path zero branch lambda tangent");

    vec_ops.stop_use_vectors(x0, tangent);
    vec_ops.free_vectors(x0, tangent);
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
    const std::size_t positive_modes = physical_size/2 - 1;
    const std::size_t state_size = 2*positive_modes;
    vec_ops_real vec_ops(state_size);
    vec_ops_real reduced_vec_ops(positive_modes);
    vec_ops.use_high_precision();
    reduced_vec_ops.use_high_precision();
    ks1d_t ks(real(2), real(4), physical_size, &vec_ops);
    ks1d_t lsq_ks(real(2), real(4), physical_size, &vec_ops);
    ks1d_reduced_t reduced_ks(real(2), real(4), physical_size, &reduced_vec_ops);
    symmetry_adapter_t symmetry(&vec_ops, positive_modes);
    auto lsq_policy = ks1d_t::default_continuation_symmetry_policy();
    lsq_policy.stabilizer =
        ::symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::lsq_multimode;
    lsq_policy.lsq.mode_min = 1;
    lsq_policy.lsq.mode_max = std::min<std::size_t>(12, positive_modes);
    lsq_policy.lsq.max_active_modes = 6;
    lsq_policy.lsq.grid_points = 96;
    lsq_policy.lsq.newton_iterations = 8;
    lsq_ks.configure_continuation_symmetry(lsq_policy);

    std::cout << "Testing full Fourier KS1D operator backend: " << KS1D_BACKEND_NAME << std::endl;
    test_zero_branch(vec_ops, ks);
    test_linear_spectrum(vec_ops, physical_size);
    test_reduced_odd_subspace_consistency(vec_ops, reduced_vec_ops, ks, reduced_ks);
    test_jacobian_u(vec_ops, ks);
    test_jacobian_alpha(vec_ops, ks);
    test_preconditioner_at_zero(vec_ops, ks);
    test_preconditioner_bypasses_exact_diagonal_pole(vec_ops, ks);
    test_deterministic_deflation_seed_profiles(vec_ops, ks);
    test_affine_preconditioner_at_zero(vec_ops, ks);
    test_projected_affine_preconditioner_gauge(
        vec_ops,
        ks);
    test_projected_stability_gauge();
    test_projected_stability_operator(vec_ops, ks);
    test_projected_linearization_provider(vec_ops, ks);
    test_equivariance(vec_ops, ks, symmetry);
    test_finite_action_equivariance(vec_ops, ks);
    test_project_hook(vec_ops, ks, symmetry);
    test_continuation_chart_bridge(vec_ops, ks, symmetry);
    test_reduced_continuation_chart_fallback(reduced_vec_ops, reduced_ks);
    test_projected_operator_hooks(vec_ops, ks, symmetry);
    test_projected_bordered_continuation_correction(vec_ops, ks);
    test_frozen_lsq_projected_operator_hooks(vec_ops, lsq_ks);
    test_projected_bordered_initial_tangent_equation(vec_ops, ks);
    test_initial_tangent_projected_path_smoke(vec_ops, ks);

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
