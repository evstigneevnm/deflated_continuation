#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(KS1D_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#endif

#include <common/scfd_backend_ext/complex.h>

#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d_full.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/linear_operator_KS_1D.h>

#include <stability/analysis/matrix_free_stability_configuration.h>
#include <stability/analysis/matrix_free_stability_scan.h>
#include <stability/analysis/spectrum_classifier.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

#include <symmetry/linearization/projected_affine_inverse_provider.h>
#include <symmetry/linearization/projected_linear_operator.h>
#include <symmetry/linearization/projected_stability_gauge.h>

#include "KS1D_backend_typedefs.h"

namespace
{

using backend_type = typename vec_ops_real::backend_type;
using complex_scalar_type =
    common::scfd_backend_ext::complex_t<backend_type, real>;
using complex_space_type =
    scfd_vector_operations<backend_type, complex_scalar_type>;
using vector_type = typename vec_ops_real::vector_type;

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cerr << "FAIL " << message << '\n';
    }
}

stability::analysis::matrix_free_stability_config<real>
scan_config()
{
    stability::analysis::matrix_free_stability_config<real> config;
    config.enabled = true;
    config.linearization_scale = real(-1);
    config.transformation.shifts = {
        {real(0.25), real(0)},
        {real(0.25), real(4)},
        {real(0.25), real(12)},
        {real(0.25), real(24)}};
    config.outer.desired_eigenvalues = 8;
    config.outer.krylov_dimension = 28;
    config.outer.restart_dimension = 14;
    config.outer.maximum_restarts = 60;
    config.outer.absolute_tolerance = real(1.0e-10);
    config.outer.relative_tolerance = real(1.0e-8);
    config.recovery.relative_basis_tolerance = real(1.0e-10);
    config.recovery.absolute_residual_tolerance = real(1.0e-8);
    config.recovery.relative_residual_tolerance = real(1.0e-7);
    config.recovery.minimum_converged_eigenpairs = 1;
    config.inner_solver.basis_size = 40;
    config.inner_solver.batch_size = 10;
    config.inner_solver.preconditioner_side = 'R';
    config.inner_solver.relative_tolerance = real(1.0e-10);
    config.inner_solver.absolute_tolerance = real(1.0e-12);
    config.inner_solver.maximum_iterations = 500;
    config.retry.enabled = true;
    config.retry.maximum_shift_retries = 2;
    config.retry.initial_shift_perturbation = real(0.05);
    config.retry.perturbation_growth = real(2);
    config.retry.preconditioner_pole_absolute_tolerance =
        real(1.0e-10);
    config.retry.preconditioner_pole_relative_tolerance =
        real(1.0e-2);
    config.aggregation.absolute_tolerance = real(1.0e-7);
    config.aggregation.relative_tolerance = real(1.0e-6);
    config.aggregation.minimum_successful_scans = 4;
    config.aggregation.minimum_eigenpairs = 6;
    config.aggregation.require_all_scans = true;
    config.aggregation.probe_count = 2;
    config.aggregation.require_all_probes = true;
    config.aggregation.eigenvector_independence_tolerance =
        real(1.0e-6);
    config.aggregation.eigenvector_orthogonalization_passes = 2;
    return config;
}

void initialize_vector(
    vec_ops_real& vector_space,
    vector_type& vector,
    bool zero)
{
    std::vector<real> host(
        vector_space.get_default_size(),
        real(0));
    if(!zero)
    {
        for(std::size_t index = 0; index < host.size(); ++index)
        {
            const real sign =
                index%2 == 0 ? real(1) : real(-1);
            host[index] =
                sign/(real(1) + static_cast<real>(index));
        }
    }
    vector_space.set(host.data(), vector, host.size());
}

template<class RealOperator, class Provider>
auto execute_scan(
    std::shared_ptr<vec_ops_real> real_space,
    std::shared_ptr<complex_space_type> complex_space,
    const RealOperator& real_operator,
    std::shared_ptr<Provider> provider,
    const vector_type& initial_vector,
    stability::analysis::matrix_free_stability_config<real>
        config = scan_config())
{
    using factorization_types =
        stability::eigensolvers::transformations::
            matrix_free_complex_factorization_types<
                vec_ops_real,
                complex_space_type,
                RealOperator,
                Provider>;
    using factor_operator_type =
        typename factorization_types::factor_operator_type;
    using factor_preconditioner_type =
        typename factorization_types::preconditioner_type;
    using monitor_type =
        nmfd::solvers::monitor_krylov<
            complex_space_type,
            log_t>;
    using solver_type =
        nmfd::solvers::gmres<
            complex_space_type,
            monitor_type,
            log_t,
            factor_operator_type,
            factor_preconditioner_type>;
    using dense_lapack_type =
        nmfd::operations::linalg::
            host_small_dense_lapack<real>;
    using scan_type =
        stability::analysis::matrix_free_stability_scan<
            factorization_types,
            solver_type,
            dense_lapack_type>;

    scan_type scan(
        std::move(real_space),
        std::move(complex_space),
        real_operator,
        std::move(provider),
        stability::analysis::
            make_matrix_free_stability_scans<scan_type>(config),
        stability::analysis::
            make_matrix_free_inner_solver_parameters<
                typename scan_type::inner_parameters_type>(config),
        stability::analysis::
            make_spectrum_scan_aggregation_options(config));
    return scan.execute(initial_vector);
}

void test_reduced_saved_near_pole_state(
    real fourth_derivative_scale)
{
    using nonlinear_operator_type =
        nonlinear_operators::kuramoto_sivashinskiy_1d<
            vec_ops_real,
            fft_backend_t,
            Blocks_x_>;
    using jacobian_operator_type =
        nonlinear_operators::linear_operator_KS_1D<
            vec_ops_real,
            nonlinear_operator_type>;
    using base_provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                vec_ops_real,
                nonlinear_operator_type>;
    using stability_operator_type =
        stability::eigensolvers::transformations::
            scaled_real_operator<
                vec_ops_real,
                jacobian_operator_type>;
    using stability_provider_type =
        stability::eigensolvers::transformations::
            scaled_real_affine_inverse_provider<
                base_provider_type>;

    constexpr std::size_t physical_size = 128;
    constexpr std::size_t mode_count =
        physical_size/2 - 1;
    constexpr real parameter =
        real(4.249286248434916);
    const std::array<real, mode_count> saved_state{{
        -5.1623446395543326e+01,
        3.7448130886486637e+00,
        -1.3461322534123629e-01,
        3.8769722358804161e-03,
        -9.7650598167580980e-05,
        2.2660042548909075e-06,
        -4.9687926413290110e-08,
        1.0453816351005793e-09,
        -2.1310041159797103e-11,
        4.2372232794416452e-13,
        -8.2572203480897414e-15,
        1.5829960302611242e-16,
        -2.9733512407289513e-18,
        -4.9143549459314787e-21,
        -6.9710380535534790e-21,
        -7.1871793381041127e-21,
        -6.5990977668225197e-21,
        2.6448374298620931e-21,
        -3.4165771757795786e-22,
        1.3901274577337648e-21,
        -1.4642021336709514e-21,
        -6.7448782122010483e-21,
        1.5613254761780199e-22,
        -2.4543094028483466e-21,
        1.9400217752081426e-21,
        -5.8455639851781778e-23,
        -4.8598650510671693e-22,
        1.2109763327041692e-21,
        -2.3168745223032719e-22,
        2.1198603838706052e-21,
        2.5366548149391440e-22,
        -3.2229662053807040e-22,
        -2.8729259227249121e-22,
        -4.2776922031335397e-22,
        2.0277451150178739e-22,
        -9.9715023184814554e-23,
        -1.2270461600589296e-22,
        9.7311468249421441e-24,
        -5.1996672906923646e-23,
        6.2364565011278154e-22,
        3.3905400109341878e-23,
        -1.2223672234709135e-22,
        -8.9872372394665657e-24,
        7.0988606970160133e-24,
        -1.9455292862111205e-23,
        1.7566927116848373e-22,
        -2.6019047547886073e-23,
        1.0718123005069427e-22,
        5.2431914378932740e-23,
        5.4883857599929538e-24,
        -9.5281366348349279e-23,
        1.0052759984313780e-22,
        -2.4059461973003911e-23,
        -6.4161354117600690e-23,
        5.9711613952823233e-23,
        -3.7982574300944400e-23,
        -7.3270087015138712e-24,
        2.7305418625529781e-23,
        -1.0852284824323438e-23,
        1.5308220414014807e-23,
        1.8026237558782774e-23,
        9.2162548300183385e-23,
        -4.3173177248888973e-24}};

    auto real_space =
        std::make_shared<vec_ops_real>(mode_count);
    auto complex_space =
        std::make_shared<complex_space_type>(mode_count);
    nonlinear_operator_type nonlinear_operator(
        real(2),
        fourth_derivative_scale,
        physical_size,
        real_space.get());
    jacobian_operator_type jacobian_operator(
        &nonlinear_operator);
    stability_operator_type stability_operator(
        *real_space,
        jacobian_operator,
        real(-1));
    auto base_provider =
        std::make_shared<base_provider_type>(
            *real_space,
            nonlinear_operator);
    auto stability_provider =
        std::make_shared<stability_provider_type>(
            base_provider,
            real(-1));

    vector_type state;
    vector_type initial;
    real_space->init_vectors(state, initial);
    real_space->start_use_vectors(state, initial);
    real_space->set(
        saved_state.data(),
        state,
        saved_state.size());
    initialize_vector(*real_space, initial, false);
    nonlinear_operator.set_linearization_point(
        state,
        parameter);

    auto diagnostic_config = scan_config();
    diagnostic_config.retry.enabled = false;
    diagnostic_config.recovery.absolute_residual_tolerance =
        real(0);
    diagnostic_config.recovery.relative_residual_tolerance =
        real(0);
    const auto diagnostic_result = execute_scan(
        real_space,
        complex_space,
        stability_operator,
        stability_provider,
        initial,
        diagnostic_config);
    require(
        !diagnostic_result.succeeded(),
        "saved near-pole state diagnostic rejection");
    require(
        diagnostic_result.diagnostic.find("min|D|=") !=
            std::string::npos,
        "saved near-pole state reports denominator health");
    require(
        diagnostic_result.diagnostic.find(
            "rejected physical Ritz values") !=
            std::string::npos,
        "saved near-pole state reports rejected Ritz residuals");

    const auto recovered_result = execute_scan(
        real_space,
        complex_space,
        stability_operator,
        stability_provider,
        initial);
    require(
        recovered_result.succeeded() &&
            recovered_result.coverage_complete,
        "saved near-pole state retry recovery");
    require(
        recovered_result.diagnostic.find(
            "recovered by retry") !=
            std::string::npos,
        "saved near-pole state reports retry");
    require(
        recovered_result.diagnostic.find(
            "true_residual=") !=
            std::string::npos,
        "saved near-pole state reports true GMRES residual");
    require(
        recovered_result.eigenpairs.size() >= 6,
        "saved near-pole state recovered spectrum size");
    for(const auto& estimate : recovered_result.eigenpairs)
    {
        require(
            estimate.converged &&
                std::isfinite(estimate.relative_residual) &&
                estimate.relative_residual <= real(1.0e-7),
            "saved near-pole state physical Ritz residual");
    }

    real_space->stop_use_vectors(state, initial);
    real_space->free_vectors(state, initial);
}

real physical_eigenvalue(
    std::size_t mode,
    real parameter,
    real fourth_derivative_scale)
{
    const real wave_number = static_cast<real>(mode);
    const real wave_number_sq = wave_number*wave_number;
    return
        parameter*wave_number_sq -
        fourth_derivative_scale*
            wave_number_sq*wave_number_sq;
}

template<class Result>
std::size_t count_eigenvalue(
    const Result& result,
    real expected,
    real tolerance)
{
    return static_cast<std::size_t>(
        std::count_if(
            result.eigenpairs.begin(),
            result.eigenpairs.end(),
            [expected, tolerance](const auto& estimate)
            {
                return
                    std::abs(
                        estimate.value.real() -
                        expected) <= tolerance &&
                    std::abs(estimate.value.imag()) <= tolerance;
            }));
}

template<class Result>
void verify_values(
    const Result& result,
    std::size_t mode_count,
    real parameter,
    real fourth_derivative_scale,
    const std::string& label,
    const real* additional_expected_eigenvalue = nullptr)
{
    require(result.succeeded(), label + " eigensolver status");
    require(result.coverage_complete, label + " scan coverage");
    require(
        result.scans_requested == 4 &&
            result.scans_succeeded == 4,
        label + " scan accounting");
    require(
        result.eigenpairs.size() >= 6,
        label + " recovered spectrum size");

    const real tolerance = real(2.0e-5);
    for(const auto& estimate : result.eigenpairs)
    {
        real best_error = std::numeric_limits<real>::infinity();
        for(std::size_t mode = 1; mode <= mode_count; ++mode)
        {
            best_error = std::min(
                best_error,
                std::abs(
                    estimate.value.real() -
                    physical_eigenvalue(
                        mode,
                        parameter,
                        fourth_derivative_scale)));
        }
        if(additional_expected_eigenvalue != nullptr)
        {
            best_error = std::min(
                best_error,
                std::abs(
                    estimate.value.real() -
                    *additional_expected_eigenvalue));
        }
        require(
            estimate.converged &&
                std::abs(estimate.value.imag()) <= tolerance &&
                best_error <=
                    tolerance*
                        (real(1) +
                         std::abs(estimate.value.real())),
            label + " analytical eigenvalue");
    }
}

template<class Result>
stability::analysis::stability_point_result<real>
classify(Result result)
{
    stability::analysis::spectrum_classifier<real> classifier;
    return classifier.classify(std::move(result));
}

void test_reduced_zero_branch(
    std::size_t physical_size,
    real fourth_derivative_scale)
{
    using nonlinear_operator_type =
        nonlinear_operators::kuramoto_sivashinskiy_1d<
            vec_ops_real,
            fft_backend_t,
            Blocks_x_>;
    using jacobian_operator_type =
        nonlinear_operators::linear_operator_KS_1D<
            vec_ops_real,
            nonlinear_operator_type>;
    using base_provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                vec_ops_real,
                nonlinear_operator_type>;
    using stability_operator_type =
        stability::eigensolvers::transformations::
            scaled_real_operator<
                vec_ops_real,
                jacobian_operator_type>;
    using stability_provider_type =
        stability::eigensolvers::transformations::
            scaled_real_affine_inverse_provider<
                base_provider_type>;

    const std::size_t mode_count = physical_size/2 - 1;
    auto real_space =
        std::make_shared<vec_ops_real>(mode_count);
    auto complex_space =
        std::make_shared<complex_space_type>(mode_count);
    nonlinear_operator_type nonlinear_operator(
        real(2),
        fourth_derivative_scale,
        physical_size,
        real_space.get());
    jacobian_operator_type jacobian_operator(
        &nonlinear_operator);
    stability_operator_type stability_operator(
        *real_space,
        jacobian_operator,
        real(-1));
    auto base_provider =
        std::make_shared<base_provider_type>(
            *real_space,
            nonlinear_operator);
    auto stability_provider =
        std::make_shared<stability_provider_type>(
            base_provider,
            real(-1));

    vector_type zero;
    vector_type initial;
    real_space->init_vectors(zero, initial);
    real_space->start_use_vectors(zero, initial);
    initialize_vector(*real_space, zero, true);
    initialize_vector(*real_space, initial, false);

    for(const real parameter : {
            real(2.5),
            real(4.249286248434916),
            real(5),
            real(80)})
    {
        nonlinear_operator.set_linearization_point(
            zero,
            parameter);
        auto result = execute_scan(
            real_space,
            complex_space,
            stability_operator,
            stability_provider,
            initial);
        verify_values(
            result,
            mode_count,
            parameter,
            fourth_derivative_scale,
            "reduced lambda=" + std::to_string(parameter));
        const auto classified = classify(result);
        require(
            classified.succeeded(),
            "reduced classification");
        require(
            classified.unstable.real ==
                    (
                        parameter == real(80)
                        ? 4
                        : (parameter > real(4) ? 1 : 0)) &&
                classified.unstable.complex_pairs == 0,
            "reduced unstable dimension");
        if(
            std::abs(
                parameter -
                real(4.249286248434916)) <
            real(1.0e-12))
        {
            require(
                result.diagnostic.find(
                    "recovered by retry") !=
                    std::string::npos,
                "reduced near-pole scan uses retry");
        }
        if(parameter == real(80))
        {
            require(
                count_eigenvalue(
                    result,
                    real(256),
                    real(2.0e-5)) >= 2,
                "reduced repeated eigenvalue multiplicity");
        }
    }

    real_space->stop_use_vectors(zero, initial);
    real_space->free_vectors(zero, initial);
}

void test_full_zero_branch(
    std::size_t physical_size,
    real fourth_derivative_scale)
{
    using nonlinear_operator_type =
        nonlinear_operators::kuramoto_sivashinskiy_1d_full<
            vec_ops_real,
            fft_backend_t,
            Blocks_x_>;
    using quotient_operator_type =
        symmetry::linearization::projected_linear_operator<
            vec_ops_real,
            nonlinear_operator_type>;
    using base_provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                vec_ops_real,
                nonlinear_operator_type>;
    using quotient_provider_type =
        symmetry::linearization::projected_affine_inverse_provider<
            vec_ops_real,
            nonlinear_operator_type,
            base_provider_type>;
    using stability_operator_type =
        stability::eigensolvers::transformations::
            scaled_real_operator<
                vec_ops_real,
                quotient_operator_type>;
    using stability_provider_type =
        stability::eigensolvers::transformations::
            scaled_real_affine_inverse_provider<
                quotient_provider_type>;

    const std::size_t mode_count = physical_size/2 - 1;
    const std::size_t state_size = 2*mode_count;
    auto real_space =
        std::make_shared<vec_ops_real>(state_size);
    auto complex_space =
        std::make_shared<complex_space_type>(state_size);
    nonlinear_operator_type nonlinear_operator(
        real(2),
        fourth_derivative_scale,
        physical_size,
        real_space.get());
    const real linearization_scale = real(-1);
    const auto stability_gauge =
        symmetry::linearization::
            make_projected_stability_gauge<real>(
                linearization_scale,
                true);
    quotient_operator_type quotient_operator(
        &nonlinear_operator,
        stability_gauge.operator_completion);
    stability_operator_type stability_operator(
        *real_space,
        quotient_operator,
        linearization_scale);
    auto base_provider =
        std::make_shared<base_provider_type>(
            *real_space,
            nonlinear_operator);
    auto quotient_provider =
        std::make_shared<quotient_provider_type>(
        *real_space,
        nonlinear_operator,
        base_provider,
        stability_gauge.operator_completion);
    auto stability_provider =
        std::make_shared<stability_provider_type>(
            quotient_provider,
            linearization_scale);

    vector_type zero;
    vector_type initial;
    real_space->init_vectors(zero, initial);
    real_space->start_use_vectors(zero, initial);
    initialize_vector(*real_space, zero, true);
    initialize_vector(*real_space, initial, false);

    for(const real parameter : {
            real(2.5),
            real(5),
            real(80)})
    {
        nonlinear_operator.set_linearization_point(
            zero,
            parameter);
        auto result = execute_scan(
            real_space,
            complex_space,
            stability_operator,
            stability_provider,
            initial);
        verify_values(
            result,
            mode_count,
            parameter,
            fourth_derivative_scale,
            "full lambda=" + std::to_string(parameter),
            &stability_gauge.scaled_eigenvalue);
        const auto classified = classify(result);
        require(
            classified.succeeded(),
            "full classification");
        require(
            classified.unstable.real ==
                    (
                        parameter == real(80)
                        ? 8
                        : (parameter > real(4) ? 2 : 0)) &&
                classified.unstable.complex_pairs == 0,
            "full unstable dimension preserves multiplicity: real=" +
                std::to_string(classified.unstable.real) +
                ", complex_pairs=" +
                std::to_string(
                    classified.unstable.complex_pairs));
        if(parameter == real(80))
        {
            require(
                count_eigenvalue(
                    result,
                    real(256),
                    real(2.0e-5)) >= 4,
                "full repeated eigenvalue multiplicity");
        }
    }

    real_space->stop_use_vectors(zero, initial);
    real_space->free_vectors(zero, initial);
}

}

int main(int argc, char** argv)
{
    try
    {
#if defined(KS1D_VECTOR_BACKEND_CUDA)
        const std::string selector =
            argc > 1 ? argv[1] : "auto";
        common::init_cuda_from_scfd_selector(selector);
#else
        (void)argc;
        (void)argv;
#endif
        constexpr std::size_t physical_size = 32;
        constexpr std::size_t production_physical_size = 128;
        constexpr real fourth_derivative_scale = real(4);
        test_reduced_zero_branch(
            physical_size,
            fourth_derivative_scale);
        test_full_zero_branch(
            physical_size,
            fourth_derivative_scale);
        test_full_zero_branch(
            production_physical_size,
            fourth_derivative_scale);
        test_reduced_saved_near_pole_state(
            fourth_derivative_scale);
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout
        << "KS1D stability scan checks: " << checks
        << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
