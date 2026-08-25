#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <stability/analysis/matrix_free_stability_configuration.h>

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

template<class Callable>
void require_throws(Callable&& callable, const std::string& message)
{
    bool threw = false;
    try
    {
        callable();
    }
    catch(const std::invalid_argument&)
    {
        threw = true;
    }
    require(threw, message);
}

void test_factor_builders()
{
    using namespace stability::analysis;
    using transformation =
        matrix_free_spectral_transformation;
    const std::complex<double> shift(0.25, 1.5);

    const auto direct = make_matrix_free_stability_factors(
        transformation::complex_shift_invert,
        0.1,
        3,
        shift);
    require(direct.size() == 1, "direct shift factor count");
    require(
        direct.front().operator_scale ==
            std::complex<double>(1.0, 0.0) &&
        direct.front().diagonal_shift == -shift,
        "direct shift factor coefficients");

    const auto euler = make_matrix_free_stability_factors(
        transformation::explicit_euler,
        0.1,
        3,
        shift);
    require(euler.size() == 3, "Euler factor count");

    const auto rk4 = make_matrix_free_stability_factors(
        transformation::classical_rk4,
        0.1,
        2,
        shift);
    require(rk4.size() == 8, "RK4 factor count");
}

void test_validation()
{
    using config_type =
        stability::analysis::matrix_free_stability_config<double>;
    config_type config;
    config.enabled = true;
    config.transformation.shifts = {
        {0.0, 0.5},
        {0.0, 2.0}};
    stability::analysis::
        validate_matrix_free_stability_config(config);
    require(true, "valid configuration accepted");

    auto invalid = config;
    invalid.outer.krylov_dimension =
        invalid.outer.desired_eigenvalues + 1;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "invalid outer dimensions rejected");

    invalid = config;
    invalid.inner_solver.preconditioner_side = 'X';
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "invalid preconditioner side rejected");

    invalid = config;
    invalid.inner_solver.basis_retry_sizes = {
        invalid.inner_solver.basis_size};
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "inner basis retry must exceed the primary basis");

    invalid = config;
    invalid.inner_solver.basis_retry_sizes = {64, 48};
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "inner basis retries must be strictly increasing");

    invalid = config;
    invalid.aggregation.minimum_successful_scans = 3;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "impossible aggregation coverage rejected");

    invalid = config;
    invalid.aggregation.minimum_eigenpairs = 0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "zero aggregate spectrum requirement rejected");

    invalid = config;
    invalid.aggregation.probe_count = 0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "zero multiplicity probe count rejected");

    invalid = config;
    invalid.aggregation.probe_count = 2;
    invalid.aggregation.minimum_successful_probes = 3;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "impossible multiplicity probe coverage rejected");

    invalid = config;
    invalid.recycling.enabled = true;
    invalid.recycling.maximum_vectors = 0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "zero recycled-subspace capacity rejected");

    invalid = config;
    invalid.recycling.innovation_weight = 1.5;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "invalid recycled-subspace innovation rejected");

    invalid = config;
    invalid.invariant_subspace_tracking.enabled = true;
    invalid.invariant_subspace_tracking.maximum_seed_vectors = 0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "zero tracked-subspace seed budget rejected");

    invalid = config;
    invalid.invariant_subspace_tracking.maximum_dimension = 4;
    invalid.invariant_subspace_tracking.maximum_seed_vectors = 5;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "tracked-subspace seed budget above capacity rejected");

    invalid = config;
    invalid.invariant_subspace_tracking.maximum_dimension = 4;
    invalid.invariant_subspace_tracking.
        coverage_recovery_maximum_seed_vectors = 5;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "tracked-subspace recovery budget above capacity rejected");

    invalid = config;
    invalid.invariant_subspace_tracking.seed_innovation_weight = 1.5;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "invalid tracked-subspace innovation rejected");

    invalid = config;
    invalid.invariant_subspace_tracking.
        minimum_retained_residual_ratio = 1.0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "invalid tracked-subspace retention threshold rejected");

    invalid = config;
    invalid.invariant_subspace_tracking.eigenvalue_group_tolerance = 0.0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "zero tracked-subspace eigenvalue group tolerance rejected");

    config.invariant_subspace_tracking.enabled = true;
    config.invariant_subspace_tracking.maximum_dimension = 13;
    config.invariant_subspace_tracking.maximum_seed_vectors = 5;
    config.invariant_subspace_tracking.
        coverage_recovery_maximum_seed_vectors = 11;
    config.invariant_subspace_tracking.
        minimum_retained_residual_ratio = 0.075;
    config.invariant_subspace_tracking.eigenvalue_group_tolerance = 2.5e-6;
    const auto tracking_options =
        stability::analysis::make_tracked_invariant_subspace_options(
            config);
    require(
        tracking_options.enabled &&
            tracking_options.maximum_dimension == 13 &&
            tracking_options.maximum_seed_vectors == 5 &&
            tracking_options.coverage_recovery_maximum_seed_vectors == 11 &&
            std::abs(
                tracking_options.minimum_retained_residual_ratio -
                0.075) < 1.0e-14 &&
            std::abs(
                tracking_options.eigenvalue_group_tolerance -
                2.5e-6) < 1.0e-14,
        "tracked-subspace configuration maps to runtime options");

    invalid = config;
    invalid.transformation.shifts.clear();
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "empty shift scan rejected");

    invalid = config;
    invalid.retry.enabled = true;
    invalid.retry.maximum_shift_retries = 0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "enabled retry requires an attempt");

    invalid = config;
    invalid.retry.enabled = true;
    invalid.retry.maximum_shift_retries = 2;
    invalid.retry.initial_shift_perturbation = 0.0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "retry requires a positive shift perturbation");

    invalid = config;
    invalid.retry.preconditioner_pole_relative_tolerance = -1.0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "negative preconditioner pole tolerance rejected");

    invalid = config;
    invalid.small_system.enabled = true;
    invalid.small_system.maximum_dimension = 0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "small-system recovery requires a dimension bound");

    invalid = config;
    invalid.small_system.relative_residual_tolerance = -1.0;
    require_throws(
        [&invalid]
        {
            stability::analysis::
                validate_matrix_free_stability_config(invalid);
        },
        "small-system recovery rejects negative tolerance");
}

struct reuse_configuration_target
{
    using recycling_options_type =
        stability::analysis::recycled_ritz_subspace_options<double>;
    using tracking_options_type =
        stability::tracking::
            tracked_invariant_subspace_options<double>;

    void set_recycling_options(recycling_options_type value)
    {
        recycling = value;
        recycling_configured = true;
    }

    void set_tracking_options(tracking_options_type value)
    {
        tracking = value;
        tracking_configured = true;
    }

    recycling_options_type recycling;
    tracking_options_type tracking;
    bool recycling_configured = false;
    bool tracking_configured = false;
};

void test_reuse_configuration()
{
    stability::analysis::matrix_free_stability_config<double> config;
    config.enabled = true;
    config.transformation.shifts = {{0.25, 0.0}};
    config.recycling.enabled = true;
    config.recycling.maximum_vectors = 7;
    config.invariant_subspace_tracking.enabled = true;
    config.invariant_subspace_tracking.maximum_dimension = 9;
    config.invariant_subspace_tracking.maximum_seed_vectors = 4;

    reuse_configuration_target target;
    stability::analysis::configure_matrix_free_stability_reuse(
        target,
        config);

    require(
        target.recycling_configured &&
            target.recycling.enabled &&
            target.recycling.maximum_vectors == 7,
        "shared recycling configuration applied");
    require(
        target.tracking_configured &&
            target.tracking.enabled &&
            target.tracking.maximum_dimension == 9 &&
            target.tracking.maximum_seed_vectors == 4,
        "shared invariant-subspace configuration applied");
}

} // namespace

int main()
{
    test_factor_builders();
    test_validation();
    test_reuse_configuration();
    std::cout
        << "Matrix-free stability configuration checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
