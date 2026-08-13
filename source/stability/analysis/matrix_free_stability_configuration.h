#ifndef __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_CONFIGURATION_H__
#define __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_CONFIGURATION_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nmfd/solvers/krylov/orthogonalization.h>

#include <stability/eigensolvers/eigenvalue_target.h>
#include <stability/eigensolvers/transformations/complex_affine_factor.h>
#include <stability/eigensolvers/transformations/stability_polynomial_factorization.h>

#include "matrix_free_stability_config.h"
#include "recycled_ritz_subspace.h"
#include "spectrum_scan_aggregator.h"

namespace stability
{
namespace analysis
{

template<class Real>
void validate_matrix_free_stability_config(
    const matrix_free_stability_config<Real>& config,
    bool require_enabled = true)
{
    const auto finite = [](Real value)
    {
        using std::isfinite;
        return isfinite(value);
    };
    const auto finite_complex = [&finite](
        const std::complex<Real>& value)
    {
        return finite(value.real()) && finite(value.imag());
    };

    if(require_enabled && !config.enabled)
    {
        throw std::invalid_argument(
            "matrix-free stability eigensolver is disabled");
    }
    if(
        !finite(config.linearization_scale) ||
        config.linearization_scale == Real{})
    {
        throw std::invalid_argument(
            "matrix-free stability linearization scale must be finite "
            "and nonzero");
    }
    if(config.transformation.shifts.empty())
    {
        throw std::invalid_argument(
            "matrix-free stability scan requires at least one shift");
    }
    for(const auto& shift : config.transformation.shifts)
    {
        if(!finite_complex(shift))
        {
            throw std::invalid_argument(
                "matrix-free stability shifts must be finite");
        }
    }
    if(
        config.transformation.type !=
            matrix_free_spectral_transformation::
                complex_shift_invert &&
        (!finite(config.transformation.step) ||
         config.transformation.step == Real{} ||
         config.transformation.repetitions == 0))
    {
        throw std::invalid_argument(
            "polynomial stability transforms require a finite nonzero "
            "step and positive repetitions");
    }

    const auto& outer = config.outer;
    if(
        outer.desired_eigenvalues == 0 ||
        outer.krylov_dimension <
            outer.desired_eigenvalues + 2 ||
        (outer.restart_dimension != 0 &&
         (outer.restart_dimension <
              outer.desired_eigenvalues ||
          outer.restart_dimension >
              outer.krylov_dimension - 2)) ||
        !finite(outer.absolute_tolerance) ||
        outer.absolute_tolerance < Real{} ||
        !finite(outer.relative_tolerance) ||
        outer.relative_tolerance < Real{} ||
        !finite(outer.dgks_eta) ||
        !(outer.dgks_eta > Real{}) ||
        !(outer.dgks_eta < Real(1)) ||
        !finite(outer.breakdown_absolute_tolerance) ||
        outer.breakdown_absolute_tolerance < Real{} ||
        !finite(outer.breakdown_relative_tolerance) ||
        outer.breakdown_relative_tolerance < Real{} ||
        outer.maximum_orthogonalization_passes == 0)
    {
        throw std::invalid_argument(
            "invalid matrix-free outer Krylov-Schur configuration");
    }
    (void)nmfd::solvers::krylov::
        parse_orthogonalization_method(
            outer.orthogonalization);
    (void)nmfd::solvers::krylov::
        parse_reorthogonalization_policy(
            outer.reorthogonalization);

    const auto& recovery = config.recovery;
    if(
        !finite(recovery.relative_basis_tolerance) ||
        !(recovery.relative_basis_tolerance > Real{}) ||
        recovery.orthogonalization_passes == 0 ||
        !finite(recovery.absolute_residual_tolerance) ||
        recovery.absolute_residual_tolerance < Real{} ||
        !finite(recovery.relative_residual_tolerance) ||
        recovery.relative_residual_tolerance < Real{} ||
        recovery.minimum_converged_eigenpairs == 0)
    {
        throw std::invalid_argument(
            "invalid matrix-free physical-spectrum recovery "
            "configuration");
    }

    const auto& inner = config.inner_solver;
    if(
        inner.basis_size == 0 ||
        inner.batch_size == 0 ||
        (inner.preconditioner_side != 'L' &&
         inner.preconditioner_side != 'R') ||
        !finite(inner.dgks_eta) ||
        !(inner.dgks_eta > Real{}) ||
        !(inner.dgks_eta < Real(1)) ||
        !finite(inner.breakdown_relative_tolerance) ||
        inner.breakdown_relative_tolerance < Real{} ||
        inner.maximum_orthogonalization_passes == 0 ||
        !finite(inner.relative_tolerance) ||
        inner.relative_tolerance < Real{} ||
        !finite(inner.absolute_tolerance) ||
        inner.absolute_tolerance < Real{} ||
        inner.maximum_iterations <= 0 ||
        inner.minimum_iterations < 0 ||
        inner.minimum_iterations > inner.maximum_iterations)
    {
        throw std::invalid_argument(
            "invalid matrix-free inner GMRES configuration");
    }
    unsigned int previous_basis_size = inner.basis_size;
    for(const auto basis_size : inner.basis_retry_sizes)
    {
        if(basis_size <= previous_basis_size)
        {
            throw std::invalid_argument(
                "matrix-free inner GMRES basis retries must be "
                "strictly increasing");
        }
        previous_basis_size = basis_size;
    }
    (void)nmfd::solvers::krylov::
        parse_orthogonalization_method(
            inner.orthogonalization);
    (void)nmfd::solvers::krylov::
        parse_reorthogonalization_policy(
            inner.reorthogonalization);

    const auto& retry = config.retry;
    if(
        (retry.enabled &&
         (retry.maximum_shift_retries == 0 ||
          !finite(retry.initial_shift_perturbation) ||
          !(retry.initial_shift_perturbation > Real{}) ||
          !finite(retry.perturbation_growth) ||
          retry.perturbation_growth < Real(1))) ||
        !finite(
            retry.preconditioner_pole_absolute_tolerance) ||
        retry.preconditioner_pole_absolute_tolerance < Real{} ||
        !finite(
            retry.preconditioner_pole_relative_tolerance) ||
        retry.preconditioner_pole_relative_tolerance < Real{})
    {
        throw std::invalid_argument(
            "invalid matrix-free spectral scan retry configuration");
    }

    const auto& aggregation = config.aggregation;
    if(
        !finite(aggregation.absolute_tolerance) ||
        aggregation.absolute_tolerance < Real{} ||
        !finite(aggregation.relative_tolerance) ||
        aggregation.relative_tolerance < Real{} ||
        aggregation.minimum_successful_scans == 0 ||
        aggregation.minimum_successful_scans >
            config.transformation.shifts.size() ||
        aggregation.minimum_eigenpairs == 0 ||
        aggregation.probe_count == 0 ||
        aggregation.minimum_successful_probes == 0 ||
        aggregation.minimum_successful_probes >
            aggregation.probe_count ||
        !finite(
            aggregation.eigenvector_independence_tolerance) ||
        !(aggregation.eigenvector_independence_tolerance >
          Real{}) ||
        aggregation.eigenvector_orthogonalization_passes == 0)
    {
        throw std::invalid_argument(
            "invalid matrix-free spectral aggregation configuration");
    }

    const auto& small_system = config.small_system;
    if(
        (small_system.enabled &&
         small_system.maximum_dimension == 0) ||
        !finite(small_system.absolute_residual_tolerance) ||
        small_system.absolute_residual_tolerance < Real{} ||
        !finite(small_system.relative_residual_tolerance) ||
        small_system.relative_residual_tolerance < Real{})
    {
        throw std::invalid_argument(
            "invalid small-system stability eigensolver "
            "configuration");
    }

    const auto& recycling = config.recycling;
    if(
        recycling.maximum_vectors == 0 ||
        !finite(recycling.innovation_weight) ||
        !(recycling.innovation_weight > Real{}) ||
        !(recycling.innovation_weight <= Real(1)) ||
        !finite(recycling.absolute_residual_tolerance) ||
        recycling.absolute_residual_tolerance < Real{} ||
        !finite(recycling.relative_residual_tolerance) ||
        recycling.relative_residual_tolerance < Real{})
    {
        throw std::invalid_argument(
            "invalid matrix-free Ritz-subspace recycling "
            "configuration");
    }
}

template<class Real>
std::vector<
    eigensolvers::transformations::
        complex_affine_factor<Real>>
make_matrix_free_stability_factors(
    matrix_free_spectral_transformation transformation,
    Real step,
    std::size_t repetitions,
    const std::complex<Real>& shift)
{
    using namespace eigensolvers::transformations;
    switch(transformation)
    {
    case matrix_free_spectral_transformation::complex_shift_invert:
        return {{
            std::complex<Real>(Real(1), Real{}),
            -shift}};
    case matrix_free_spectral_transformation::explicit_euler:
        return euler_denominator_factors(
            step,
            repetitions,
            shift);
    case matrix_free_spectral_transformation::classical_rk4:
        return rk4_denominator_factors(
            step,
            repetitions,
            shift);
    }
    throw std::logic_error(
        "unsupported matrix-free spectral transformation");
}

template<class Scan>
std::vector<typename Scan::scan_definition_type>
make_matrix_free_stability_scans(
    const matrix_free_stability_config<
        typename Scan::real_type>& config)
{
    using real_type = typename Scan::real_type;
    validate_matrix_free_stability_config(config);

    std::vector<typename Scan::scan_definition_type> result;
    result.reserve(config.transformation.shifts.size());
    for(std::size_t index = 0;
        index < config.transformation.shifts.size();
        ++index)
    {
        const auto shift = config.transformation.shifts[index];
        typename Scan::scan_definition_type scan;
        std::ostringstream label;
        label
            << matrix_free_spectral_transformation_name(
                   config.transformation.type)
            << "[" << index << "] shift=("
            << shift.real() << "," << shift.imag() << ")";
        scan.label = label.str();
        scan.factors = make_matrix_free_stability_factors(
            config.transformation.type,
            config.transformation.step,
            config.transformation.repetitions,
            shift);
        scan.preconditioner_pole_absolute_tolerance =
            config.retry.preconditioner_pole_absolute_tolerance;
        scan.preconditioner_pole_relative_tolerance =
            config.retry.preconditioner_pole_relative_tolerance;
        scan.inner_basis_retry_sizes =
            config.inner_solver.basis_retry_sizes;
        if(config.retry.enabled)
        {
            scan.retry_attempts.reserve(
                config.retry.maximum_shift_retries);
            for(std::size_t retry_index = 0;
                retry_index <
                    config.retry.maximum_shift_retries;
                ++retry_index)
            {
                const std::size_t level = retry_index/2;
                const real_type magnitude =
                    config.retry.initial_shift_perturbation*
                    std::pow(
                        config.retry.perturbation_growth,
                        static_cast<real_type>(level));
                const real_type direction =
                    retry_index%2 == 0
                    ? real_type(1)
                    : real_type(-1);
                const std::complex<real_type> retry_shift =
                    shift +
                    std::complex<real_type>(
                        direction*magnitude,
                        real_type{});
                typename Scan::scan_attempt_type attempt;
                std::ostringstream retry_label;
                retry_label
                    << "retry[" << retry_index << "] shift=("
                    << retry_shift.real() << ","
                    << retry_shift.imag() << ")";
                attempt.label = retry_label.str();
                attempt.factors =
                    make_matrix_free_stability_factors(
                        config.transformation.type,
                        config.transformation.step,
                        config.transformation.repetitions,
                        retry_shift);
                scan.retry_attempts.emplace_back(
                    std::move(attempt));
            }
        }

        auto& transformed = scan.solver_options.transformed;
        transformed.desired_eigenvalues =
            config.outer.desired_eigenvalues;
        transformed.krylov_dimension =
            config.outer.krylov_dimension;
        transformed.restart_dimension =
            config.outer.restart_dimension;
        transformed.max_restarts =
            config.outer.maximum_restarts;
        transformed.absolute_tolerance =
            config.outer.absolute_tolerance;
        transformed.relative_tolerance =
            config.outer.relative_tolerance;
        transformed.preserve_conjugate_pairs =
            config.outer.preserve_conjugate_pairs;
        transformed.target.kind =
            eigensolvers::spectrum_target::largest_magnitude;
        transformed.orthogonalization.method =
            nmfd::solvers::krylov::
                parse_orthogonalization_method(
                    config.outer.orthogonalization);
        transformed.orthogonalization.reorthogonalization =
            nmfd::solvers::krylov::
                parse_reorthogonalization_policy(
                    config.outer.reorthogonalization);
        transformed.orthogonalization.dgks_eta =
            config.outer.dgks_eta;
        transformed.orthogonalization.breakdown_absolute =
            config.outer.breakdown_absolute_tolerance;
        transformed.orthogonalization.breakdown_relative =
            config.outer.breakdown_relative_tolerance;
        transformed.orthogonalization.max_passes =
            config.outer.maximum_orthogonalization_passes;

        auto& recovery = scan.solver_options.recovery;
        recovery.relative_basis_tolerance =
            config.recovery.relative_basis_tolerance;
        recovery.orthogonalization_passes =
            config.recovery.orthogonalization_passes;
        recovery.absolute_residual_tolerance =
            config.recovery.absolute_residual_tolerance;
        recovery.relative_residual_tolerance =
            config.recovery.relative_residual_tolerance;
        scan.solver_options.
            minimum_converged_physical_eigenpairs =
                config.recovery.minimum_converged_eigenpairs;
        result.emplace_back(std::move(scan));
    }
    return result;
}

template<class InnerParameters, class Real>
InnerParameters make_matrix_free_inner_solver_parameters(
    const matrix_free_stability_config<Real>& config)
{
    validate_matrix_free_stability_config(config);
    InnerParameters result;
    const auto& inner = config.inner_solver;
    result.basis_size = inner.basis_size;
    result.batch_size = inner.batch_size;
    result.preconditioner_side = inner.preconditioner_side;
    result.orthogonalization = inner.orthogonalization;
    result.reorthogonalization_policy =
        inner.reorthogonalization;
    result.dgks_eta = inner.dgks_eta;
    result.breakdown_relative_tolerance =
        inner.breakdown_relative_tolerance;
    result.max_orthogonalization_passes =
        inner.maximum_orthogonalization_passes;
    result.do_restart_on_false_ritz_convergence =
        inner.restart_on_false_ritz_convergence;
    result.monitor.rel_tol = inner.relative_tolerance;
    result.monitor.abs_tol = inner.absolute_tolerance;
    result.monitor.max_iters_num = inner.maximum_iterations;
    result.monitor.min_iters_num = inner.minimum_iterations;
    result.monitor.save_convergence_history =
        inner.save_convergence_history;
    result.monitor.divide_out_norms_by_rel_base =
        inner.divide_norms_by_relative_base;
    result.monitor.out_min_resid_norm =
        inner.output_minimum_residual;
    return result;
}

template<class Real>
spectrum_scan_aggregation_options<Real>
make_spectrum_scan_aggregation_options(
    const matrix_free_stability_config<Real>& config)
{
    validate_matrix_free_stability_config(config);
    spectrum_scan_aggregation_options<Real> result;
    result.absolute_tolerance =
        config.aggregation.absolute_tolerance;
    result.relative_tolerance =
        config.aggregation.relative_tolerance;
    result.minimum_successful_scans =
        config.aggregation.minimum_successful_scans;
    result.minimum_eigenpairs =
        config.aggregation.minimum_eigenpairs;
    result.require_all_scans =
        config.aggregation.require_all_scans;
    result.probe_count =
        config.aggregation.probe_count;
    result.minimum_successful_probes =
        config.aggregation.minimum_successful_probes;
    result.require_all_probes =
        config.aggregation.require_all_probes;
    result.eigenvector_independence_tolerance =
        config.aggregation.
            eigenvector_independence_tolerance;
    result.eigenvector_orthogonalization_passes =
        config.aggregation.
            eigenvector_orthogonalization_passes;
    return result;
}

template<class Real>
recycled_ritz_subspace_options<Real>
make_recycled_ritz_subspace_options(
    const matrix_free_stability_config<Real>& config)
{
    validate_matrix_free_stability_config(config);
    recycled_ritz_subspace_options<Real> result;
    result.enabled = config.recycling.enabled;
    result.maximum_vectors = config.recycling.maximum_vectors;
    result.innovation_weight = config.recycling.innovation_weight;
    result.absolute_residual_tolerance =
        config.recycling.absolute_residual_tolerance;
    result.relative_residual_tolerance =
        config.recycling.relative_residual_tolerance;
    return result;
}

} // namespace analysis
} // namespace stability

#endif
