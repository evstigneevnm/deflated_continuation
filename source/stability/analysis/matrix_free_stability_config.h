#ifndef __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_CONFIG_H__
#define __STABILITY_ANALYSIS_MATRIX_FREE_STABILITY_CONFIG_H__

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stability
{
namespace analysis
{

enum class matrix_free_spectral_transformation
{
    complex_shift_invert,
    explicit_euler,
    classical_rk4
};

inline const char* matrix_free_spectral_transformation_name(
    matrix_free_spectral_transformation transformation)
{
    switch(transformation)
    {
    case matrix_free_spectral_transformation::complex_shift_invert:
        return "complex_shift_invert";
    case matrix_free_spectral_transformation::explicit_euler:
        return "explicit_euler";
    case matrix_free_spectral_transformation::classical_rk4:
        return "classical_rk4";
    }
    return "unknown";
}

inline matrix_free_spectral_transformation
parse_matrix_free_spectral_transformation(const std::string& value)
{
    if(value == "complex_shift_invert" || value == "shift_invert")
    {
        return
            matrix_free_spectral_transformation::
                complex_shift_invert;
    }
    if(value == "explicit_euler" || value == "euler")
    {
        return
            matrix_free_spectral_transformation::
                explicit_euler;
    }
    if(value == "classical_rk4" || value == "rk4")
    {
        return
            matrix_free_spectral_transformation::
                classical_rk4;
    }
    throw std::invalid_argument(
        "unknown matrix-free spectral transformation: " + value);
}

template<class Real>
struct matrix_free_stability_config
{
    struct transformation_config
    {
        matrix_free_spectral_transformation type =
            matrix_free_spectral_transformation::
                complex_shift_invert;
        Real step = Real(0.1);
        std::size_t repetitions = 3;
        std::vector<std::complex<Real>> shifts;
    };

    struct outer_solver_config
    {
        std::size_t desired_eigenvalues = 6;
        std::size_t krylov_dimension = 24;
        std::size_t restart_dimension = 12;
        std::size_t maximum_restarts = 100;
        Real absolute_tolerance = Real{};
        Real relative_tolerance = Real(1.0e-8);
        bool preserve_conjugate_pairs = true;
        std::string orthogonalization = "mgs";
        std::string reorthogonalization = "dgks";
        Real dgks_eta = Real(0.717);
        Real breakdown_absolute_tolerance = Real{};
        Real breakdown_relative_tolerance = Real(1.0e-12);
        unsigned int maximum_orthogonalization_passes = 2;
    };

    struct recovery_config
    {
        Real relative_basis_tolerance = Real(1.0e-10);
        std::size_t orthogonalization_passes = 2;
        Real absolute_residual_tolerance = Real(1.0e-9);
        Real relative_residual_tolerance = Real(1.0e-8);
        std::size_t minimum_converged_eigenpairs = 1;
    };

    struct inner_solver_config
    {
        unsigned int basis_size = 30;
        unsigned int batch_size = 5;
        char preconditioner_side = 'R';
        std::string orthogonalization = "mgs";
        std::string reorthogonalization = "dgks";
        Real dgks_eta = Real(0.717);
        Real breakdown_relative_tolerance = Real(1.0e-12);
        unsigned int maximum_orthogonalization_passes = 2;
        bool restart_on_false_ritz_convergence = false;
        std::vector<unsigned int> basis_retry_sizes;
        Real relative_tolerance = Real(1.0e-10);
        Real absolute_tolerance = Real(1.0e-12);
        int maximum_iterations = 300;
        int minimum_iterations = 0;
        bool save_convergence_history = false;
        bool divide_norms_by_relative_base = false;
        bool output_minimum_residual = false;
        bool verbose = false;
    };

    struct retry_config
    {
        bool enabled = false;
        std::size_t maximum_shift_retries = 0;
        Real initial_shift_perturbation = Real(0.05);
        Real perturbation_growth = Real(2);
        Real preconditioner_pole_absolute_tolerance = Real{};
        Real preconditioner_pole_relative_tolerance =
            Real(1.0e-10);
    };

    struct aggregation_config
    {
        Real absolute_tolerance = Real(1.0e-8);
        Real relative_tolerance = Real(1.0e-7);
        std::size_t minimum_successful_scans = 1;
        std::size_t minimum_eigenpairs = 1;
        bool require_all_scans = true;
        std::size_t probe_count = 1;
        std::size_t minimum_successful_probes = 1;
        bool require_all_probes = true;
        Real eigenvector_independence_tolerance =
            Real(1.0e-6);
        std::size_t eigenvector_orthogonalization_passes = 2;
    };

    struct recycling_config
    {
        bool enabled = false;
        std::size_t maximum_vectors = 16;
        Real innovation_weight = Real(0.1);
        Real absolute_residual_tolerance = Real(1.0e-8);
        Real relative_residual_tolerance = Real(0.25);
    };

    struct invariant_subspace_tracking_config
    {
        bool enabled = false;
        std::size_t maximum_dimension = 16;
        std::size_t maximum_seed_vectors = 8;
        std::size_t coverage_recovery_maximum_seed_vectors = 0;
        std::size_t orthogonalization_passes = 2;
        Real seed_innovation_weight = Real(0.1);
        Real dependence_tolerance = Real(1.0e-8);
        Real minimum_retained_residual_ratio = Real(0.05);
        Real absolute_invariance_tolerance = Real(1.0e-8);
        Real relative_invariance_tolerance = Real(0.25);
        Real real_eigenvalue_tolerance = Real(1.0e-8);
        Real eigenvalue_group_tolerance = Real(1.0e-6);
        Real principal_angle_rank_tolerance = Real(1.0e-8);
    };

    struct small_system_config
    {
        bool enabled = false;
        std::size_t maximum_dimension = 0;
        bool prefer = false;
        Real absolute_residual_tolerance = Real(1.0e-9);
        Real relative_residual_tolerance = Real(1.0e-8);
    };

    bool enabled = false;
    Real linearization_scale = Real(1);
    transformation_config transformation;
    outer_solver_config outer;
    recovery_config recovery;
    inner_solver_config inner_solver;
    retry_config retry;
    aggregation_config aggregation;
    recycling_config recycling;
    invariant_subspace_tracking_config invariant_subspace_tracking;
    small_system_config small_system;
};

} // namespace analysis
} // namespace stability

#endif
