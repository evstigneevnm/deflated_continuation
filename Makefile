# this Makefile is configured by the selected build config.

ifndef CONFIG_FILE
CONFIG_FILE = build_configs/config_omen_home_release.inc
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
$(info config file $(CONFIG_FILE) does not exist.)
$(error Create $(CONFIG_FILE) from example or specify another config via: make <target> CONFIG_FILE=<config_filename> )
endif

-include $(CONFIG_FILE)

ifndef REAL_TYPE
$(info REAL_TYPE is not defined in config. Use float by default.)
REAL_TYPE = float
endif

ifndef SCALAR_TYPE
SCALAR_TYPE = -DSCALAR_TYPE=$(REAL_TYPE)
endif

ifndef CPPSTD
$(info CPPSTD is not defined in config. Use c++17 by default.)
CPPSTD = c++17
endif

ifndef GCC_ROOT_PATH
$(info GCC_ROOT_PATH is not defined in config. Use /usr by default.)
GCC_ROOT_PATH = /usr
endif

ifndef BUILD_DIR
$(info BUILD_DIR is not defined in config. Use ./build by default.)
BUILD_DIR = ./build
endif

ifdef CUDA_ARCH
$(info CUDA_ARCH is defined in config. This parameter is deprecated and will be overwritten by CUDA_ARCH_LIST content.)
$(info Use in config something like CUDA_ARCH_LIST = 75)
endif

ifndef CUDA_ARCH_LIST
CUDA_ARCH_LIST = 75
$(info CUDA_ARCH_LIST is not defined in config. Use $(CUDA_ARCH_LIST) by default.)
endif

CUDA_ARCH = $(shell echo '$(CUDA_ARCH_LIST)' | awk '{for (i=1;i<=NF;++i) {printf " -gencode arch=compute_%s,code=sm_%s", $$i, $$i}}')
$(info CUDA_ARCH = $(CUDA_ARCH))

BUILD_STAMP = $(BUILD_DIR)/dir_tag.out
RESULTS = $(BUILD_DIR)/results.make

$(BUILD_STAMP):
	mkdir -p $(BUILD_DIR)
	touch $(BUILD_STAMP)

ifneq ($(strip $(MAKECMDGOALS)),)
$(MAKECMDGOALS): | $(BUILD_STAMP)
endif

NVCC = $(CUDA_ROOT_PATH)/bin/nvcc
NVCC_CHECK_REGS = -Xptxas -v
NVCCFLAGS = -Wno-deprecated-gpu-targets --expt-relaxed-constexpr $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC)
ifdef HIP_ROOT_PATH
HIPCC ?= $(HIP_ROOT_PATH)/bin/hipcc
else
HIPCC ?= hipcc
endif
HIP_EXTENDED_LAMBDA ?= $(shell $(HIPCC) --version 2>/dev/null | grep -iq nvcc && printf '%s' '--extended-lambda')
ifdef HIP_ARCH_LIST
HIP_ARCH = $(shell echo '$(HIP_ARCH_LIST)' | awk '{for (i=1;i<=NF;++i) {if ($$i ~ /^[0-9]+$$/) printf " -gencode arch=compute_%s,code=sm_%s", $$i, $$i; else printf " --offload-arch=%s", $$i}}')
HIP_NVIDIA_BACKEND = $(shell echo '$(HIP_ARCH_LIST)' | awk '{for (i=1;i<=NF;++i) {if ($$i ~ /^[0-9]+$$/) {printf "1"; exit}}}')
else
HIP_ARCH =
HIP_NVIDIA_BACKEND =
endif
ifeq ($(HIP_NVIDIA_BACKEND),1)
HIP_PLATFORM_DEFINES = -DNMFD_HIGH_PRECISION_BLAS1_ENABLE_HIP_NVIDIA=1
else
HIP_PLATFORM_DEFINES =
endif
HIPFLAGS ?= $(HIP_ARCH) $(HIP_PLATFORM_DEFINES) -std=$(CPPSTD) $(HIP_EXTENDED_LAMBDA) $(TARGET_HIPCC)
OPENMP = -fopenmp -lpthread
NVOPENMP = -Xcompiler $(OPENMP)
CONTRIB_SCFD = source/contrib/scfd
CONTRIB_NMFD_LINSOLVERS = source/contrib/nmfd-linsolvers
COMMON_NMFD_OPERATIONS = source/common/NMFD-operations

G++ = $(GCC_ROOT_PATH)/g++
LIBFLAGS = --compiler-options -fPIC
G++FLAGS = -std=$(CPPSTD) $(TARGET_GCC)
ICUDA = -I$(CUDA_ROOT_PATH)/include
IPROJECT = -I $(COMMON_NMFD_OPERATIONS) -I source/ -I $(CONTRIB_SCFD)/include
INMFD_LINSOLVERS = -I $(CONTRIB_NMFD_LINSOLVERS)/include
IBOOST = -I$(BOOST_ROOT_PATH)/include
HIGH_PRECISION_BLAS1_CUDA_HEADERS = source/common/NMFD-operations/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita.h source/common/NMFD-operations/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_type.h source/common/NMFD-operations/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_impl.cuh source/common/NMFD-operations/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_impl_functions.cuh source/common/NMFD-operations/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_impl_shmem.cuh
HIGH_PRECISION_BLAS1_HIP_HEADERS = source/common/NMFD-operations/nmfd/operations/blas1/high_precision/hip/gpu_reduction_ogita.h
HIGH_PRECISION_BLAS1_HEADERS = source/common/NMFD-operations/nmfd/operations/blas1/high_precision/compensated_reduction.h $(HIGH_PRECISION_BLAS1_CUDA_HEADERS) $(HIGH_PRECISION_BLAS1_HIP_HEADERS)
SCFD_VECTOR_OPS_HEADERS = source/common/scfd_vector_operations.h source/common/NMFD-operations/nmfd/operations/scfd_vector_operations.h $(HIGH_PRECISION_BLAS1_HEADERS) source/common/NMFD-operations/nmfd/operations/vector_operations_base.h source/common/NMFD-operations/nmfd/operations/vector_space_base.h
SCFD_SERIAL_VECTOR_OPS_HEADERS = $(SCFD_VECTOR_OPS_HEADERS) source/common/scfd_serial_cpu_vector_operations.h
SCFD_VECTOR_OPS_TEST_HEADERS = $(SCFD_VECTOR_OPS_HEADERS) source/common/tests/scfd_vector_operations_high_precision_tests.h source/common/tests/scfd_vector_operations_nmfd_interface_tests.h
CPU_VECTOR_OPS_VAR_PREC_HEADERS = source/common/cpu_vector_operations_var_prec.h source/common/NMFD-operations/nmfd/operations/cpu_vector_operations_var_prec.h source/common/NMFD-operations/nmfd/operations/vector_operations_base.h source/common/NMFD-operations/nmfd/operations/vector_space_base.h
COMMON_FILE_OPS_HEADERS = source/common/file_operations.h source/common/cpu_file_operations.h source/common/cpu_matrix_file_operations.h source/common/gpu_file_operations.h source/common/gpu_matrix_file_operations.h source/common/NMFD-operations/nmfd/operations/io/file_operations.h source/common/NMFD-operations/nmfd/operations/io/vector_file_operations.h source/common/NMFD-operations/nmfd/operations/io/matrix_file_operations.h
SMALL_DENSE_LINALG_HEADERS = source/common/NMFD-operations/nmfd/operations/linalg/small_dense.h
HOST_SMALL_DENSE_HEADERS = source/common/NMFD-operations/nmfd/operations/linalg/host_small_dense_backend.h source/common/NMFD-operations/nmfd/operations/linalg/host_small_dense_lapack.h source/contrib/scfd/include/scfd/external_libraries/lapack_wrap.h
MATRIX_MARKET_HEADERS = source/common/NMFD-operations/nmfd/operations/io/matrix_market.h source/common/NMFD-operations/nmfd/operations/sparse/host_csr_matrix.h
NMFD_KRYLOV_HEADERS = source/contrib/nmfd-linsolvers/include/nmfd/solvers/krylov/operator_apply.h source/contrib/nmfd-linsolvers/include/nmfd/solvers/krylov/orthogonalization.h source/contrib/nmfd-linsolvers/include/nmfd/solvers/krylov/arnoldi.h source/contrib/nmfd-linsolvers/include/nmfd/solvers/krylov/basis_storage.h
ANALYTICAL_EIGENPROBLEM_HEADERS = source/stability/tests/common/analytical_eigenproblem.h source/stability/tests/common/analytical_dense_operator.h source/stability/tests/common/analytical_real_affine_inverse_model.h source/stability/tests/common/eigensolver_test_harness.h
EIGENSOLVER_HEADERS = source/stability/eigensolvers/eigensolver_result.h source/stability/eigensolvers/eigenvalue_target.h source/stability/eigensolvers/inverse_iteration.h source/stability/eigensolvers/ritz_recovery.h source/stability/eigensolvers/detail/ordered_real_schur.h source/stability/eigensolvers/krylov_schur.h source/stability/eigensolvers/matrix_free_factorized_krylov_schur.h
STABILITY_ANALYSIS_HEADERS = source/stability/model_adapter_contract.h source/stability/analysis/stability_point_result.h source/stability/analysis/spectrum_classifier.h source/stability/analysis/spectrum_scan_aggregator.h source/stability/analysis/validated_spectrum_union.h source/stability/analysis/eigenvector_rank_aggregator.h source/stability/analysis/recycled_ritz_subspace.h source/stability/analysis/eigensolver_adapter.h source/stability/analysis/dimension_guarded_eigensolver.h source/stability/analysis/matrix_free_stability_config.h source/stability/analysis/matrix_free_stability_configuration.h source/stability/analysis/matrix_free_stability_solver.h source/stability/analysis/matrix_free_stability_scan.h source/stability/analysis/initial_vector_policy.h source/stability/analysis/detail/vector_workspace.h source/stability/analysis/transition_state_alignment.h source/stability/analysis/source_parameter_path.h source/stability/analysis/source_parameter_state_marcher.h source/stability/analysis/stability_evaluator.h source/stability/analysis/stability_transition_refiner.h source/stability/tracking/principal_angles.h source/stability/tracking/tracked_invariant_subspace.h source/stability/persistence/classification_uncertainty_registry.h source/stability/stability_analysis.hpp
AFFINE_PRECONDITIONER_HEADERS = source/stability/eigensolvers/transformations/affine_inverse_health.h source/stability/eigensolvers/transformations/complex_affine_preconditioner.h source/stability/eigensolvers/transformations/complexified_real_affine_preconditioner.h source/stability/eigensolvers/transformations/complexified_real_operator.h source/stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h source/stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h source/stability/eigensolvers/transformations/scaled_real_operator.h source/stability/eigensolvers/transformations/scaled_real_affine_inverse_provider.h
EIGENSOLVER_TRANSFORMATION_HEADERS = source/stability/eigensolvers/transformations/identity_operator.h source/stability/eigensolvers/transformations/affine_pencil_operator.h source/stability/eigensolvers/transformations/inverse_composed_operator.h source/stability/eigensolvers/transformations/inexact_exponential_operator.h source/stability/eigensolvers/transformations/stability_polynomial_operator.h source/stability/eigensolvers/transformations/complex_affine_factor.h source/stability/eigensolvers/transformations/complex_affine_block_operator.h source/stability/eigensolvers/transformations/complex_shift_block_operator.h source/stability/eigensolvers/transformations/pointwise_diagonal_preconditioner.h $(AFFINE_PRECONDITIONER_HEADERS) source/stability/eigensolvers/transformations/complex_solver_product_adapter.h source/stability/eigensolvers/transformations/tracked_linear_solver.h source/stability/eigensolvers/transformations/factor_solver_statistics.h source/stability/eigensolvers/transformations/iterative_factor_solver_bundle.h source/stability/eigensolvers/transformations/spectral_mapping.h source/stability/eigensolvers/transformations/polynomial_denominator_factorization.h source/stability/eigensolvers/transformations/stability_polynomial_factorization.h source/stability/eigensolvers/transformations/euler_polynomial_factorization.h source/stability/eigensolvers/transformations/factorized_inverse_solver.h source/stability/eigensolvers/projected_spectrum_recovery.h source/stability/eigensolvers/rotated_projected_spectrum_recovery.h
PRODUCT_VECTOR_SPACE_HEADERS = source/common/NMFD-operations/nmfd/operations/product_vector_space.h source/common/NMFD-operations/nmfd/operations/scfd_complex_vector_bridge.h
NMFD_GMRES_HEADERS = source/contrib/nmfd-linsolvers/include/nmfd/solvers/gmres.h source/contrib/nmfd-linsolvers/include/nmfd/solvers/iter_solver_base.h source/contrib/nmfd-linsolvers/include/nmfd/solvers/default_monitor.h source/contrib/nmfd-linsolvers/include/nmfd/solvers/monitor_krylov.h $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS)
NMFD_GMRES_BASELINE_HEADERS = source/numerical_algos/lin_solvers/tests/nmfd_cpu_reference_vector_space.h source/contrib/nmfd-linsolvers/test/solvers/linear_operator_advection.h source/contrib/nmfd-linsolvers/test/solvers/linear_operator_diffusion.h source/contrib/nmfd-linsolvers/test/solvers/linear_operator_elliptic.h source/contrib/nmfd-linsolvers/test/solvers/preconditioner_advection.h source/contrib/nmfd-linsolvers/test/solvers/preconditioner_diffusion.h source/contrib/nmfd-linsolvers/test/solvers/preconditioner_elliptic.h source/contrib/nmfd-linsolvers/test/solvers/residual_regularization_test.h
COMPLEX_TRAITS_HEADERS = source/common/scfd_backend_ext/complex.h
CONTINUATION_SYMMETRY_HEADERS = source/symmetry/translation/orbit_type.h source/symmetry/continuation/continuation_chart_state.h source/symmetry/continuation/geometry_traits.h source/symmetry/continuation/identity_continuation_chart.h source/symmetry/continuation/isotropy_transition.h source/symmetry/continuation/local_representative_policy.h
CONTINUATION_CHART_HEADERS = source/continuation/chart_helpers.h source/continuation/predictor_chart_validation.h source/continuation/predictor_chart_probe.h source/continuation/predictor_chart_diagnostics.h source/continuation/corrector_retry_policy.h source/continuation/progress_monitor.h source/continuation/continuation_step_state.h source/continuation/continuation_endpoint_state.h source/continuation/observational_knot_sample.h source/continuation/pending_branch_event.h source/continuation/initial_tangent_candidates.h source/continuation/initial_tangent_chart_validator.h source/continuation/initial_tangent_secant_builder.h source/continuation/semicurve_tangent_cache.h source/continuation/tangent_normalization.h $(CONTINUATION_SYMMETRY_HEADERS)
SYMMETRY_LINEARIZATION_HEADERS = source/symmetry/linearization/projected_linear_operator.h source/symmetry/linearization/projected_stability_linear_operator.h source/symmetry/linearization/projected_preconditioner.h source/symmetry/linearization/projected_affine_inverse_provider.h source/symmetry/linearization/projected_linearization_provider.h source/symmetry/linearization/projected_stability_gauge.h
SYMMETRY_CORE_HEADERS = source/symmetry/slice_data.h source/symmetry/slice_projector.h source/symmetry/quotient_classifier.h source/symmetry/stabilized_storage.h source/symmetry/generated_finite_group.h source/symmetry/finite_group_manifest.h source/symmetry/finite_action_registry.h source/symmetry/finite_quotient_adapter.h
SYMMETRY_FOURIER_HEADERS = $(CONTINUATION_SYMMETRY_HEADERS) source/symmetry/fourier/mode_descriptor.h source/symmetry/fourier/mode_access.h source/symmetry/fourier/phase_conditions.h source/symmetry/fourier/translation_generators.h source/symmetry/fourier/translation_action.h source/symmetry/fourier/periodic_affine_element_2d.h source/symmetry/fourier/periodic_affine_action_2d.h source/symmetry/fourier/residual_translation_group_2d.h source/symmetry/fourier/residual_translation_orbit_aligner_2d.h source/symmetry/fourier/active_mode_basis.h source/symmetry/fourier/fourier_spectrum_ops.h source/symmetry/fourier/fourier_isotropy_1d.h source/symmetry/fourier/fourier_slice_1d.h source/symmetry/fourier/fourier_slice_differential_1d.h source/symmetry/fourier/lsq_phase_solver_1d.h source/symmetry/fourier/lsq_fourier_slice_1d_policy.h source/symmetry/fourier/lsq_fourier_slice_1d_strategy.h source/symmetry/fourier/frozen_fourier_chart_1d.h source/symmetry/fourier/fourier_slice.h source/symmetry/fourier/real_packed_fourier_codec_1d.h source/symmetry/fourier/real_packed_fourier_slice_1d_policy.h source/symmetry/fourier/real_packed_fourier_slice_1d_policy_json.h source/symmetry/fourier/real_packed_fourier_slice_1d_adapter.h source/symmetry/fourier/real_packed_fourier_actions_1d.h $(SYMMETRY_CORE_HEADERS) $(COMPLEX_TRAITS_HEADERS) $(SMALL_DENSE_LINALG_HEADERS)
DEFLATION_SYMMETRY_HEADERS = source/deflation/symmetry_solution_storage.h $(SYMMETRY_FOURIER_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
FFT_FACADE_HEADERS = source/external_libraries/fft_facade.h source/external_libraries/fft_facade_fftw.h source/external_libraries/fft_facade_cufft.h source/external_libraries/fftw_wrap.h source/external_libraries/cufft_wrap.h
RESIDUAL_DECOMPOSITION_TEST_HEADER = source/nonlinear_operators/tests/linear_nonlinear_decomposition_test.h
ADJOINT_JACOBIAN_TEST_HEADERS = source/nonlinear_operators/adjoint_jacobian_capability.h source/nonlinear_operators/tests/adjoint_jacobian_test.h
FOURIER_DISCRETIZATION_HEADERS = source/discretization/common/structured_extent.h source/discretization/common/component_field.h source/discretization/fourier/periodic_grid.h source/discretization/fourier/r2c_index_space.h source/discretization/fourier/spectral_field.h source/discretization/fourier/normalized_fft.h source/discretization/fourier/wavevector_table.h source/discretization/fourier/diagonal_symbols.h source/discretization/fourier/dealiasing_policy.h source/discretization/fourier/initialization/low_mode_odd_field.h source/discretization/fourier/codecs/codec_descriptor_2d.h source/discretization/fourier/codecs/full_real_field.h source/discretization/fourier/codecs/translation_equivariant_real_field.h source/discretization/fourier/codecs/inversion_odd_field.h source/discretization/fourier/codecs/component_product.h source/discretization/fourier/operations/derivative.h source/discretization/fourier/operations/laplacian.h source/discretization/fourier/operations/inverse_laplacian.h source/discretization/fourier/operations/pseudospectral_product.h source/discretization/fourier/operations/vector_advection.h
KS2D_REFACTORED_HEADERS = source/models/KS_2D/KS2D_backend_typedefs.h source/nonlinear_operators/detail/linear_nonlinear_terms.h source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h source/symmetry/generated_finite_group.h source/symmetry/finite_group_manifest.h source/symmetry/fourier/periodic_affine_element_2d.h source/symmetry/fourier/periodic_affine_action_2d.h source/symmetry/fourier/residual_translation_group_2d.h source/symmetry/fourier/residual_translation_orbit_aligner_2d.h source/symmetry/finite_action_registry.h source/symmetry/finite_quotient_adapter.h source/containers/bifurcation_diagram/symmetry_archive_audit.h source/deflation/symmetry_solution_storage.h $(FOURIER_DISCRETIZATION_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
VISUALIZATION_HEADERS = source/visualization/bd_curve_file_reader.h source/visualization/physical_solution_writer.h source/visualization/structured_physical_solution_writer.h source/visualization/io/numpy_array_writer.h source/visualization/bd_prepare_visualization.hpp
CIRCLE_MODEL_HEADERS = source/models/circle/circle_backend_typedefs.h source/nonlinear_operators/circle/circle.h source/nonlinear_operators/circle/convergence_strategy.h source/nonlinear_operators/circle/linear_operator_circle.h source/nonlinear_operators/circle/preconditioner_circle.h source/nonlinear_operators/circle/system_operator.h
BRATU_MODEL_HEADERS = source/models/bratu/bratu_backend_typedefs.h source/models/bratu/bratu_model_config.h source/nonlinear_operators/detail/linear_nonlinear_terms.h source/nonlinear_operators/bratu/bratu.h source/nonlinear_operators/bratu/convergence_strategy.h source/nonlinear_operators/bratu/linear_operator_bratu.h source/nonlinear_operators/bratu/preconditioner_bratu.h source/nonlinear_operators/bratu/system_operator.h
STAR_SHAPED_MODEL_HEADERS = source/models/star_shaped/star_shaped_backend_typedefs.h source/nonlinear_operators/star_shaped/star_shaped.h source/nonlinear_operators/star_shaped/convergence_strategy.h source/nonlinear_operators/star_shaped/linear_operator_star_shaped.h source/nonlinear_operators/star_shaped/preconditioner_star_shaped.h source/nonlinear_operators/star_shaped/system_operator.h
KS1D_MODEL_HEADERS = source/models/KS_1D/KS1D_backend_typedefs.h source/nonlinear_operators/detail/linear_nonlinear_terms.h source/nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h source/nonlinear_operators/Kuramoto_Sivashinskiy_1D/convergence_strategy.h source/nonlinear_operators/Kuramoto_Sivashinskiy_1D/linear_operator_KS_1D.h source/nonlinear_operators/Kuramoto_Sivashinskiy_1D/preconditioner_KS_1D.h source/nonlinear_operators/Kuramoto_Sivashinskiy_1D/system_operator.h
KS1D_FULL_MODEL_HEADERS = $(KS1D_MODEL_HEADERS) source/nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d_full.h $(SYMMETRY_LINEARIZATION_HEADERS) source/nonlinear_operators/projected_system_operator.h source/nonlinear_operators/projected_operator_helpers.h source/continuation/projected_system_operator_continuation.h $(CONTINUATION_CHART_HEADERS) $(DEFLATION_SYMMETRY_HEADERS)
STABILITY_CONTINUATION_HEADERS = source/main/stability_continuation.hpp source/containers/stability_diagram.h $(STABILITY_ANALYSIS_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(HOST_SMALL_DENSE_HEADERS)
KS1D_STABILITY_HEADERS = source/models/KS_1D/KS1D_stability_cli.h $(STABILITY_CONTINUATION_HEADERS)
KS1D_STABILITY_REPLAY_SCRIPT = scripts/test_ks1d_full_stability_replay.py
KS1D_STABILITY_REPLAY_MANIFEST = source/models/KS_1D/tests/data/stability_replay/manifest.json

LCUDA = -L$(CUDA_ROOT_PATH)/lib64
LBOOST = -L$(BOOST_ROOT_PATH)/lib
LIBS1 = $(LCUDA) -lcublas -lcurand
LIBS2 = $(LCUDA) -lcufft $(LIBS1)
LIBS3 = $(LCUDA) -lcusolver $(LIBS2)
LIBSAll = $(LCUDA) -lcublas -lcurand -lcufft -lcusolver
LIBBOOST = -lboost_serialization
LLAPACK = -L$(OPENBLAS_ROOT_PATH)/lib -lopenblas
LFFTW = -lfftw3 -lfftw3f
ISUITESPARSE ?=
LUMFPACK ?= -lumfpack

#clean
clean:
	rm -f $(BUILD_DIR)/*.bin $(BUILD_DIR)/*.o $(RESULTS)

#component tests
test_cpu_vector_operations.bin: source/common/tests/test_cpu_vector_operations.cpp source/common/tests/vector_operations_template_tests.h $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/common/tests/test_cpu_vector_operations.cpp $(OPENMP) -o $(BUILD_DIR)/test_cpu_vector_operations.bin 2>$(RESULTS)

test_cpu_vector_operations_var_prec.bin: source/common/tests/test_cpu_vector_operations_var_prec.cpp $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(IBOOST) source/common/tests/test_cpu_vector_operations_var_prec.cpp -o $(BUILD_DIR)/test_cpu_vector_operations_var_prec.bin 2>$(RESULTS)

test_multivector.bin: source/common/tests/test_multivector.cpp $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/common/tests/test_multivector.cpp $(OPENMP) -o $(BUILD_DIR)/test_multivector.bin 2>$(RESULTS)

test_file_operations.bin: source/common/tests/test_file_operations.cpp $(COMMON_FILE_OPS_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/common/tests/test_file_operations.cpp $(OPENMP) -o $(BUILD_DIR)/test_file_operations.bin 2>$(RESULTS)

test_file_operations_cuda.bin: source/common/tests/test_file_operations_cuda.cu $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) source/common/cuda_init_scfd.h
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/tests/test_file_operations_cuda.cu $(LIBS1) -o $(BUILD_DIR)/test_file_operations_cuda.bin 2>$(RESULTS)

test_fftw_operations.bin: source/external_libraries/tests/test_fftw_operations.cpp source/external_libraries/fftw_wrap.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/external_libraries/tests/test_fftw_operations.cpp $(LFFTW) -o $(BUILD_DIR)/test_fftw_operations.bin 2>$(RESULTS)

test_fft_facade_fftw.bin: source/external_libraries/tests/test_fft_facade_fftw.cpp $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/external_libraries/tests/test_fft_facade_fftw.cpp $(LFFTW) -o $(BUILD_DIR)/test_fft_facade_fftw.bin 2>$(RESULTS)

test_fft_facade_cufft.bin: source/external_libraries/tests/test_fft_facade_cufft.cu $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/external_libraries/tests/test_fft_facade_cufft.cu $(LIBS2) -o $(BUILD_DIR)/test_fft_facade_cufft.bin 2>$(RESULTS)

test_fourier_discretization_cpu_omp.bin: source/discretization/fourier/tests/test_fourier_foundation_cpu_omp.cpp source/discretization/fourier/tests/fourier_foundation_test_suite.h source/discretization/fourier/tests/legacy_ks2d_codec_adapter.h $(FOURIER_DISCRETIZATION_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/discretization/fourier/tests/test_fourier_foundation_cpu_omp.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/test_fourier_discretization_cpu_omp.bin 2>$(RESULTS)

test_fourier_discretization_cuda.bin: source/discretization/fourier/tests/test_fourier_foundation_cuda.cu source/discretization/fourier/tests/fourier_foundation_test_suite.h source/discretization/fourier/tests/legacy_ks2d_codec_adapter.h $(FOURIER_DISCRETIZATION_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/discretization/fourier/tests/test_fourier_foundation_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_fourier_discretization_cuda.bin 2>$(RESULTS)

test_fourier_translation_cpu_omp.bin: source/symmetry/tests/test_fourier_translation_action.cpp source/symmetry/fourier/translation_action.h source/symmetry/fourier/active_mode_basis.h $(FOURIER_DISCRETIZATION_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/symmetry/tests/test_fourier_translation_action.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/test_fourier_translation_cpu_omp.bin 2>$(RESULTS)

test_fourier_translation_cuda.bin: source/symmetry/tests/test_fourier_translation_action.cpp source/symmetry/fourier/translation_action.h source/symmetry/fourier/active_mode_basis.h $(FOURIER_DISCRETIZATION_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DTEST_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/symmetry/tests/test_fourier_translation_action.cpp $(LIBS2) -o $(BUILD_DIR)/test_fourier_translation_cuda.bin 2>$(RESULTS)

test_periodic_affine_group_2d.bin: source/symmetry/tests/test_periodic_affine_group_2d.cpp source/symmetry/generated_finite_group.h source/symmetry/fourier/periodic_affine_element_2d.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_periodic_affine_group_2d.cpp -o $(BUILD_DIR)/test_periodic_affine_group_2d.bin 2>$(RESULTS)

test_residual_translation_group_2d.bin: source/symmetry/tests/test_residual_translation_group_2d.cpp source/symmetry/fourier/residual_translation_group_2d.h source/symmetry/fourier/mode_descriptor.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_residual_translation_group_2d.cpp -o $(BUILD_DIR)/test_residual_translation_group_2d.bin 2>$(RESULTS)

test_finite_group_manifest.bin: source/symmetry/tests/test_finite_group_manifest.cpp source/symmetry/finite_group_manifest.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_finite_group_manifest.cpp -o $(BUILD_DIR)/test_finite_group_manifest.bin 2>$(RESULTS)

test_KS2D_operator_cpu_omp.bin: source/models/KS_2D/test_KS2D_operator.cpp source/symmetry/fourier/translation_action.h $(KS2D_REFACTORED_HEADERS) $(RESIDUAL_DECOMPOSITION_TEST_HEADER) $(ADJOINT_JACOBIAN_TEST_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/KS_2D/test_KS2D_operator.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/test_KS2D_operator_cpu_omp.bin 2>$(RESULTS)

test_KS2D_operator_cuda.bin: source/models/KS_2D/test_KS2D_operator.cpp source/symmetry/fourier/translation_action.h $(KS2D_REFACTORED_HEADERS) $(RESIDUAL_DECOMPOSITION_TEST_HEADER) $(ADJOINT_JACOBIAN_TEST_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DTEST_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/KS_2D/test_KS2D_operator.cpp -c -o $(BUILD_DIR)/test_KS2D_operator_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_KS2D_operator_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_KS2D_operator_cuda.bin 2>$(RESULTS)

test_KS2D_stability_cli.bin: source/models/KS_2D/tests/test_KS2D_stability_cli.cpp source/models/KS_2D/KS2D_stability_cli.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/models/KS_2D/tests/test_KS2D_stability_cli.cpp -o $(BUILD_DIR)/test_KS2D_stability_cli.bin 2>$(RESULTS)

test_numpy_array_writer.bin: source/visualization/tests/test_numpy_array_writer.cpp source/visualization/io/numpy_array_writer.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/visualization/tests/test_numpy_array_writer.cpp -o $(BUILD_DIR)/test_numpy_array_writer.bin 2>$(RESULTS)

KS2D_bd_cpu_omp: source/models/KS_2D/KS2D_bd.cpp $(KS2D_REFACTORED_HEADERS) $(COMMON_FILE_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DKS2D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/KS_2D/KS2D_bd.cpp $(OPENMP) $(LFFTW) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS2D_bd_cpu_omp.bin 2>$(RESULTS)

KS2D_bd_cuda: source/models/KS_2D/KS2D_bd.cpp $(KS2D_REFACTORED_HEADERS) $(COMMON_FILE_OPS_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS2D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/KS_2D/KS2D_bd.cpp -c -o $(BUILD_DIR)/KS2D_bd_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/KS2D_bd_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS2D_bd_cuda.bin 2>$(RESULTS)

KS2D_prepare_visualization_cpu_omp: source/models/KS_2D/KS2D_prepare_visualization.cpp $(KS2D_REFACTORED_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(VISUALIZATION_HEADERS)
	$(G++) $(G++FLAGS) -DKS2D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/KS_2D/KS2D_prepare_visualization.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/KS2D_prepare_visualization_cpu_omp.bin 2>$(RESULTS)

KS2D_prepare_visualization_cuda: source/models/KS_2D/KS2D_prepare_visualization.cpp $(KS2D_REFACTORED_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(VISUALIZATION_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS2D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/KS_2D/KS2D_prepare_visualization.cpp -c -o $(BUILD_DIR)/KS2D_prepare_visualization_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/KS2D_prepare_visualization_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/KS2D_prepare_visualization_cuda.bin 2>$(RESULTS)

KS2D_stability_cpu_omp: source/models/KS_2D/KS2D_stability.cpp source/models/KS_2D/KS2D_stability_cli.h $(KS2D_REFACTORED_HEADERS) $(STABILITY_CONTINUATION_HEADERS) $(COMMON_FILE_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DKS2D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(INMFD_LINSOLVERS) $(IBOOST) source/models/KS_2D/KS2D_stability.cpp $(OPENMP) $(LFFTW) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS2D_stability_cpu_omp.bin 2>$(RESULTS)

KS2D_stability_cuda: source/models/KS_2D/KS2D_stability.cpp source/models/KS_2D/KS2D_stability_cli.h $(KS2D_REFACTORED_HEADERS) $(STABILITY_CONTINUATION_HEADERS) $(COMMON_FILE_OPS_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS2D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) $(IBOOST) -x cu source/models/KS_2D/KS2D_stability.cpp -c -o $(BUILD_DIR)/KS2D_stability_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/KS2D_stability_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS2D_stability_cuda.bin 2>$(RESULTS)

test_lapack_wrap.bin: source/external_libraries/tests/test_lapack_wrap.cpp source/contrib/scfd/include/scfd/external_libraries/lapack_wrap.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/external_libraries/tests/test_lapack_wrap.cpp $(LLAPACK) -o $(BUILD_DIR)/test_lapack_wrap.bin 2>$(RESULTS)

test_lapack_wrap_cuda.bin: source/external_libraries/tests/test_lapack_wrap_cuda.cu source/external_libraries/lapack_wrap.h source/contrib/scfd/include/scfd/external_libraries/lapack_wrap.h source/contrib/scfd/include/scfd/external_libraries/lapack_wrap_device.h source/common/cuda_init_scfd.h
	$(NVCC) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/external_libraries/tests/test_lapack_wrap_cuda.cu $(LLAPACK) -o $(BUILD_DIR)/test_lapack_wrap_cuda.bin 2>$(RESULTS)

test_small_dense_linalg.bin: source/common/tests/test_small_dense_linalg.cpp $(SMALL_DENSE_LINALG_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/common/tests/test_small_dense_linalg.cpp -o $(BUILD_DIR)/test_small_dense_linalg.bin 2>$(RESULTS)

test_host_small_dense_backend.bin: source/common/tests/test_host_small_dense_backend.cpp $(HOST_SMALL_DENSE_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/common/tests/test_host_small_dense_backend.cpp $(LLAPACK) -o $(BUILD_DIR)/test_host_small_dense_backend.bin 2>$(RESULTS)

test_matrix_market.bin: source/common/tests/test_matrix_market.cpp $(MATRIX_MARKET_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/common/tests/test_matrix_market.cpp $(OPENMP) -o $(BUILD_DIR)/test_matrix_market.bin 2>$(RESULTS)

test_matrix_market_report_data_cpu_omp.bin: source/stability/tests/test_matrix_market_report_data.cpp source/stability/tests/common/host_csr_operator.h $(MATRIX_MARKET_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_matrix_market_report_data.cpp $(OPENMP) -o $(BUILD_DIR)/test_matrix_market_report_data_cpu_omp.bin 2>$(RESULTS)

test_analytical_eigenproblems.bin: source/stability/tests/test_analytical_eigenproblems.cpp $(ANALYTICAL_EIGENPROBLEM_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_analytical_eigenproblems.cpp -o $(BUILD_DIR)/test_analytical_eigenproblems.bin 2>$(RESULTS)

test_stability_model_adapter_contract.bin: source/stability/tests/test_model_adapter_contract.cpp source/stability/model_adapter_contract.h source/stability/eigensolvers/eigensolver_result.h source/stability/eigensolvers/transformations/affine_inverse_health.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_model_adapter_contract.cpp -o $(BUILD_DIR)/test_stability_model_adapter_contract.bin 2>$(RESULTS)

test_direct_and_dense_eigensolvers_cpu_omp.bin: source/stability/tests/test_direct_and_dense_eigensolvers.cpp source/stability/eigensolvers/direct_scalar_eigensolver.h source/stability/eigensolvers/host_dense_operator_eigensolver.h source/stability/analysis/dimension_guarded_eigensolver.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_direct_and_dense_eigensolvers.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_direct_and_dense_eigensolvers_cpu_omp.bin 2>$(RESULTS)

test_scalar_affine_inverse_cpu_omp.bin: source/models/tests/test_scalar_affine_inverse.cpp $(CIRCLE_MODEL_HEADERS) $(STAR_SHAPED_MODEL_HEADERS) $(BRATU_MODEL_HEADERS) $(ADJOINT_JACOBIAN_TEST_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/tests/test_scalar_affine_inverse.cpp $(OPENMP) -o $(BUILD_DIR)/test_scalar_affine_inverse_cpu_omp.bin 2>$(RESULTS)

test_scalar_affine_inverse_cuda.bin: source/models/tests/test_scalar_affine_inverse.cpp $(CIRCLE_MODEL_HEADERS) $(STAR_SHAPED_MODEL_HEADERS) $(BRATU_MODEL_HEADERS) $(ADJOINT_JACOBIAN_TEST_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DTEST_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/tests/test_scalar_affine_inverse.cpp -c -o $(BUILD_DIR)/test_scalar_affine_inverse_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_scalar_affine_inverse_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_scalar_affine_inverse_cuda.bin 2>$(RESULTS)

test_bratu_stability_operators_cpu_omp.bin: source/models/bratu/test_bratu_stability_operators.cpp $(BRATU_MODEL_HEADERS) $(RESIDUAL_DECOMPOSITION_TEST_HEADER) $(ADJOINT_JACOBIAN_TEST_HEADERS) source/stability/eigensolvers/host_dense_operator_eigensolver.h $(HOST_SMALL_DENSE_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(INMFD_LINSOLVERS) source/models/bratu/test_bratu_stability_operators.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_bratu_stability_operators_cpu_omp.bin 2>$(RESULTS)

test_stability_analysis_pipeline_cpu_omp.bin: source/stability/tests/test_stability_analysis_pipeline.cpp $(STABILITY_ANALYSIS_HEADERS) $(CIRCLE_MODEL_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_stability_analysis_pipeline.cpp $(OPENMP) -o $(BUILD_DIR)/test_stability_analysis_pipeline_cpu_omp.bin 2>$(RESULTS)

test_source_parameter_path.bin: source/stability/tests/test_source_parameter_path.cpp source/stability/analysis/source_parameter_path.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_source_parameter_path.cpp -o $(BUILD_DIR)/test_source_parameter_path.bin 2>$(RESULTS)

test_source_parameter_state_marcher.bin: source/stability/tests/test_source_parameter_state_marcher.cpp source/stability/analysis/source_parameter_state_marcher.h source/stability/analysis/source_parameter_path.h source/stability/analysis/transition_state_alignment.h source/stability/analysis/detail/vector_workspace.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_source_parameter_state_marcher.cpp -o $(BUILD_DIR)/test_source_parameter_state_marcher.bin 2>$(RESULTS)

test_stability_analysis_pipeline_cuda.bin: source/stability/tests/test_stability_analysis_pipeline_cuda.cu source/common/cuda_init_scfd.h $(STABILITY_ANALYSIS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_stability_analysis_pipeline_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_stability_analysis_pipeline_cuda.bin 2>$(RESULTS)

test_classification_uncertainty_registry.bin: source/stability/tests/test_classification_uncertainty_registry.cpp source/stability/persistence/classification_uncertainty_registry.h source/stability/analysis/stability_point_result.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_classification_uncertainty_registry.cpp -o $(BUILD_DIR)/test_classification_uncertainty_registry.bin 2>$(RESULTS)

test_spectrum_scan_aggregator.bin: source/stability/tests/test_spectrum_scan_aggregator.cpp source/stability/analysis/spectrum_scan_aggregator.h source/stability/analysis/validated_spectrum_union.h source/stability/analysis/spectrum_classifier.h source/stability/eigensolvers/eigensolver_result.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/stability/tests/test_spectrum_scan_aggregator.cpp -o $(BUILD_DIR)/test_spectrum_scan_aggregator.bin 2>$(RESULTS)

test_eigenvector_rank_aggregator_cpu_omp.bin: source/stability/tests/test_eigenvector_rank_aggregator.cpp source/stability/tests/common/recycled_ritz_subspace_test_suite.h source/stability/analysis/eigenvector_rank_aggregator.h source/stability/analysis/recycled_ritz_subspace.h source/stability/analysis/detail/vector_workspace.h source/stability/eigensolvers/ritz_recovery.h source/stability/eigensolvers/eigensolver_result.h $(SCFD_SERIAL_VECTOR_OPS_HEADERS) $(NMFD_KRYLOV_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_eigenvector_rank_aggregator.cpp $(OPENMP) -o $(BUILD_DIR)/test_eigenvector_rank_aggregator_cpu_omp.bin 2>$(RESULTS)

test_tracked_invariant_subspace_cpu_omp.bin: source/stability/tests/test_tracked_invariant_subspace.cpp source/stability/tests/common/tracked_invariant_subspace_test_suite.h source/stability/tracking/principal_angles.h source/stability/tracking/tracked_invariant_subspace.h source/stability/analysis/eigenvector_rank_aggregator.h source/stability/analysis/detail/vector_workspace.h source/stability/eigensolvers/ritz_recovery.h source/stability/eigensolvers/eigensolver_result.h $(SCFD_SERIAL_VECTOR_OPS_HEADERS) $(NMFD_KRYLOV_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_tracked_invariant_subspace.cpp $(OPENMP) -o $(BUILD_DIR)/test_tracked_invariant_subspace_cpu_omp.bin 2>$(RESULTS)

test_tracked_invariant_subspace_cuda.bin: source/stability/tests/test_tracked_invariant_subspace_cuda.cu source/stability/tests/common/tracked_invariant_subspace_test_suite.h source/stability/tracking/principal_angles.h source/stability/tracking/tracked_invariant_subspace.h source/stability/analysis/eigenvector_rank_aggregator.h source/stability/analysis/detail/vector_workspace.h source/stability/eigensolvers/ritz_recovery.h source/stability/eigensolvers/eigensolver_result.h source/common/cuda_init_scfd.h $(SCFD_VECTOR_OPS_HEADERS) $(NMFD_KRYLOV_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_tracked_invariant_subspace_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_tracked_invariant_subspace_cuda.bin 2>$(RESULTS)

test_matrix_free_stability_configuration.bin: source/stability/tests/test_matrix_free_stability_configuration.cpp source/stability/analysis/matrix_free_stability_config.h source/stability/analysis/matrix_free_stability_configuration.h $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(NMFD_KRYLOV_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_matrix_free_stability_configuration.cpp -o $(BUILD_DIR)/test_matrix_free_stability_configuration.bin 2>$(RESULTS)

test_stability_continuation_interface_cpu.bin: source/stability/tests/test_stability_continuation_interface.cpp source/main/stability_continuation.hpp source/main/parameters.hpp $(STABILITY_ANALYSIS_HEADERS) $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(IBOOST) source/stability/tests/test_stability_continuation_interface.cpp $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/test_stability_continuation_interface_cpu.bin 2>$(RESULTS)

test_shared_arnoldi_cpu_omp.bin: source/stability/tests/test_shared_arnoldi.cpp $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_shared_arnoldi.cpp $(OPENMP) -o $(BUILD_DIR)/test_shared_arnoldi_cpu_omp.bin 2>$(RESULTS)

test_inverse_iteration_cpu_omp.bin: source/stability/tests/test_inverse_iteration.cpp source/stability/tests/common/analytical_shifted_solver.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_HEADERS) $(NMFD_KRYLOV_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_inverse_iteration.cpp $(OPENMP) -o $(BUILD_DIR)/test_inverse_iteration_cpu_omp.bin 2>$(RESULTS)

test_krylov_schur_cpu_omp.bin: source/stability/tests/test_krylov_schur.cpp $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_krylov_schur.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_krylov_schur_cpu_omp.bin 2>$(RESULTS)

test_projected_spectrum_recovery_cpu_omp.bin: source/stability/tests/test_projected_spectrum_recovery.cpp source/stability/tests/common/projected_spectrum_recovery_test_suite.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_projected_spectrum_recovery.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_projected_spectrum_recovery_cpu_omp.bin 2>$(RESULTS)

test_projected_spectrum_recovery_cuda.bin: source/stability/tests/test_projected_spectrum_recovery_cuda.cu source/stability/tests/common/projected_spectrum_recovery_test_suite.h source/common/cuda_init_scfd.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_projected_spectrum_recovery_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) $(LLAPACK) -o $(BUILD_DIR)/test_projected_spectrum_recovery_cuda.bin 2>$(RESULTS)

test_complex_affine_transformations_cpu_omp.bin: source/stability/tests/test_complex_affine_transformations.cpp source/stability/tests/common/complex_affine_transformations_test_suite.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_KRYLOV_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_complex_affine_transformations.cpp $(OPENMP) -o $(BUILD_DIR)/test_complex_affine_transformations_cpu_omp.bin 2>$(RESULTS)

test_complex_affine_transformations_cuda.bin: source/stability/tests/test_complex_affine_transformations_cuda.cu source/stability/tests/common/complex_affine_transformations_test_suite.h source/common/cuda_init_scfd.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_KRYLOV_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_complex_affine_transformations_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_complex_affine_transformations_cuda.bin 2>$(RESULTS)

test_iterative_factor_solver_bundle_cpu_omp.bin: source/stability/tests/test_iterative_factor_solver_bundle.cpp source/stability/tests/common/iterative_factor_solver_bundle_test_suite.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_iterative_factor_solver_bundle.cpp $(OPENMP) -o $(BUILD_DIR)/test_iterative_factor_solver_bundle_cpu_omp.bin 2>$(RESULTS)

test_iterative_factor_solver_bundle_cuda.bin: source/stability/tests/test_iterative_factor_solver_bundle_cuda.cu source/stability/tests/common/iterative_factor_solver_bundle_test_suite.h source/common/cuda_init_scfd.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_iterative_factor_solver_bundle_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_iterative_factor_solver_bundle_cuda.bin 2>$(RESULTS)

test_matrix_free_complexification_cpu_omp.bin: source/stability/tests/test_matrix_free_complexification.cpp source/stability/tests/common/matrix_free_complexification_test_suite.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_matrix_free_complexification.cpp $(OPENMP) -o $(BUILD_DIR)/test_matrix_free_complexification_cpu_omp.bin 2>$(RESULTS)

test_matrix_free_complexification_cuda.bin: source/stability/tests/test_matrix_free_complexification_cuda.cu source/stability/tests/common/matrix_free_complexification_test_suite.h source/common/cuda_init_scfd.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_matrix_free_complexification_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_matrix_free_complexification_cuda.bin 2>$(RESULTS)

test_matrix_free_factorized_krylov_schur_cpu_omp.bin: source/stability/tests/test_matrix_free_factorized_krylov_schur.cpp source/stability/tests/common/matrix_free_factorized_krylov_schur_test_suite.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(STABILITY_ANALYSIS_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_matrix_free_factorized_krylov_schur.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_matrix_free_factorized_krylov_schur_cpu_omp.bin 2>$(RESULTS)

test_matrix_free_factorized_krylov_schur_cuda.bin: source/stability/tests/test_matrix_free_factorized_krylov_schur_cuda.cu source/stability/tests/common/matrix_free_factorized_krylov_schur_test_suite.h source/common/cuda_init_scfd.h $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(STABILITY_ANALYSIS_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_matrix_free_factorized_krylov_schur_cuda.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) $(LLAPACK) -o $(BUILD_DIR)/test_matrix_free_factorized_krylov_schur_cuda.bin 2>$(RESULTS)

test_krylov_schur_dat_A_cpu_omp.bin: source/stability/tests/test_krylov_schur_dat_A.cpp $(EIGENSOLVER_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_krylov_schur_dat_A.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_krylov_schur_dat_A_cpu_omp.bin 2>$(RESULTS)

test_krylov_schur_rdb450_cpu_omp.bin: source/stability/tests/test_krylov_schur_rdb450.cpp source/stability/tests/common/host_csr_operator.h $(MATRIX_MARKET_HEADERS) $(EIGENSOLVER_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_krylov_schur_rdb450.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_krylov_schur_rdb450_cpu_omp.bin 2>$(RESULTS)

test_transformed_krylov_schur_rdb450_cpu_omp.bin: source/stability/tests/test_transformed_krylov_schur_rdb450.cpp source/stability/tests/common/host_csr_operator.h $(MATRIX_MARKET_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_transformed_krylov_schur_rdb450.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_transformed_krylov_schur_rdb450_cpu_omp.bin 2>$(RESULTS)

characterize_krylov_schur_S80PI_n1_cpu_omp.bin: source/stability/tests/characterize_krylov_schur_S80PI_n1.cpp source/stability/tests/common/host_csr_operator.h $(MATRIX_MARKET_HEADERS) $(EIGENSOLVER_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/characterize_krylov_schur_S80PI_n1.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/characterize_krylov_schur_S80PI_n1_cpu_omp.bin 2>$(RESULTS)

characterize_transformed_krylov_schur_S80PI_n1_cpu_omp.bin: source/stability/tests/characterize_transformed_krylov_schur_S80PI_n1.cpp source/stability/tests/common/host_csr_operator.h $(MATRIX_MARKET_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/characterize_transformed_krylov_schur_S80PI_n1.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/characterize_transformed_krylov_schur_S80PI_n1_cpu_omp.bin 2>$(RESULTS)

characterize_transformed_krylov_schur_S80PI_n1_exact_cpu_omp.bin: source/stability/tests/characterize_transformed_krylov_schur_S80PI_n1_exact.cpp source/stability/tests/common/host_csr_operator.h source/stability/tests/common/umfpack_complex_affine_solver.h $(MATRIX_MARKET_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) $(ISUITESPARSE) source/stability/tests/characterize_transformed_krylov_schur_S80PI_n1_exact.cpp $(OPENMP) $(LLAPACK) $(LUMFPACK) -o $(BUILD_DIR)/characterize_transformed_krylov_schur_S80PI_n1_exact_cpu_omp.bin 2>$(RESULTS)

characterize_factorized_inverse_S80PI_n1_cpu_omp.bin: source/stability/tests/characterize_factorized_inverse_S80PI_n1.cpp source/stability/tests/common/host_csr_operator.h source/stability/tests/common/umfpack_complex_affine_solver.h $(MATRIX_MARKET_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) $(ISUITESPARSE) source/stability/tests/characterize_factorized_inverse_S80PI_n1.cpp $(OPENMP) $(LUMFPACK) -o $(BUILD_DIR)/characterize_factorized_inverse_S80PI_n1_cpu_omp.bin 2>$(RESULTS)

characterize_factorized_krylov_schur_S80PI_n1_cpu_omp.bin: source/stability/tests/characterize_factorized_krylov_schur_S80PI_n1.cpp source/stability/tests/common/host_csr_operator.h $(MATRIX_MARKET_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/characterize_factorized_krylov_schur_S80PI_n1.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/characterize_factorized_krylov_schur_S80PI_n1_cpu_omp.bin 2>$(RESULTS)

test_transformed_operators_cpu_omp.bin: source/stability/tests/test_transformed_operators.cpp source/stability/tests/common/analytical_matrix_solver.h $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_KRYLOV_HEADERS) $(HOST_SMALL_DENSE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_transformed_operators.cpp $(OPENMP) $(LLAPACK) -o $(BUILD_DIR)/test_transformed_operators_cpu_omp.bin 2>$(RESULTS)

test_product_vector_space_cpu_omp.bin: source/common/tests/test_product_vector_space.cpp $(PRODUCT_VECTOR_SPACE_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/common/tests/test_product_vector_space.cpp $(OPENMP) -o $(BUILD_DIR)/test_product_vector_space_cpu_omp.bin 2>$(RESULTS)

test_transformed_operators_gmres_cpu_omp.bin: source/stability/tests/test_transformed_operators_gmres.cpp $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_GMRES_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_transformed_operators_gmres.cpp $(OPENMP) -o $(BUILD_DIR)/test_transformed_operators_gmres_cpu_omp.bin 2>$(RESULTS)

test_factorized_inverse_solver_cpu_omp.bin: source/stability/tests/test_factorized_inverse_solver.cpp $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_KRYLOV_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_factorized_inverse_solver.cpp $(OPENMP) -o $(BUILD_DIR)/test_factorized_inverse_solver_cpu_omp.bin 2>$(RESULTS)

test_stability_polynomial_operator_cpu_omp.bin: source/stability/tests/test_stability_polynomial_operator.cpp $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(NMFD_KRYLOV_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_stability_polynomial_operator.cpp $(OPENMP) -o $(BUILD_DIR)/test_stability_polynomial_operator_cpu_omp.bin 2>$(RESULTS)

test_complex_solver_product_adapter_cpu_omp.bin: source/stability/tests/test_complex_solver_product_adapter.cpp $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(PRODUCT_VECTOR_SPACE_HEADERS) $(NMFD_KRYLOV_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/stability/tests/test_complex_solver_product_adapter.cpp $(OPENMP) -o $(BUILD_DIR)/test_complex_solver_product_adapter_cpu_omp.bin 2>$(RESULTS)

test_nmfd_gmres_shared_krylov_cpu_omp.bin: source/numerical_algos/lin_solvers/tests/test_nmfd_gmres_shared_krylov.cpp $(ANALYTICAL_EIGENPROBLEM_HEADERS) $(NMFD_GMRES_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/numerical_algos/lin_solvers/tests/test_nmfd_gmres_shared_krylov.cpp $(OPENMP) -o $(BUILD_DIR)/test_nmfd_gmres_shared_krylov_cpu_omp.bin 2>$(RESULTS)

test_nmfd_gmres_baseline_cpu.bin: source/numerical_algos/lin_solvers/tests/test_nmfd_gmres_baseline.cpp $(NMFD_GMRES_HEADERS) $(NMFD_GMRES_BASELINE_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(INMFD_LINSOLVERS) source/numerical_algos/lin_solvers/tests/test_nmfd_gmres_baseline.cpp -o $(BUILD_DIR)/test_nmfd_gmres_baseline_cpu.bin 2>$(RESULTS)

test_complex_traits.bin: source/common/tests/test_complex_traits.cpp $(COMPLEX_TRAITS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/common/tests/test_complex_traits.cpp -o $(BUILD_DIR)/test_complex_traits.bin 2>$(RESULTS)

test_continuation_chart_helpers.bin: source/continuation/tests/test_chart_helpers.cpp $(CONTINUATION_CHART_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_chart_helpers.cpp -o $(BUILD_DIR)/test_continuation_chart_helpers.bin 2>$(RESULTS)

test_predictor_chart_validation.bin: source/continuation/tests/test_predictor_chart_validation.cpp $(CONTINUATION_CHART_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_predictor_chart_validation.cpp -o $(BUILD_DIR)/test_predictor_chart_validation.bin 2>$(RESULTS)

test_predictor_chart_probe.bin: source/continuation/tests/test_predictor_chart_probe.cpp $(CONTINUATION_CHART_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_predictor_chart_probe.cpp -o $(BUILD_DIR)/test_predictor_chart_probe.bin 2>$(RESULTS)

test_advance_solution_chart_retry.bin: source/continuation/tests/test_advance_solution_chart_retry.cpp source/continuation/advance_solution.h source/continuation/predictor_adaptive.h $(CONTINUATION_CHART_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_advance_solution_chart_retry.cpp -o $(BUILD_DIR)/test_advance_solution_chart_retry.bin 2>$(RESULTS)

test_continuation_retry_progress.bin: source/continuation/tests/test_retry_and_progress_policy.cpp source/continuation/corrector_retry_policy.h source/continuation/predictor_adaptive.h source/continuation/progress_monitor.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_retry_and_progress_policy.cpp -o $(BUILD_DIR)/test_continuation_retry_progress.bin 2>$(RESULTS)

test_initial_tangent_components.bin: source/continuation/tests/test_initial_tangent_components.cpp source/continuation/initial_tangent_candidates.h source/continuation/initial_tangent_chart_validator.h source/continuation/initial_tangent_secant_builder.h source/continuation/semicurve_tangent_cache.h source/continuation/tangent_normalization.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_initial_tangent_components.cpp -o $(BUILD_DIR)/test_initial_tangent_components.bin 2>$(RESULTS)

test_continuation_event_state.bin: source/continuation/tests/test_continuation_event_state.cpp source/continuation/continuation_endpoint_state.h source/continuation/pending_branch_event.h source/containers/curve_endpoint_reason.h source/containers/bifurcation_diagram/curve_provenance.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_continuation_event_state.cpp -o $(BUILD_DIR)/test_continuation_event_state.bin 2>$(RESULTS)

test_observational_knot_sample.bin: source/continuation/tests/test_observational_knot_sample.cpp source/continuation/observational_knot_sample.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/continuation/tests/test_observational_knot_sample.cpp -o $(BUILD_DIR)/test_observational_knot_sample.bin 2>$(RESULTS)

test_branch_intersection_policy.bin: source/containers/tests/test_branch_intersection_policy.cpp source/containers/branch_intersection.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_branch_intersection_policy.cpp -o $(BUILD_DIR)/test_branch_intersection_policy.bin 2>$(RESULTS)

test_symmetry_event_persistence.bin: source/containers/tests/test_symmetry_event_persistence.cpp source/containers/symmetry_event_record.h source/containers/symmetry_event_journal.h source/containers/symmetry_event_registry.h source/symmetry/translation/orbit_type.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_symmetry_event_persistence.cpp -o $(BUILD_DIR)/test_symmetry_event_persistence.bin 2>$(RESULTS)

test_parameters_json.bin: source/main/tests/test_parameters_json.cpp source/main/parameters.hpp
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_parameters_json.cpp -o $(BUILD_DIR)/test_parameters_json.bin 2>$(RESULTS)

test_solver_bundle.bin: source/main/tests/test_solver_bundle.cpp source/main/deflation_continuation/solver_bundle.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_solver_bundle.cpp -o $(BUILD_DIR)/test_solver_bundle.bin 2>$(RESULTS)

test_parameter_application.bin: source/main/tests/test_parameter_application.cpp source/main/deflation_continuation/parameter_application.h source/main/parameters.hpp
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_parameter_application.cpp -o $(BUILD_DIR)/test_parameter_application.bin 2>$(RESULTS)

test_rejected_candidate_cache.bin: source/main/tests/test_rejected_candidate_cache.cpp source/main/deflation_continuation/rejected_candidate_cache.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_rejected_candidate_cache.cpp -o $(BUILD_DIR)/test_rejected_candidate_cache.bin 2>$(RESULTS)

test_continuation_registries.bin: source/main/tests/test_continuation_registries.cpp source/containers/failed_continuation_registry.h source/main/deflation_continuation/continuation_recovery_registry.h source/continuation/continuation_result.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_continuation_registries.cpp -o $(BUILD_DIR)/test_continuation_registries.bin 2>$(RESULTS)

test_branch_topology_registry.bin: source/containers/tests/test_branch_topology_registry.cpp source/containers/bifurcation_diagram/topology/branch_topology_registry.h source/containers/curve_endpoint_reason.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_branch_topology_registry.cpp -o $(BUILD_DIR)/test_branch_topology_registry.bin 2>$(RESULTS)

test_residual_translation_orbit_aligner_2d.bin: source/symmetry/tests/test_residual_translation_orbit_aligner_2d.cpp source/symmetry/fourier/residual_translation_orbit_aligner_2d.h source/symmetry/fourier/residual_translation_group_2d.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_residual_translation_orbit_aligner_2d.cpp -o $(BUILD_DIR)/test_residual_translation_orbit_aligner_2d.bin 2>$(RESULTS)

test_exact_solution_registry.bin: source/main/tests/test_exact_solution_registry.cpp source/main/deflation_continuation/exact_solution_registry.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_exact_solution_registry.cpp -o $(BUILD_DIR)/test_exact_solution_registry.bin 2>$(RESULTS)

test_knot_relocation_controller.bin: source/main/tests/test_knot_relocation_controller.cpp source/main/deflation_continuation/knot_relocation_controller.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_knot_relocation_controller.cpp -o $(BUILD_DIR)/test_knot_relocation_controller.bin 2>$(RESULTS)

test_analytical_branch_executor.bin: source/main/tests/test_analytical_branch_executor.cpp source/main/deflation_continuation/analytical_branch_executor.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_analytical_branch_executor.cpp -o $(BUILD_DIR)/test_analytical_branch_executor.bin 2>$(RESULTS)

test_knot_executor.bin: source/main/tests/test_knot_executor.cpp source/main/deflation_continuation/knot_executor.h source/main/deflation_continuation/rejected_candidate_cache.h source/containers/knots.hpp
	$(G++) $(G++FLAGS) $(IPROJECT) source/main/tests/test_knot_executor.cpp -o $(BUILD_DIR)/test_knot_executor.bin 2>$(RESULTS)

test_deflation_seed_registry.bin: source/containers/tests/test_deflation_seed_registry.cpp source/containers/deflation_seed_registry.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_deflation_seed_registry.cpp -o $(BUILD_DIR)/test_deflation_seed_registry.bin 2>$(RESULTS)

test_linear_solve_recovery.bin: source/numerical_algos/lin_solvers/tests/test_linear_solve_recovery.cpp source/numerical_algos/lin_solvers/linear_solve_recovery.h source/numerical_algos/lin_solvers/iter_solver_base.h source/numerical_algos/lin_solvers/default_monitor.h source/numerical_algos/lin_solvers/tests/nmfd_cpu_reference_vector_space.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/numerical_algos/lin_solvers/tests/test_linear_solve_recovery.cpp -o $(BUILD_DIR)/test_linear_solve_recovery.bin 2>$(RESULTS)

test_curve_metadata_io.bin: source/containers/tests/test_curve_metadata_io.cpp source/containers/bifurcation_diagram/curve_point.h source/containers/bifurcation_diagram/curve_metadata_io.h source/containers/curve_endpoint_reason.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_curve_metadata_io.cpp -o $(BUILD_DIR)/test_curve_metadata_io.bin 2>$(RESULTS)

test_curve_vector_store.bin: source/containers/tests/test_curve_vector_store.cpp source/containers/bifurcation_diagram/curve_vector_store.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_curve_vector_store.cpp -o $(BUILD_DIR)/test_curve_vector_store.bin 2>$(RESULTS)

test_curve_interpolator.bin: source/containers/tests/test_curve_interpolator.cpp source/containers/bifurcation_diagram/curve_interpolator.h source/containers/bifurcation_diagram/curve_point.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_curve_interpolator.cpp -o $(BUILD_DIR)/test_curve_interpolator.bin 2>$(RESULTS)

test_curve_intersection_search.bin: source/containers/tests/test_curve_intersection_search.cpp source/containers/bifurcation_diagram/curve_intersection_search.h source/containers/bifurcation_diagram/curve_point.h source/containers/branch_intersection.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_curve_intersection_search.cpp -o $(BUILD_DIR)/test_curve_intersection_search.bin 2>$(RESULTS)

test_curve_provenance.bin: source/containers/tests/test_curve_provenance.cpp source/containers/bifurcation_diagram/curve_provenance.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_curve_provenance.cpp -o $(BUILD_DIR)/test_curve_provenance.bin 2>$(RESULTS)

test_diagram_archive.bin: source/containers/tests/test_diagram_archive.cpp source/containers/bifurcation_diagram/diagram_archive.h
	$(G++) $(G++FLAGS) $(IPROJECT) $(IBOOST) source/containers/tests/test_diagram_archive.cpp $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/test_diagram_archive.bin 2>$(RESULTS)

test_stability_diagram_archive.bin: source/containers/tests/test_stability_diagram_archive.cpp source/containers/stability_diagram.h source/containers/bifurcation_diagram/diagram_archive.h
	$(G++) $(G++FLAGS) $(IPROJECT) $(IBOOST) source/containers/tests/test_stability_diagram_archive.cpp $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/test_stability_diagram_archive.bin 2>$(RESULTS)

test_symmetry_event_registry_sync.bin: source/containers/tests/test_symmetry_event_registry_sync.cpp source/containers/bifurcation_diagram/symmetry_event_registry_sync.h source/containers/symmetry_event_registry.h source/containers/symmetry_event_record.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/containers/tests/test_symmetry_event_registry_sync.cpp -o $(BUILD_DIR)/test_symmetry_event_registry_sync.bin 2>$(RESULTS)

test_fourier_mode_primitives.bin: source/symmetry/tests/test_fourier_mode_primitives.cpp $(SYMMETRY_FOURIER_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_fourier_mode_primitives.cpp -o $(BUILD_DIR)/test_fourier_mode_primitives.bin 2>$(RESULTS)

test_quotient_classifier.bin: source/symmetry/tests/test_quotient_classifier.cpp $(SYMMETRY_CORE_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_quotient_classifier.cpp -o $(BUILD_DIR)/test_quotient_classifier.bin 2>$(RESULTS)

test_fourier_slice_1d.bin: source/symmetry/tests/test_fourier_slice_1d.cpp $(SYMMETRY_FOURIER_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_fourier_slice_1d.cpp -o $(BUILD_DIR)/test_fourier_slice_1d.bin 2>$(RESULTS)

test_real_packed_fourier_canonical_adapter.bin: source/symmetry/tests/test_real_packed_fourier_canonical_adapter.cpp $(SYMMETRY_FOURIER_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_real_packed_fourier_canonical_adapter.cpp $(OPENMP) -o $(BUILD_DIR)/test_real_packed_fourier_canonical_adapter.bin 2>$(RESULTS)

test_fourier_isotropy_transition.bin: source/symmetry/tests/test_fourier_isotropy_transition.cpp $(SYMMETRY_FOURIER_HEADERS) $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_fourier_isotropy_transition.cpp $(OPENMP) -o $(BUILD_DIR)/test_fourier_isotropy_transition.bin 2>$(RESULTS)

test_fourier_slice_differential_1d.bin: source/symmetry/tests/test_fourier_slice_differential_1d.cpp $(SYMMETRY_FOURIER_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_fourier_slice_differential_1d.cpp -o $(BUILD_DIR)/test_fourier_slice_differential_1d.bin 2>$(RESULTS)

test_frozen_fourier_chart_1d.bin: source/symmetry/tests/test_frozen_fourier_chart_1d.cpp $(SYMMETRY_FOURIER_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_frozen_fourier_chart_1d.cpp -o $(BUILD_DIR)/test_frozen_fourier_chart_1d.bin 2>$(RESULTS)

test_stabilized_storage.bin: source/symmetry/tests/test_stabilized_storage.cpp $(SYMMETRY_CORE_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) source/symmetry/tests/test_stabilized_storage.cpp -o $(BUILD_DIR)/test_stabilized_storage.bin 2>$(RESULTS)

test_symmetry_solution_storage.bin: source/deflation/tests/test_symmetry_solution_storage.cpp $(DEFLATION_SYMMETRY_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/deflation/tests/test_symmetry_solution_storage.cpp $(OPENMP) -o $(BUILD_DIR)/test_symmetry_solution_storage.bin 2>$(RESULTS)


test_cpu_glued_vector_operations.bin: source/common/tests/test_cpu_glued_vector_operations.cpp $(SCFD_SERIAL_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/common/tests/test_cpu_glued_vector_operations.cpp $(OPENMP) -o $(BUILD_DIR)/test_cpu_glued_vector_operations.bin 2>$(RESULTS)

test_multi_precision.bin: source/test_inst/test_multi_precision/test_multi_precision.cpp $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) $(IPROJECT) $(IBOOST) $(ICUDA) source/test_inst/test_multi_precision/test_multi_precision.cpp -o $(BUILD_DIR)/test_multi_precision.bin $(OPENMP)


#all targets
lapack_test.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/tests/lapack.cpp $(LLAPACK) -o $(BUILD_DIR)/test_lapack.bin

iram_bulge_test.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/IRAM/predefined_matrix.cpp $(LIBS1) $(LLAPACK) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o -o $(BUILD_DIR)/test_IRAM.bin

iram_simulation.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/IRAM/iram_simulation.cpp $(LIBS1) $(LLAPACK) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o -o $(BUILD_DIR)/test_IRAM.bin

iram_process.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/IRAM/iram_test.cpp $(LCUDA) $(LIBSAll) $(LLAPACK) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o -o $(BUILD_DIR)/test_IRAM.bin

arnoldi_iters.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/Arnoldi_Iterations/arnoldi_iteraitons_test.cpp $(LIBS1) $(LLAPACK) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o -o $(BUILD_DIR)/test_ArIter.bin


cpu_1st_call_alloc_test.bin: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o $(BUILD_DIR)/cpu_1st_call_alloc_test_float.bin

sm_test.bin: source/numerical_algos/lin_solvers/tests/sm_test.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/numerical_algos/lin_solvers/tests/sm_test.cpp -o $(BUILD_DIR)/sm_test.bin 2>$(RESULTS) $(OPENMP)

iterative_solvers_test.bin: source/numerical_algos/lin_solvers/tests/iterative_solvers_test.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/numerical_algos/lin_solvers/tests/iterative_solvers_test.cpp -o $(BUILD_DIR)/iterative_solvers_test.bin 2>$(RESULTS) $(OPENMP)

iterative_solvers_test_different_operators.bin: source/numerical_algos/lin_solvers/tests/iterative_solvers_test_different_operators.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/numerical_algos/lin_solvers/tests/iterative_solvers_test_different_operators.cpp -o $(BUILD_DIR)/iterative_solvers_test_different_operators.bin 2>$(RESULTS) $(OPENMP)


exact_linsolver_test.bin: source/numerical_algos/lin_solvers/tests/exact_solver_test.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/numerical_algos/lin_solvers/tests/exact_solver_test.cpp $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o $(BUILD_DIR)/exact_solver_test.bin 2>$(RESULTS)

lin_solvers_test.bin: source/models/tests/linear_solvers_test.cpp
	$(NVCC) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/tests/linear_solvers_test.cpp -o $(BUILD_DIR)/linear_solvers_test.bin 2>$(RESULTS)

cufft_test.bin: source/models/tests/cufft_test.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/models/tests/cufft_test_kernels.cu -c -o $(BUILD_DIR)/cufft_test_kernels.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -g source/models/tests/cufft_test.cpp $(BUILD_DIR)/cufft_test_kernels.o $(LIBS2) -o $(BUILD_DIR)/cufft_test.bin 2>$(RESULTS)

cufft_test_2D.bin: source/models/tests/cufft_test_2D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_kernels.cu -c -o $(BUILD_DIR)/cufft_test_kernels.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_2D.cpp $(BUILD_DIR)/cufft_test_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LIBS3) -o $(BUILD_DIR)/cufft_test_2D.bin 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_2D_1.cpp $(BUILD_DIR)/cufft_test_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LIBS3) -o $(BUILD_DIR)/cufft_test_2D_1.bin 2>$(RESULTS)

gpu_vector_operations.bin: source/common/tests/test_gpu_vector_operations.cu source/common/tests/vector_operations_template_tests.h $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/tests/test_gpu_vector_operations.cu $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_vector_operations.bin 2>$(RESULTS)

scfd_vector_operations.bin: scfd_vector_operations_cuda.bin

scfd_vector_operations_cuda.bin: source/common/tests/test_scfd_vector_operations.cu source/common/tests/vector_operations_template_tests.h $(SCFD_VECTOR_OPS_TEST_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/tests/test_scfd_vector_operations.cu $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_scfd_vector_operations_cuda.bin 2>$(RESULTS)

scfd_vector_operations_hip.bin: $(BUILD_DIR)/test_scfd_vector_operations_hip.o $(BUILD_DIR)/gpu_reduction_ogita_kernels_hip.o
	$(HIPCC) $(HIPFLAGS) $(BUILD_DIR)/test_scfd_vector_operations_hip.o $(BUILD_DIR)/gpu_reduction_ogita_kernels_hip.o -o $(BUILD_DIR)/test_scfd_vector_operations_hip.bin 2>$(RESULTS)

$(BUILD_DIR)/test_scfd_vector_operations_hip.o: source/common/tests/test_scfd_vector_operations_hip.cpp source/common/tests/vector_operations_template_tests.h source/common/hip_init_scfd.h $(SCFD_VECTOR_OPS_TEST_HEADERS) | $(BUILD_STAMP)
	$(HIPCC) $(HIPFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/common/tests/test_scfd_vector_operations_hip.cpp -c -o $(BUILD_DIR)/test_scfd_vector_operations_hip.o 2>$(RESULTS)

scfd_vector_operations_cpu.bin: source/common/tests/test_scfd_vector_operations_cpu.cpp source/common/tests/vector_operations_template_tests.h $(SCFD_VECTOR_OPS_TEST_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/common/tests/test_scfd_vector_operations_cpu.cpp $(OPENMP) -o $(BUILD_DIR)/test_scfd_vector_operations_cpu.bin 2>$(RESULTS)

test_gpu_reduction_ogita.bin: source/common/tests/test_gpu_reduction_ogita.cpp $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/common/tests/test_gpu_reduction_ogita.cpp $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LCUDA) -o $(BUILD_DIR)/test_gpu_reduction_ogita.bin 2>$(RESULTS)

test_vector_snapshot_queue.bin: source/common/tests/test_vector_snapshot_queue.cu $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) source/common/tests/test_vector_snapshot_queue.cu $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_vector_snapshot_queue.bin 2>$(RESULTS)

gpu_reduction_ogita_ker: $(BUILD_DIR)/gpu_reduction_ogita_kernels.o

$(BUILD_DIR)/gpu_reduction_ogita_kernels.o: source/common/NMFD-operations/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu $(HIGH_PRECISION_BLAS1_HEADERS) | $(BUILD_STAMP)
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/common/NMFD-operations/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu -c -o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o 2>$(RESULTS)

$(BUILD_DIR)/gpu_reduction_ogita_kernels_hip.o: source/common/NMFD-operations/nmfd/operations/blas1/high_precision/hip/gpu_reduction_ogita_kernels.cu $(HIGH_PRECISION_BLAS1_HEADERS) | $(BUILD_STAMP)
	$(HIPCC) $(HIPFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/common/NMFD-operations/nmfd/operations/blas1/high_precision/hip/gpu_reduction_ogita_kernels.cu -c -o $(BUILD_DIR)/gpu_reduction_ogita_kernels_hip.o 2>$(RESULTS)

gpu_vector_operations_ker: $(BUILD_DIR)/gpu_vector_operations_kernels.o

$(BUILD_DIR)/gpu_vector_operations_kernels.o: source/common/gpu_vector_operations_kernels.cu | $(BUILD_STAMP)
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/common/gpu_vector_operations_kernels.cu -c -o $(BUILD_DIR)/gpu_vector_operations_kernels.o 2>$(RESULTS)

gpu_matrix_vector_operations_ker: $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o

$(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o: source/common/gpu_matrix_vector_operations_kernels.cu | $(BUILD_STAMP)
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)   source/common/gpu_matrix_vector_operations_kernels.cu -c -o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o 2>$(RESULTS)
gpu_matrix_vector_operations.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/test_matrix_vector_operations.cpp $(LLAPACK) $(LIBS3) $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o -o $(BUILD_DIR)/test_matrix_vector_operations.bin 2>$(RESULTS)


Kuramoto_Sivashinskiy_2D_ker:
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)   source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o 2>$(RESULTS)

test_Kuramoto_Sivashinskiy_2D_RHS.bin: source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_RHS.cpp $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_Kuramoto_Sivashinskiy_2D_RHS.bin 2>$(RESULTS)

test_Kuramoto_Sivashinskiy_2D_Newton.bin: source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_Newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_Newton.cpp $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_Kuramoto_Sivashinskiy_2D_Newton.bin 2>$(RESULTS)


deflation_KS_2D_S: source/models/KS_2D/test_deflation_KS.cpp
	$(NVCC) -DHIGH_PREC=false $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_KS.cpp $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_deflation_S.bin 2>$(RESULTS)

deflation_KS_2D_H: source/models/KS_2D/test_deflation_KS.cpp
	$(NVCC) -DHIGH_PREC=true $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_KS.cpp $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_deflation_H.bin 2>$(RESULTS)



cont_def_KS_2D: source/models/KS_2D/test_deflation_continuation_KS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_continuation_KS.cpp $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_deflation_continuation.bin 2>$(RESULTS)

circle_ker: $(BUILD_DIR)/circle_ker.o

$(BUILD_DIR)/circle_ker.o: source/nonlinear_operators/circle/circle_ker.cu | $(BUILD_STAMP)
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/nonlinear_operators/circle/circle_ker.cu -c -o $(BUILD_DIR)/circle_ker.o 2>$(RESULTS)

cont_def_circle: cont_def_circle_cuda

cont_def_circle_cuda: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) source/common/cuda_init_scfd.h $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/circle/circle_test_deflation_continuation.cpp -c -o $(BUILD_DIR)/circle_test_deflation_continuation_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/circle_test_deflation_continuation_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/circle_test_deflation_continuation_cuda.bin 2>$(RESULTS)

cont_def_circle_cpu_omp: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/circle/circle_test_deflation_continuation.cpp $(OPENMP) -o $(BUILD_DIR)/circle_test_deflation_continuation_cpu_omp.bin 2>$(RESULTS)

cont_def_circle_hip: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) source/common/hip_init_scfd.h $(SCFD_VECTOR_OPS_HEADERS)
	$(HIPCC) $(HIPFLAGS) -DCIRCLE_VECTOR_BACKEND_HIP $(SCALAR_TYPE) $(IPROJECT) source/models/circle/circle_test_deflation_continuation.cpp -o $(BUILD_DIR)/circle_test_deflation_continuation_hip.bin 2>$(RESULTS)

cont_def_circle_var_prec: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/circle/circle_test_deflation_continuation.cpp $(OPENMP) -o $(BUILD_DIR)/circle_test_deflation_continuation_var_prec.bin 2>$(RESULTS)

circle_curve_container: circle_curve_container_cuda

circle_curve_container_cuda: source/models/circle/circle_test_curve_container.cpp source/models/circle/circle_test_curve_container.h source/models/circle/circle_test_deflation_continuation_typedefs.h source/nonlinear_operators/circle/circle.h $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/circle/circle_test_curve_container.cpp -c -o $(BUILD_DIR)/circle_test_curve_container_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/circle_test_curve_container_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/circle_test_curve_container_cuda.bin 2>$(RESULTS)

circle_bd: circle_bd_cuda

circle_bd_cuda: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/circle/circle_bd.cpp -c -o $(BUILD_DIR)/circle_bd_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/circle_bd_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd_cuda.bin 2>$(RESULTS)

circle_bd_cpu_omp: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/circle/circle_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd_cpu_omp.bin 2>$(RESULTS)

circle_bd_hip: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(HIPCC) $(HIPFLAGS) -DCIRCLE_VECTOR_BACKEND_HIP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/circle/circle_bd.cpp $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd_hip.bin 2>$(RESULTS)

circle_bd_var_prec: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/circle/circle_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd_var_prec.bin 2>$(RESULTS)

circle_stability_cpu_omp: source/models/circle/circle_stability.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(STABILITY_ANALYSIS_HEADERS) source/stability/eigensolvers/direct_scalar_eigensolver.h
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/circle/circle_stability.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_stability_cpu_omp.bin 2>$(RESULTS)

circle_stability_cuda: source/models/circle/circle_stability.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(STABILITY_ANALYSIS_HEADERS) source/stability/eigensolvers/direct_scalar_eigensolver.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/circle/circle_stability.cpp -c -o $(BUILD_DIR)/circle_stability_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/circle_stability_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_stability_cuda.bin 2>$(RESULTS)

circle_prepare_visualization_cpu_omp: source/models/circle/circle_prepare_visualization.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(VISUALIZATION_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/circle/circle_prepare_visualization.cpp $(OPENMP) -o $(BUILD_DIR)/circle_prepare_visualization_cpu_omp.bin 2>$(RESULTS)

bratu_bd_cpu_omp: source/models/bratu/bratu_bd.cpp $(BRATU_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DBRATU_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/bratu/bratu_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/bratu_bd_cpu_omp.bin 2>$(RESULTS)

bratu_bd_var_prec: source/models/bratu/bratu_bd.cpp $(BRATU_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DBRATU_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/bratu/bratu_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/bratu_bd_var_prec.bin 2>$(RESULTS)

bratu_prepare_visualization_cpu_omp: source/models/bratu/bratu_prepare_visualization.cpp $(BRATU_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(VISUALIZATION_HEADERS)
	$(G++) $(G++FLAGS) -DBRATU_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/bratu/bratu_prepare_visualization.cpp $(OPENMP) -o $(BUILD_DIR)/bratu_prepare_visualization_cpu_omp.bin 2>$(RESULTS)

bratu_stability_cpu_omp: source/models/bratu/bratu_stability.cpp $(BRATU_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(STABILITY_ANALYSIS_HEADERS) $(EIGENSOLVER_HEADERS) $(EIGENSOLVER_TRANSFORMATION_HEADERS) $(NMFD_GMRES_HEADERS) $(HOST_SMALL_DENSE_HEADERS)
	$(G++) $(G++FLAGS) -DBRATU_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(INMFD_LINSOLVERS) $(IBOOST) source/models/bratu/bratu_stability.cpp $(OPENMP) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/bratu_stability_cpu_omp.bin 2>$(RESULTS)

star_shaped_bd: star_shaped_bd_cuda

star_shaped_bd_cuda: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/star_shaped/star_shaped_bd.cpp -c -o $(BUILD_DIR)/star_shaped_bd_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/star_shaped_bd_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd_cuda.bin 2>$(RESULTS)

star_shaped_bd_cpu_omp: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/star_shaped/star_shaped_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd_cpu_omp.bin 2>$(RESULTS)

star_shaped_bd_hip: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(HIPCC) $(HIPFLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_HIP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/star_shaped/star_shaped_bd.cpp $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd_hip.bin 2>$(RESULTS)

star_shaped_bd_var_prec: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/star_shaped/star_shaped_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd_var_prec.bin 2>$(RESULTS)

star_shaped_stability_cpu_omp: source/models/star_shaped/star_shaped_stability.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(STABILITY_ANALYSIS_HEADERS) source/stability/eigensolvers/direct_scalar_eigensolver.h
	$(G++) $(G++FLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/star_shaped/star_shaped_stability.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_stability_cpu_omp.bin 2>$(RESULTS)

star_shaped_stability_cuda: source/models/star_shaped/star_shaped_stability.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(STABILITY_ANALYSIS_HEADERS) source/stability/eigensolvers/direct_scalar_eigensolver.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/star_shaped/star_shaped_stability.cpp -c -o $(BUILD_DIR)/star_shaped_stability_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/star_shaped_stability_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_stability_cuda.bin 2>$(RESULTS)

star_shaped_prepare_visualization_cpu_omp: source/models/star_shaped/star_shaped_prepare_visualization.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(VISUALIZATION_HEADERS)
	$(G++) $(G++FLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/star_shaped/star_shaped_prepare_visualization.cpp $(OPENMP) -o $(BUILD_DIR)/star_shaped_prepare_visualization_cpu_omp.bin 2>$(RESULTS)

KS1D_operator_cpu_omp.bin: source/models/KS_1D/test_KS1D_operator.cpp $(KS1D_MODEL_HEADERS) $(RESIDUAL_DECOMPOSITION_TEST_HEADER) $(ADJOINT_JACOBIAN_TEST_HEADERS) $(AFFINE_PRECONDITIONER_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/KS_1D/test_KS1D_operator.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/test_KS1D_operator_cpu_omp.bin 2>$(RESULTS)

KS1D_operator_cuda.bin: source/models/KS_1D/test_KS1D_operator.cpp $(KS1D_MODEL_HEADERS) $(RESIDUAL_DECOMPOSITION_TEST_HEADER) $(ADJOINT_JACOBIAN_TEST_HEADERS) $(AFFINE_PRECONDITIONER_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS1D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/KS_1D/test_KS1D_operator.cpp -c -o $(BUILD_DIR)/test_KS1D_operator_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_KS1D_operator_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_KS1D_operator_cuda.bin 2>$(RESULTS)

KS1D_full_operator_cpu_omp.bin: source/models/KS_1D/test_KS1D_full_operator.cpp $(KS1D_FULL_MODEL_HEADERS) $(RESIDUAL_DECOMPOSITION_TEST_HEADER) $(ADJOINT_JACOBIAN_TEST_HEADERS) $(AFFINE_PRECONDITIONER_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/KS_1D/test_KS1D_full_operator.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/test_KS1D_full_operator_cpu_omp.bin 2>$(RESULTS)

KS1D_full_operator_cuda.bin: source/models/KS_1D/test_KS1D_full_operator.cpp $(KS1D_FULL_MODEL_HEADERS) $(RESIDUAL_DECOMPOSITION_TEST_HEADER) $(ADJOINT_JACOBIAN_TEST_HEADERS) $(AFFINE_PRECONDITIONER_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS1D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/KS_1D/test_KS1D_full_operator.cpp -c -o $(BUILD_DIR)/test_KS1D_full_operator_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_KS1D_full_operator_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_KS1D_full_operator_cuda.bin 2>$(RESULTS)

KS1D_stability_scan_cpu_omp.bin: source/models/KS_1D/test_KS1D_stability_scan.cpp $(KS1D_FULL_MODEL_HEADERS) $(KS1D_STABILITY_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(INMFD_LINSOLVERS) source/models/KS_1D/test_KS1D_stability_scan.cpp $(OPENMP) $(LFFTW) $(LLAPACK) -o $(BUILD_DIR)/test_KS1D_stability_scan_cpu_omp.bin 2>$(RESULTS)

KS1D_stability_scan_cuda.bin: source/models/KS_1D/test_KS1D_stability_scan.cpp $(KS1D_FULL_MODEL_HEADERS) $(KS1D_STABILITY_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS1D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) -x cu source/models/KS_1D/test_KS1D_stability_scan.cpp -c -o $(BUILD_DIR)/test_KS1D_stability_scan_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/test_KS1D_stability_scan_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS2) $(LLAPACK) -o $(BUILD_DIR)/test_KS1D_stability_scan_cuda.bin 2>$(RESULTS)

KS1D_bd_cpu_omp: source/models/KS_1D/KS1D_bd.cpp $(KS1D_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/KS_1D/KS1D_bd.cpp $(OPENMP) $(LFFTW) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_bd_cpu_omp.bin 2>$(RESULTS)

KS1D_full_bd_cpu_omp: source/models/KS_1D/KS1D_full_bd.cpp $(KS1D_FULL_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/KS_1D/KS1D_full_bd.cpp $(OPENMP) $(LFFTW) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_full_bd_cpu_omp.bin 2>$(RESULTS)

KS1D_bd: KS1D_bd_cuda

KS1D_bd_cuda: source/models/KS_1D/KS1D_bd.cpp $(KS1D_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS1D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/KS_1D/KS1D_bd.cpp -c -o $(BUILD_DIR)/KS1D_bd_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/KS1D_bd_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_bd_cuda.bin 2>$(RESULTS)

KS1D_full_bd_cuda: source/models/KS_1D/KS1D_full_bd.cpp $(KS1D_FULL_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS1D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/KS_1D/KS1D_full_bd.cpp -c -o $(BUILD_DIR)/KS1D_full_bd_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/KS1D_full_bd_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_full_bd_cuda.bin 2>$(RESULTS)

KS1D_stability_cpu_omp: source/models/KS_1D/KS1D_stability.cpp $(KS1D_MODEL_HEADERS) $(KS1D_STABILITY_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(INMFD_LINSOLVERS) $(IBOOST) source/models/KS_1D/KS1D_stability.cpp $(OPENMP) $(LFFTW) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_stability_cpu_omp.bin 2>$(RESULTS)

KS1D_full_stability_cpu_omp: source/models/KS_1D/KS1D_full_stability.cpp $(KS1D_FULL_MODEL_HEADERS) $(KS1D_STABILITY_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(INMFD_LINSOLVERS) $(IBOOST) source/models/KS_1D/KS1D_full_stability.cpp $(OPENMP) $(LFFTW) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_full_stability_cpu_omp.bin 2>$(RESULTS)

test_KS1D_full_stability_replay_cpu_omp: KS1D_full_stability_cpu_omp $(KS1D_STABILITY_REPLAY_SCRIPT) $(KS1D_STABILITY_REPLAY_MANIFEST)
	OMP_NUM_THREADS=8 python3 $(KS1D_STABILITY_REPLAY_SCRIPT) --executable $(BUILD_DIR)/KS1D_full_stability_cpu_omp.bin --config json_project_files/KS1D_test_full.json --manifest $(KS1D_STABILITY_REPLAY_MANIFEST)

KS1D_stability_cuda: source/models/KS_1D/KS1D_stability.cpp $(KS1D_MODEL_HEADERS) $(KS1D_STABILITY_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS1D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) $(IBOOST) -x cu source/models/KS_1D/KS1D_stability.cpp -c -o $(BUILD_DIR)/KS1D_stability_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/KS1D_stability_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_stability_cuda.bin 2>$(RESULTS)

KS1D_full_stability_cuda: source/models/KS_1D/KS1D_full_stability.cpp $(KS1D_FULL_MODEL_HEADERS) $(KS1D_STABILITY_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) source/common/cuda_init_scfd.h $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda -DKS1D_VECTOR_BACKEND_CUDA $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(INMFD_LINSOLVERS) $(IBOOST) -x cu source/models/KS_1D/KS1D_full_stability.cpp -c -o $(BUILD_DIR)/KS1D_full_stability_cuda_main.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/KS1D_full_stability_cuda_main.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/KS1D_full_stability_cuda.bin 2>$(RESULTS)

test_KS1D_full_stability_replay_cuda: KS1D_full_stability_cuda $(KS1D_STABILITY_REPLAY_SCRIPT) $(KS1D_STABILITY_REPLAY_MANIFEST)
	python3 $(KS1D_STABILITY_REPLAY_SCRIPT) --executable $(BUILD_DIR)/KS1D_full_stability_cuda.bin --config json_project_files/KS1D_test_full.json --manifest $(KS1D_STABILITY_REPLAY_MANIFEST) --device auto

KS1D_prepare_visualization_cpu_omp: source/models/KS_1D/KS1D_prepare_visualization.cpp $(KS1D_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) $(VISUALIZATION_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/KS_1D/KS1D_prepare_visualization.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/KS1D_prepare_visualization_cpu_omp.bin 2>$(RESULTS)

KS1D_full_prepare_visualization_cpu_omp: source/models/KS_1D/KS1D_full_prepare_visualization.cpp $(KS1D_FULL_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS) $(FFT_FACADE_HEADERS) $(VISUALIZATION_HEADERS)
	$(G++) $(G++FLAGS) -DKS1D_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/KS_1D/KS1D_full_prepare_visualization.cpp $(OPENMP) $(LFFTW) -o $(BUILD_DIR)/KS1D_full_prepare_visualization_cpu_omp.bin 2>$(RESULTS)

prepare_visualization_refactored_cpu_omp: circle_prepare_visualization_cpu_omp bratu_prepare_visualization_cpu_omp star_shaped_prepare_visualization_cpu_omp KS1D_prepare_visualization_cpu_omp KS1D_full_prepare_visualization_cpu_omp KS2D_prepare_visualization_cpu_omp

prepare_visualization_refactored_cuda: KS2D_prepare_visualization_cuda

KS_bd: source/models/KS_2D/KS_bd_json_new.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KS_2D/KS_bd_json_new.cpp $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBSAll) $(LLAPACK) -o $(BUILD_DIR)/KS_bd_json.bin 2>$(RESULTS)

Kolmogorov_3D_ker: source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT)   source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu -c -o $(BUILD_DIR)/Kolmogorov_3D_ker.o 2>$(RESULTS)

Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_RHS.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_Kolmogorov_3D_RHS.bin 2>$(RESULTS)

Kolmogorov_3D_all: Kolmogorov_3D_ker Kolmogorov_3D

Taylor_Green_ker: source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT)   source/nonlinear_operators/Taylor_Green/Taylor_Green_ker.cu -c -o $(BUILD_DIR)/Taylor_Green_ker.o 2>$(RESULTS)

Taylor_Green: source/models/Taylor_Green/test_Taylor_Green_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/Taylor_Green/test_Taylor_Green_RHS.cpp $(BUILD_DIR)/Taylor_Green_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_Taylor_Green_RHS.bin 2>$(RESULTS)

Taylor_Green_all: Taylor_Green_ker Taylor_Green


newton_Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_Newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_Newton.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/newton_Kolmogorov_3D.bin 2>$(RESULTS)

stability_newton_Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_Newton_stability.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_Newton_stability.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LLAPACK) $(LIBS2) -o $(BUILD_DIR)/newton_stability_Kolmogorov_3D.bin 2>$(RESULTS)

file_stability_newton_Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_Newton_file_stability.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KF_3D/test_Kolmogorov_3D_Newton_file_stability.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LLAPACK) $(LIBSAll) $(LIBBOOST) -o $(BUILD_DIR)/newton_stability_file_Kolmogorov_3D.bin 2>$(RESULTS)


deflation_Kolmogorov_3D: source/models/KF_3D/test_deflation_Kolmogorov_3D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_deflation_Kolmogorov_3D.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o  $(LIBS2) -o $(BUILD_DIR)/deflation_Kolmogorov_3D.bin 2>$(RESULTS)

compare_soluitons_files: source/models/KF_3D/compare_solutons_files.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/compare_solutons_files.cpp $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/Kolmogorov_3D_ker.o  $(LIBS2) -o $(BUILD_DIR)/compare_solutions_files.bin 2>$(RESULTS)

deflation_translation_Kolmogorov_3D: source/models/KF_3D/test_deflation_translation_Kolmogorov_3D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_deflation_translation_Kolmogorov_3D.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o  $(LIBS2) -o $(BUILD_DIR)/deflation_translation_Kolmogorov_3D.bin 2>$(RESULTS)

KF3D_bd: source/models/KF_3D/KF3D_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/KF_3D/KF3D_bd_json.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBSAll) $(LLAPACK) -o $(BUILD_DIR)/KF3D_bd_json.bin 2>$(RESULTS)

Kolmogorov_2D_ker: source/nonlinear_operators/Kolmogorov_flow_2D/Kolmogorov_2D_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/nonlinear_operators/Kolmogorov_flow_2D/Kolmogorov_2D_ker.cu -c -o $(BUILD_DIR)/Kolmogorov_2D_ker.o 2>$(RESULTS)

Kolmogorov_2D: source/nonlinear_operators/Kolmogorov_flow_2D/
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/test_Kolmogorov_2D_RHS.cpp $(BUILD_DIR)/Kolmogorov_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_Kolmogorov_2D_RHS.bin 2>$(RESULTS)

newton_Kolmogorov_2D: source/models/KF_2D/test_Kolmogorov_2D_Newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/test_Kolmogorov_2D_Newton.cpp $(BUILD_DIR)/Kolmogorov_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/newton_Kolmogorov_2D.bin 2>$(RESULTS)

deflation_Kolmogorov_2D: source/models/KF_2D/test_deflation_Kolmogorov_2D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/test_deflation_Kolmogorov_2D.cpp $(BUILD_DIR)/Kolmogorov_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/deflation_Kolmogorov_2D.bin 2>$(RESULTS)

KF2D_bd: source/models/KF_2D/KF2D_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KF_2D/KF2D_bd.cpp $(BUILD_DIR)/Kolmogorov_2D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBBOOST) $(LIBS2) $(LLAPACK) -o $(BUILD_DIR)/KF2D_bd.bin 2>$(RESULTS)

KF3D_view: source/models/KF_3D/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/view_solution.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o $(BUILD_DIR)/KF3D_view_solution.bin 2>$(RESULTS)

KF3D_view_translation: source/models/KF_3D/view_solution_translation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/view_solution_translation.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o $(BUILD_DIR)/KF3D_view_solution_translation.bin 2>$(RESULTS)

KS2D_view: source/models/KS_2D/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/view_solution.cpp $(BUILD_DIR)/Kuramoto_Sivashinskiy_2D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o $(BUILD_DIR)/KS2D_view_solution.bin 2>$(RESULTS)

KF2D_view: source/models/KF_2D/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/view_solution.cpp $(BUILD_DIR)/Kolmogorov_2D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o $(BUILD_DIR)/KF2D_view_solution.bin 2>$(RESULTS)

KF3D_1_bd: source/models/KF_3D/KF3D_1_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(LBOOST) source/models/KF_3D/KF3D_1_bd.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o  $(LCUDA)  $(LIBS2) $(LIBBOOST) -o $(BUILD_DIR)/KF3D_1_bd.bin 2>$(RESULTS)

test_Kolmogorov_3D_continuation_file: source/models/KF_3D/test_Kolmogorov_3D_continuation_file.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_continuation_file.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBSAll) -o $(BUILD_DIR)/test_Kolmogorov_3D_continuation_file.bin 2>$(RESULTS)

Kolmogorov_3D_time_stepping: source/models/KF_3D/Kolmogorov_3D_continuation_time_stepping.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/Kolmogorov_3D_continuation_time_stepping.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/Kolmogorov_3D_time_stepping.bin 2>$(RESULTS)

Kolmogorov_3D_perodic_orbit_stabilization: source/models/KF_3D/Kolmogorov_3D_perioidc_orbit_stabilization.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/Kolmogorov_3D_perioidc_orbit_stabilization.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LIBS3) $(LLAPACK) -o $(BUILD_DIR)/Kolmogorov_3D_perodic_orbit_stabilization.bin 2>$(RESULTS)

test_Kolmogorov_3D_stiff_solve: source/models/KF_3D/test_Kolmogorov_3D_stiff_solve.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_stiff_solve.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LIBS3) $(LLAPACK) -o $(BUILD_DIR)/test_Kolmogorov_3D_stiff_solve.bin 2>$(RESULTS)

Kolmogorov_3D_lyapunov_exponents: source/models/KF_3D/Kolmogorov_3D_lyapunov_exponents.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/Kolmogorov_3D_lyapunov_exponents.cpp $(BUILD_DIR)/Kolmogorov_3D_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/Kolmogorov_3D_lyapunov_exponents.bin 2>$(RESULTS)


# abc_flow
abc_flow_ker: source/nonlinear_operators/abc_flow/abc_flow_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/nonlinear_operators/abc_flow/abc_flow_ker.cu -c -o $(BUILD_DIR)/abc_flow_ker.o 2>$(RESULTS)


abc_flow_rhs: source/models/abc_flow/test_abc_flow_rhs.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/test_abc_flow_rhs.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_abc_flow_rhs.bin 2>$(RESULTS)

abc_flow_newton: source/models/abc_flow/test_abc_flow_newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/test_abc_flow_newton.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/test_abc_flow_newton.bin 2>$(RESULTS)

abc_flow_deflation: source/models/abc_flow/test_deflation_abc_flow.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/test_deflation_abc_flow.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBSAll) -o $(BUILD_DIR)/test_deflation_abc_flow.bin 2>$(RESULTS)

abc_bd_deb: source/models/abc_flow/abc_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/abc_flow/abc_bd_json.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBS2) $(LLAPACK) -o $(BUILD_DIR)/abc_bd_json.bin 2>$(RESULTS)

abc_bd_rel: source/models/abc_flow/abc_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/abc_flow/abc_bd_json.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBS3) $(LLAPACK) -o $(BUILD_DIR)/abc_bd_json.bin 2>$(RESULTS)


abc_bd: source/models/abc_flow/abc_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/abc_flow/abc_bd_json.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBSAll) $(LLAPACK) -o $(BUILD_DIR)/abc_bd_json.bin 2>$(RESULTS)

abc_view: source/models/abc_flow/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/view_solution.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LCUDA) $(LIBSAll) -o $(BUILD_DIR)/abc_view_solution.bin 2>$(RESULTS)

abc_flow_time_stepping: source/models/abc_flow/abc_flow_time_stepping.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/abc_flow_time_stepping.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/abc_flow_time_stepping.bin 2>$(RESULTS)

abc_flow_perodic_orbit_stabilization: source/models/abc_flow/abc_flow_perodic_orbit_stabilization.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/abc_flow_perodic_orbit_stabilization.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LIBS3) $(LLAPACK) -o $(BUILD_DIR)/abc_flow_perodic_orbit_stabilization.bin 2>$(RESULTS)

abc_flow_lyapunov_exponents: source/models/abc_flow/abc_flow_lyapunov_exponents.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/abc_flow_lyapunov_exponents.cpp $(BUILD_DIR)/abc_flow_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(LIBS2) -o $(BUILD_DIR)/abc_flow_lyapunov_exponents.bin 2>$(RESULTS)


test_butcher_tables: source/time_stepper/tests/butcher_tables.cpp
	$(G++) $(G++FLAGS) $(IPROJECT) source/time_stepper/tests/butcher_tables.cpp -o $(BUILD_DIR)/test_butcher_tables.bin 2>$(RESULTS)

#overscreening breakdown
ob_ker: source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cu -c -o $(BUILD_DIR)/overscreening_breakdown_ker.o 2>$(RESULTS)

ob_ker_var_prec: source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cpp
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) $(IBOOST) source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cpp -c -o $(BUILD_DIR)/overscreening_breakdown_ker_var_prec.o $(NVOPENMP) 2>$(RESULTS)


test_ob_kernels:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_kernels.cpp $(BUILD_DIR)/overscreening_breakdown_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o $(BUILD_DIR)/test_ob_kernels.bin 2>$(RESULTS)

test_ob_kernels_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_kernels_var_prec.cpp $(BUILD_DIR)/overscreening_breakdown_ker_var_prec.o $(NVOPENMP) -o $(BUILD_DIR)/test_ob_kernels_var_prec.bin 2>$(RESULTS)


test_ob_problem:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_problem.cpp $(BUILD_DIR)/overscreening_breakdown_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o $(BUILD_DIR)/test_ob_problem.bin 2>$(RESULTS)

test_ob_problem_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_problem_var_prec.cpp $(BUILD_DIR)/overscreening_breakdown_ker_var_prec.o -o $(BUILD_DIR)/test_ob_problem_var_prec.bin $(NVOPENMP) 2>$(RESULTS)


ob_view:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/view_solution.cpp $(BUILD_DIR)/overscreening_breakdown_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o $(BUILD_DIR)/ob_view_solution.bin 2>$(RESULTS)


test_ob_newton:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_newton.cpp $(BUILD_DIR)/overscreening_breakdown_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) $(LLAPACK) -o $(BUILD_DIR)/test_ob_newton.bin 2>$(RESULTS)

test_ob_newton_deflation:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_newton_deflation.cpp $(BUILD_DIR)/overscreening_breakdown_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) $(LLAPACK) -o $(BUILD_DIR)/test_ob_newton_deflation.bin 2>$(RESULTS)


test_ob_newton_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_newton_var_prec.cpp $(BUILD_DIR)/overscreening_breakdown_ker_var_prec.o $(NVOPENMP) -o $(BUILD_DIR)/test_ob_newton_var_prec.bin 2>$(RESULTS)

test_ob_newton_deflation_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_newton_deflation_var_prec.cpp $(BUILD_DIR)/overscreening_breakdown_ker_var_prec.o $(NVOPENMP) -o $(BUILD_DIR)/test_ob_newton_deflation_var_prec.bin 2>$(RESULTS)


ob_json:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/ob_bd_json.cpp $(BUILD_DIR)/overscreening_breakdown_ker.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/ob_bd_json.bin 2>$(RESULTS)

# test small problems time stepper:
test_vdp_time_stepping:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/time_stepper/tests/vdp_1.cpp $(OPENMP) -o $(BUILD_DIR)/test_vdp.bin 2>$(RESULTS)
test_rossler_time_stepping:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/time_stepper/tests/rossler_1.cpp $(OPENMP) -o $(BUILD_DIR)/test_rossler.bin 2>$(RESULTS)

test_lorentz_time_stepping:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/time_stepper/tests/lorentz.cpp $(OPENMP) -o $(BUILD_DIR)/test_lorentz.bin 2>$(RESULTS)
# test periodic orbits:
test_rossler_to_section:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_to_section.cpp $(OPENMP) -o $(BUILD_DIR)/test_rossler_to_section.bin 2>$(RESULTS)

test_glued_nonlinear_operator:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/glued_nonlinear_operator_and_jacobian.cpp $(OPENMP) -o $(BUILD_DIR)/test_glued_nonlinear_operator_and_jacobian.bin 2>$(RESULTS)

test_glued_poincare_map_linear_operator:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/glued_poincare_map_linear_operator.cpp $(OPENMP) -o $(BUILD_DIR)/test_glued_poincare_map_linear_operator.bin 2>$(RESULTS)

test_rossler_periodic_orbit_cpu:
	$(G++)  $(G++FLAGS)  $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_periodic_cpu.cpp $(OPENMP) -o $(BUILD_DIR)/test_rossler_periodic_cpu.bin 2>$(RESULTS)


test_lorentz_periodic_orbit_cpu:
	$(G++)  $(G++FLAGS)  $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/lorentz_periodic_cpu.cpp $(OPENMP) -o $(BUILD_DIR)/test_lorentz_periodic_cpu.bin 2>$(RESULTS)

test_rossler_periodic_continuate_cpu:
	$(G++)  $(G++FLAGS)  $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) $(IBOOST) source/periodic_orbit/tests/rossler_periodic_continuate_cpu.cpp $(OPENMP) $(LBOOST) -o $(BUILD_DIR)/rossler_periodic_continuate_cpu.bin 2>$(RESULTS)


test_rossler_periodic_orbit_gpu:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_operator_ker.cu -c -o $(BUILD_DIR)/rossler_operator_kernels.o 2>$(RESULTS)
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_periodic_gpu.cpp $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_matrix_vector_operations_kernels.o $(BUILD_DIR)/rossler_operator_kernels.o -o $(BUILD_DIR)/test_rossler_periodic_gpu.bin $(LCUDA) $(LIBS3) $(LLAPACK) 2>$(RESULTS)

# make all common kernels
ker: gpu_matrix_vector_operations_ker gpu_vector_operations_ker gpu_reduction_ogita_ker abc_flow_ker Kolmogorov_3D_ker Taylor_Green_ker
