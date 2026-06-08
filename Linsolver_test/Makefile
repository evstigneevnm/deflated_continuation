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
HIPCC ?= hipcc
HIP_EXTENDED_LAMBDA ?= $(shell $(HIPCC) --version 2>/dev/null | grep -iq nvcc && printf '%s' '--extended-lambda')
HIPFLAGS ?= -std=$(CPPSTD) $(HIP_EXTENDED_LAMBDA) $(TARGET_HIPCC)
OPENMP = -fopenmp -lpthread
NVOPENMP = -Xcompiler $(OPENMP)
CONTRIB_SCFD = source/contrib/scfd
COMMON_NMFD_OPERATIONS = source/common/NMFD-operations

G++ = $(GCC_ROOT_PATH)/g++
LIBFLAGS = --compiler-options -fPIC
G++FLAGS = -std=$(CPPSTD) $(TARGET_GCC)
ICUDA = -I$(CUDA_ROOT_PATH)/include
IPROJECT = -I $(COMMON_NMFD_OPERATIONS) -I source/ -I $(CONTRIB_SCFD)/include
IBOOST = -I$(BOOST_ROOT_PATH)/include
SCFD_VECTOR_OPS_HEADERS = source/common/scfd_vector_operations.h source/common/NMFD-operations/nmfd/operations/scfd_vector_operations.h source/common/NMFD-operations/nmfd/operations/vector_operations_base.h source/common/NMFD-operations/nmfd/operations/vector_space_base.h
SCFD_SERIAL_VECTOR_OPS_HEADERS = $(SCFD_VECTOR_OPS_HEADERS) source/common/scfd_serial_cpu_vector_operations.h
SCFD_VECTOR_OPS_TEST_HEADERS = $(SCFD_VECTOR_OPS_HEADERS) source/common/tests/scfd_vector_operations_nmfd_interface_tests.h
CPU_VECTOR_OPS_VAR_PREC_HEADERS = source/common/cpu_vector_operations_var_prec.h source/common/NMFD-operations/nmfd/operations/cpu_vector_operations_var_prec.h source/common/NMFD-operations/nmfd/operations/vector_operations_base.h source/common/NMFD-operations/nmfd/operations/vector_space_base.h
COMMON_FILE_OPS_HEADERS = source/common/file_operations.h source/common/cpu_file_operations.h source/common/cpu_matrix_file_operations.h source/common/gpu_file_operations.h source/common/gpu_matrix_file_operations.h source/common/NMFD-operations/nmfd/operations/io/file_operations.h source/common/NMFD-operations/nmfd/operations/io/vector_file_operations.h source/common/NMFD-operations/nmfd/operations/io/matrix_file_operations.h
FFT_FACADE_HEADERS = source/external_libraries/fft_facade.h source/external_libraries/fft_facade_fftw.h source/external_libraries/fft_facade_cufft.h source/external_libraries/fftw_wrap.h source/external_libraries/cufft_wrap.h
CIRCLE_MODEL_HEADERS = source/models/circle/circle_backend_typedefs.h source/nonlinear_operators/circle/circle.h source/nonlinear_operators/circle/convergence_strategy.h source/nonlinear_operators/circle/linear_operator_circle.h source/nonlinear_operators/circle/preconditioner_circle.h source/nonlinear_operators/circle/system_operator.h
BRATU_MODEL_HEADERS = source/models/bratu/bratu_backend_typedefs.h source/nonlinear_operators/bratu/bratu.h source/nonlinear_operators/bratu/convergence_strategy.h source/nonlinear_operators/bratu/linear_operator_bratu.h source/nonlinear_operators/bratu/preconditioner_bratu.h source/nonlinear_operators/bratu/system_operator.h
STAR_SHAPED_MODEL_HEADERS = source/models/star_shaped/star_shaped_backend_typedefs.h source/nonlinear_operators/star_shaped/star_shaped.h source/nonlinear_operators/star_shaped/convergence_strategy.h source/nonlinear_operators/star_shaped/linear_operator_star_shaped.h source/nonlinear_operators/star_shaped/preconditioner_star_shaped.h source/nonlinear_operators/star_shaped/system_operator.h

LCUDA = -L$(CUDA_ROOT_PATH)/lib64
LBOOST = -L$(BOOST_ROOT_PATH)/lib
LIBS1 = $(LCUDA) -lcublas -lcurand
LIBS2 = $(LCUDA) -lcufft $(LIBS1)
LIBS3 = $(LCUDA) -lcusolver $(LIBS2)
LIBSAll = $(LCUDA) -lcublas -lcurand -lcufft -lcusolver
LIBBOOST = -lboost_serialization
LLAPACK = -L$(OPENBLAS_ROOT_PATH)/lib -lopenblas
LFFTW = -lfftw3 -lfftw3f

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

test_lapack_wrap.bin: source/external_libraries/tests/test_lapack_wrap.cpp source/contrib/scfd/include/scfd/external_libraries/lapack_wrap.h
	$(G++) $(G++FLAGS) $(IPROJECT) source/external_libraries/tests/test_lapack_wrap.cpp $(LLAPACK) -o $(BUILD_DIR)/test_lapack_wrap.bin 2>$(RESULTS)

test_lapack_wrap_cuda.bin: source/external_libraries/tests/test_lapack_wrap_cuda.cu source/external_libraries/lapack_wrap.h source/contrib/scfd/include/scfd/external_libraries/lapack_wrap.h source/contrib/scfd/include/scfd/external_libraries/lapack_wrap_device.h source/common/cuda_init_scfd.h
	$(NVCC) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/external_libraries/tests/test_lapack_wrap_cuda.cu $(LLAPACK) -o $(BUILD_DIR)/test_lapack_wrap_cuda.bin 2>$(RESULTS)


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

scfd_vector_operations.bin: source/common/tests/test_scfd_vector_operations.cu source/common/tests/vector_operations_template_tests.h $(SCFD_VECTOR_OPS_TEST_HEADERS)
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/tests/test_scfd_vector_operations.cu $(LIBS1) -o $(BUILD_DIR)/test_scfd_vector_operations.bin 2>$(RESULTS)

scfd_vector_operations_hip.bin: source/common/tests/test_scfd_vector_operations_hip.cpp source/common/tests/vector_operations_template_tests.h source/common/hip_init_scfd.h $(SCFD_VECTOR_OPS_TEST_HEADERS)
	$(HIPCC) $(HIPFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/common/tests/test_scfd_vector_operations_hip.cpp -o $(BUILD_DIR)/test_scfd_vector_operations_hip.bin 2>$(RESULTS)

scfd_vector_operations_cpu.bin: source/common/tests/test_scfd_vector_operations_cpu.cpp source/common/tests/vector_operations_template_tests.h $(SCFD_VECTOR_OPS_TEST_HEADERS)
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/common/tests/test_scfd_vector_operations_cpu.cpp $(OPENMP) -o $(BUILD_DIR)/test_scfd_vector_operations_cpu.bin 2>$(RESULTS)

test_gpu_reduction_ogita.bin: source/common/tests/test_gpu_reduction_ogita.cpp $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/common/tests/test_gpu_reduction_ogita.cpp $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LCUDA) -o $(BUILD_DIR)/test_gpu_reduction_ogita.bin 2>$(RESULTS)

test_vector_snapshot_queue.bin: source/common/tests/test_vector_snapshot_queue.cu $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(ICUDA) $(IPROJECT) source/common/tests/test_vector_snapshot_queue.cu $(BUILD_DIR)/gpu_vector_operations_kernels.o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o $(LIBS1) -o $(BUILD_DIR)/test_vector_snapshot_queue.bin 2>$(RESULTS)

gpu_reduction_ogita_ker: $(BUILD_DIR)/gpu_reduction_ogita_kernels.o

$(BUILD_DIR)/gpu_reduction_ogita_kernels.o: source/common/ogita/gpu_reduction_ogita_kernels.cu | $(BUILD_STAMP)
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/common/ogita/gpu_reduction_ogita_kernels.cu -c -o $(BUILD_DIR)/gpu_reduction_ogita_kernels.o 2>$(RESULTS)

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

cont_def_circle: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) source/common/cuda_init_scfd.h $(SCFD_VECTOR_OPS_HEADERS)
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/circle/circle_test_deflation_continuation.cpp $(LIBS1) -o $(BUILD_DIR)/circle_test_deflation_continuation.bin 2>$(RESULTS)

cont_def_circle_cpu_omp: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) source/models/circle/circle_test_deflation_continuation.cpp $(OPENMP) -o $(BUILD_DIR)/circle_test_deflation_continuation_cpu_omp.bin 2>$(RESULTS)

cont_def_circle_hip: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) source/common/hip_init_scfd.h $(SCFD_VECTOR_OPS_HEADERS)
	$(HIPCC) $(HIPFLAGS) -DCIRCLE_VECTOR_BACKEND_HIP $(SCALAR_TYPE) $(IPROJECT) source/models/circle/circle_test_deflation_continuation.cpp -o $(BUILD_DIR)/circle_test_deflation_continuation_hip.bin 2>$(RESULTS)

cont_def_circle_var_prec: source/models/circle/circle_test_deflation_continuation.cpp source/models/circle/circle_test_deflation_continuation_typedefs.h $(CIRCLE_MODEL_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/circle/circle_test_deflation_continuation.cpp $(OPENMP) -o $(BUILD_DIR)/circle_test_deflation_continuation_var_prec.bin 2>$(RESULTS)

circle_curve_container: source/models/circle/circle_test_curve_container.cpp source/models/circle/circle_test_curve_container.h source/models/circle/circle_test_deflation_continuation_typedefs.h source/nonlinear_operators/circle/circle.h $(SCFD_VECTOR_OPS_HEADERS)
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -x cu source/models/circle/circle_test_curve_container.cpp $(LIBS1) -o $(BUILD_DIR)/circle_test_curve_container.bin 2>$(RESULTS)

circle_bd: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/circle/circle_bd.cpp $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd.bin 2>$(RESULTS)

circle_bd_cpu_omp: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/circle/circle_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd_cpu_omp.bin 2>$(RESULTS)

circle_bd_hip: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(HIPCC) $(HIPFLAGS) -DCIRCLE_VECTOR_BACKEND_HIP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/circle/circle_bd.cpp $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd_hip.bin 2>$(RESULTS)

circle_bd_var_prec: source/models/circle/circle_bd.cpp $(CIRCLE_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DCIRCLE_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/circle/circle_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/circle_bd_var_prec.bin 2>$(RESULTS)

bratu_bd_cpu_omp: source/models/bratu/bratu_bd.cpp $(BRATU_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DBRATU_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/bratu/bratu_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/bratu_bd_cpu_omp.bin 2>$(RESULTS)

bratu_bd_var_prec: source/models/bratu/bratu_bd.cpp $(BRATU_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DBRATU_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/bratu/bratu_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/bratu_bd_var_prec.bin 2>$(RESULTS)

star_shaped_bd: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(NVCC) $(NVCCFLAGS) --extended-lambda $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) -x cu source/models/star_shaped/star_shaped_bd.cpp $(LIBSAll) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd.bin 2>$(RESULTS)

star_shaped_bd_cpu_omp: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(G++) $(G++FLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_OMP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/star_shaped/star_shaped_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd_cpu_omp.bin 2>$(RESULTS)

star_shaped_bd_hip: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(SCFD_VECTOR_OPS_HEADERS)
	$(HIPCC) $(HIPFLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_HIP $(SCALAR_TYPE) $(IPROJECT) $(IBOOST) source/models/star_shaped/star_shaped_bd.cpp $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd_hip.bin 2>$(RESULTS)

star_shaped_bd_var_prec: source/models/star_shaped/star_shaped_bd.cpp $(STAR_SHAPED_MODEL_HEADERS) $(COMMON_FILE_OPS_HEADERS) $(CPU_VECTOR_OPS_VAR_PREC_HEADERS)
	$(G++) $(G++FLAGS) -DSTAR_SHAPED_VECTOR_BACKEND_VAR_PREC $(IPROJECT) $(IBOOST) source/models/star_shaped/star_shaped_bd.cpp $(OPENMP) $(LBOOST) $(LIBBOOST) -o $(BUILD_DIR)/star_shaped_bd_var_prec.bin 2>$(RESULTS)

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
