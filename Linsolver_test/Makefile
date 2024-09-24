# this Makefile is configured by the 'config.inc' that describes your system env.

ifndef CONFIG_FILE
CONFIG_FILE = config.inc
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
$(info config file $(CONFIG_FILE) does not exist.)
$(error Create $(CONFIG_FILE) from example or specify another config via: make <target> CONFIG_FILE=<config_filename> )
endif

-include $(CONFIG_FILE)


NVCC = $(CUDA_ROOT_PATH)/bin/nvcc
NVCC_CHECK_REGS = -Xptxas -v 
NVCCFLAGS = -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC)
OPENMP = -fopenmp -lpthread
NVOPENMP = -Xcompiler $(OPENMP)
CONTRIB_SCFD = source/contrib/scfd

G++ = $(GCC_ROOT_PATH)/g++
LIBFLAGS = --compiler-options -fPIC
G++FLAGS = -std=$(CPPSTD) $(TARGET_GCC)
ICUDA = -I$(CUDA_ROOT_PATH)/include
IPROJECT = -I source/ -I $(CONTRIB_SCFD)/include
IBOOST = -I$(BOOST_ROOT_PATH)/include

LCUDA = -L$(CUDA_ROOT_PATH)/lib64
LBOOST = -L$(BOOST_ROOT_PATH)/lib
LIBS1 = $(LCUDA) -lcublas -lcurand 
LIBS2 = $(LCUDA) -lcufft $(LIBS1)
LIBS3 = $(LCUDA) -lcusolver $(LIBS2)
LIBSAll = $(LCUDA) -lcublas -lcurand -lcufft -lcusolver
LIBBOOST = -lboost_serialization
LLAPACK = -L$(OPENBLAS_ROOT_PATH)/lib -lopenblas

#clean
clean:
	rm *.bin O/*.o

#component tests
test_cpu_vector_operations.bin: test_cpu_vector_operations.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/common/tests/test_cpu_vector_operations.cpp $(OPENMP) -o test_cpu_vector_operations.bin 2>results.make

test_multivector.bin: source/common/tests/test_multivector.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/common/tests/test_multivector.cpp $(OPENMP) -o test_multivector.bin 2>results.make


test_cpu_glued_vector_operations.bin:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/common/tests/test_cpu_glued_vector_operations.cpp $(OPENMP) -o test_cpu_glued_vector_operations.bin 2>results.make

test_multi_precision.bin:
	$(G++) $(G++FLAGS) $(IPROJECT) $(IBOOST) $(ICUDA) source/test_inst/test_multi_precision/test_multi_precision.cpp -o test_multi_precision.bin $(OPENMP)


#all targets
lapack_test.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/tests/lapack.cpp $(LLAPACK) -o test_lapack.bin

iram_bulge_test.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/IRAM/predefined_matrix.cpp $(LIBS1) $(LLAPACK) O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o -o test_IRAM.bin	

iram_simulation.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/IRAM/iram_simulation.cpp $(LIBS1) $(LLAPACK) O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o -o test_IRAM.bin	

iram_process.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/IRAM/iram_test.cpp $(LCUDA) $(LIBSAll) $(LLAPACK) O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o -o test_IRAM.bin	

arnoldi_iters.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/test_inst/Arnoldi_Iterations/arnoldi_iteraitons_test.cpp $(LIBS1) $(LLAPACK) O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o -o test_ArIter.bin	


cpu_1st_call_alloc_test.bin: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o cpu_1st_call_alloc_test_float.bin

sm_test.bin: source/numerical_algos/lin_solvers/tests/sm_test.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/numerical_algos/lin_solvers/tests/sm_test.cpp -o sm_test.bin 2>results.make $(OPENMP)

iterative_solvers_test.bin: source/numerical_algos/lin_solvers/tests/iterative_solvers_test.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/numerical_algos/lin_solvers/tests/iterative_solvers_test.cpp -o iterative_solvers_test.bin 2>results.make $(OPENMP)

iterative_solvers_test_different_operators.bin: source/numerical_algos/lin_solvers/tests/iterative_solvers_test_different_operators.cpp
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/numerical_algos/lin_solvers/tests/iterative_solvers_test_different_operators.cpp -o iterative_solvers_test_different_operators.bin 2>results.make $(OPENMP)


exact_linsolver_test.bin: source/numerical_algos/lin_solvers/tests/exact_solver_test.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/numerical_algos/lin_solvers/tests/exact_solver_test.cpp O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o exact_solver_test.bin 2>results.make 

lin_solvers_test.bin: source/models/tests/linear_solvers_test.cpp
	$(NVCC) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/tests/linear_solvers_test.cpp -o linear_solvers_test.bin 2>results.make

cufft_test.bin: source/models/tests/cufft_test.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/models/tests/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results.make
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -g source/models/tests/cufft_test.cpp O/cufft_test_kernels.o $(LIBS2) -o cufft_test.bin 2>results.make

cufft_test_2D.bin: source/models/tests/cufft_test_2D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results.make
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_2D.cpp O/cufft_test_kernels.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LIBS3) -o cufft_test_2D.bin 2>results.make
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_2D_1.cpp O/cufft_test_kernels.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LIBS3) -o cufft_test_2D_1.bin 2>results.make

gpu_vector_operations.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results.make
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/test_vector_operations.cpp O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o $(LIBS2) -o test_vector_operations.bin 2>results.make

gpu_reduction_ogita_ker:
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/common/ogita/gpu_reduction_ogita_kernels.cu -c -o O/gpu_reduction_ogita_kernels.o 2>results.make

gpu_vector_operations_ker:
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/common/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results.make

gpu_matrix_vector_operations_ker:
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)   source/common/gpu_matrix_vector_operations_kernels.cu -c -o O/gpu_matrix_vector_operations_kernels.o 2>results.make
gpu_matrix_vector_operations.bin:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/test_matrix_vector_operations.cpp $(LLAPACK) $(LIBS3) O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o O/gpu_matrix_vector_operations_kernels.o -o test_matrix_vector_operations.bin 2>results.make	


Kuramoto_Sivashinskiy_2D_ker:
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)   source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results.make

test_Kuramoto_Sivashinskiy_2D_RHS.bin: source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_RHS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o $(LIBS2) -o test_Kuramoto_Sivashinskiy_2D_RHS.bin 2>results.make

test_Kuramoto_Sivashinskiy_2D_Newton.bin: source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_Newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_Newton.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o $(LIBS2) -o test_Kuramoto_Sivashinskiy_2D_Newton.bin 2>results.make


deflation_KS_2D_S: source/models/KS_2D/test_deflation_KS.cpp
	$(NVCC) -DHIGH_PREC=false $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_KS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o $(LIBS2) -o test_deflation_S.bin 2>results.make

deflation_KS_2D_H: source/models/KS_2D/test_deflation_KS.cpp
	$(NVCC) -DHIGH_PREC=true $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_KS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o $(LIBS2) -o test_deflation_H.bin 2>results.make



cont_def_KS_2D: source/models/KS_2D/test_deflation_continuation_KS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_continuation_KS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_deflation_continuation.bin 2>results.make

circle_ker: source/nonlinear_operators/circle/circle_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/nonlinear_operators/circle/circle_ker.cu -c -o O/circle_ker.o 2>results.make

cont_def_circle: source/models/circle/circle_test_deflation_continuation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/circle/circle_test_deflation_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o $(LIBS1) -o circle_test_deflation_continuation.bin 2>results.make	

circle_bd: source/models/circle/circle_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/circle/circle_bd.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o O/circle_ker.o $(LIBSAll) $(LBOOST) $(LIBBOOST) -o circle_bd.bin 2>results.make	

KS_bd: source/models/KS_2D/KS_bd_json_new.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KS_2D/KS_bd_json_new.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBSAll) $(LLAPACK) -o KS_bd_json.bin 2>results.make	

Kolmogorov_3D_ker: source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT)   source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu -c -o O/Kolmogorov_3D_ker.o 2>results.make

Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_RHS.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_Kolmogorov_3D_RHS.bin 2>results.make

Kolmogorov_3D_all: Kolmogorov_3D_ker Kolmogorov_3D

Taylor_Green_ker: source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT)   source/nonlinear_operators/Taylor_Green/Taylor_Green_ker.cu -c -o O/Taylor_Green_ker.o 2>results.make

Taylor_Green: source/models/Taylor_Green/test_Taylor_Green_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/Taylor_Green/test_Taylor_Green_RHS.cpp O/Taylor_Green_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_Taylor_Green_RHS.bin 2>results.make

Taylor_Green_all: Taylor_Green_ker Taylor_Green


newton_Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_Newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_Newton.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o newton_Kolmogorov_3D.bin 2>results.make

stability_newton_Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_Newton_stability.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_Newton_stability.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LLAPACK) $(LIBS2) -o newton_stability_Kolmogorov_3D.bin 2>results.make

file_stability_newton_Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_Newton_file_stability.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KF_3D/test_Kolmogorov_3D_Newton_file_stability.cpp O/Kolmogorov_3D_ker.o O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LLAPACK) $(LIBSAll) $(LIBBOOST) -o newton_stability_file_Kolmogorov_3D.bin 2>results.make


deflation_Kolmogorov_3D: source/models/KF_3D/test_deflation_Kolmogorov_3D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_deflation_Kolmogorov_3D.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o  $(LIBS2) -o deflation_Kolmogorov_3D.bin 2>results.make

compare_soluitons_files: source/models/KF_3D/compare_solutons_files.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/compare_solutons_files.cpp O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/Kolmogorov_3D_ker.o  $(LIBS2) -o compare_solutions_files.bin 2>results.make

deflation_translation_Kolmogorov_3D: source/models/KF_3D/test_deflation_translation_Kolmogorov_3D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_deflation_translation_Kolmogorov_3D.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o  $(LIBS2) -o deflation_translation_Kolmogorov_3D.bin 2>results.make

KF3D_bd: source/models/KF_3D/KF3D_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/KF_3D/KF3D_bd_json.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBS2) $(LLAPACK) -o KF3D_bd_json.bin 2>results.make	

Kolmogorov_2D_ker: source/nonlinear_operators/Kolmogorov_flow_2D/Kolmogorov_2D_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT)  source/nonlinear_operators/Kolmogorov_flow_2D/Kolmogorov_2D_ker.cu -c -o O/Kolmogorov_2D_ker.o 2>results.make

Kolmogorov_2D: source/nonlinear_operators/Kolmogorov_flow_2D/
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/test_Kolmogorov_2D_RHS.cpp O/Kolmogorov_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_Kolmogorov_2D_RHS.bin 2>results.make

newton_Kolmogorov_2D: source/models/KF_2D/test_Kolmogorov_2D_Newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/test_Kolmogorov_2D_Newton.cpp O/Kolmogorov_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o newton_Kolmogorov_2D.bin 2>results.make

deflation_Kolmogorov_2D: source/models/KF_2D/test_deflation_Kolmogorov_2D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/test_deflation_Kolmogorov_2D.cpp O/Kolmogorov_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o deflation_Kolmogorov_2D.bin 2>results.make

KF2D_bd: source/models/KF_2D/KF2D_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KF_2D/KF2D_bd.cpp O/Kolmogorov_2D_ker.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBBOOST) $(LIBS2) $(LLAPACK) -o KF2D_bd.bin 2>results.make	

KF3D_view: source/models/KF_3D/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/view_solution.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o KF3D_view_solution.bin 2>results.make

KF3D_view_translation: source/models/KF_3D/view_solution_translation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/view_solution_translation.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o KF3D_view_solution_translation.bin 2>results.make

KS2D_view: source/models/KS_2D/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/view_solution.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o KS2D_view_solution.bin 2>results.make

KF2D_view: source/models/KF_2D/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_2D/view_solution.cpp O/Kolmogorov_2D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LCUDA) $(LIBS2) -o KF2D_view_solution.bin 2>results.make

KF3D_1_bd: source/models/KF_3D/KF3D_1_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(LBOOST) source/models/KF_3D/KF3D_1_bd.cpp O/Kolmogorov_3D_ker.o O/gpu_vector_operations_kernels.o  $(LCUDA)  $(LIBS2) $(LIBBOOST) -o KF3D_1_bd.bin 2>results.make	

test_Kolmogorov_3D_continuation_file: source/models/KF_3D/test_Kolmogorov_3D_continuation_file.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_continuation_file.cpp O/Kolmogorov_3D_ker.o O/gpu_vector_operations_kernels.o O/gpu_reduction_ogita_kernels.o $(LIBSAll) -o test_Kolmogorov_3D_continuation_file.bin 2>results.make

Kolmogorov_3D_time_stepping: source/models/KF_3D/Kolmogorov_3D_continuation_time_stepping.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/Kolmogorov_3D_continuation_time_stepping.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o Kolmogorov_3D_time_stepping.bin 2>results.make

Kolmogorov_3D_perodic_orbit_stabilization: source/models/KF_3D/Kolmogorov_3D_perioidc_orbit_stabilization.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/Kolmogorov_3D_perioidc_orbit_stabilization.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LIBS3) $(LLAPACK) -o Kolmogorov_3D_perodic_orbit_stabilization.bin 2>results.make

test_Kolmogorov_3D_stiff_solve: source/models/KF_3D/test_Kolmogorov_3D_stiff_solve.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_stiff_solve.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LIBS3) $(LLAPACK) -o test_Kolmogorov_3D_stiff_solve.bin 2>results.make

Kolmogorov_3D_lyapunov_exponents: source/models/KF_3D/Kolmogorov_3D_lyapunov_exponents.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/Kolmogorov_3D_lyapunov_exponents.cpp O/Kolmogorov_3D_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o Kolmogorov_3D_lyapunov_exponents.bin 2>results.make


# abc_flow
abc_flow_ker: source/nonlinear_operators/abc_flow/abc_flow_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/nonlinear_operators/abc_flow/abc_flow_ker.cu -c -o O/abc_flow_ker.o 2>results.make


abc_flow_rhs: source/models/abc_flow/test_abc_flow_rhs.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/test_abc_flow_rhs.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_abc_flow_rhs.bin 2>results.make

abc_flow_newton: source/models/abc_flow/test_abc_flow_newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/test_abc_flow_newton.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_abc_flow_newton.bin 2>results.make

abc_flow_deflation: source/models/abc_flow/test_deflation_abc_flow.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/test_deflation_abc_flow.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBSAll) -o test_deflation_abc_flow.bin 2>results.make

abc_bd_deb: source/models/abc_flow/abc_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/abc_flow/abc_bd_json.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBS2) $(LLAPACK) -o abc_bd_json.bin 2>results.make	

abc_bd_rel: source/models/abc_flow/abc_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/abc_flow/abc_bd_json.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBS2) $(LLAPACK) -o abc_bd_json.bin 2>results.make	


abc_bd: source/models/abc_flow/abc_bd_json.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IBOOST) $(IPROJECT) source/models/abc_flow/abc_bd_json.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LBOOST) $(LIBBOOST) $(LIBSAll) $(LLAPACK) -o abc_bd_json.bin 2>results.make	

abc_view: source/models/abc_flow/view_solution.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/view_solution.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LCUDA) $(LIBSAll) -o abc_view_solution.bin 2>results.make

abc_flow_time_stepping: source/models/abc_flow/abc_flow_time_stepping.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/abc_flow_time_stepping.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o abc_flow_time_stepping.bin 2>results.make

abc_flow_perodic_orbit_stabilization: source/models/abc_flow/abc_flow_perodic_orbit_stabilization.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/abc_flow_perodic_orbit_stabilization.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LIBS2) $(LLAPACK) -o abc_flow_perodic_orbit_stabilization.bin 2>results.make

abc_flow_lyapunov_exponents: source/models/abc_flow/abc_flow_lyapunov_exponents.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/abc_flow/abc_flow_lyapunov_exponents.cpp O/abc_flow_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LIBS2) -o abc_flow_lyapunov_exponents.bin 2>results.make


test_butcher_tables: source/time_stepper/tests/butcher_tables.cpp
	$(G++) $(G++FLAGS) $(IPROJECT) source/time_stepper/tests/butcher_tables.cpp -o test_butcher_tables.bin 2>results.make

#overscreening breakdown
ob_ker: source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cu
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cu -c -o O/overscreening_breakdown_ker.o 2>results.make

ob_ker_var_prec: source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cpp
	$(NVCC) $(LIBFLAGS) $(NVCCFLAGS) $(ICUDA) $(IPROJECT) $(IBOOST) source/nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.cpp -c -o O/overscreening_breakdown_ker_var_prec.o $(NVOPENMP) 2>results.make


test_ob_kernels:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_kernels.cpp O/overscreening_breakdown_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o test_ob_kernels.bin 2>results.make

test_ob_kernels_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_kernels_var_prec.cpp O/overscreening_breakdown_ker_var_prec.o -o test_ob_kernels_var_prec.bin 2>results.make


test_ob_problem:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_problem.cpp O/overscreening_breakdown_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o test_ob_problem.bin 2>results.make

test_ob_problem_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_problem_var_prec.cpp O/overscreening_breakdown_ker_var_prec.o -o test_ob_problem_var_prec.bin $(NVOPENMP) 2>results.make


ob_view:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/view_solution.cpp O/overscreening_breakdown_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) -o ob_view_solution.bin 2>results.make


test_ob_newton:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_newton.cpp O/overscreening_breakdown_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) $(LLAPACK) -o test_ob_newton.bin 2>results.make

test_ob_newton_deflation:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/overscreening_breakdown/test_newton_deflation.cpp O/overscreening_breakdown_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) $(LLAPACK) -o test_ob_newton_deflation.bin 2>results.make


test_ob_newton_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_newton_var_prec.cpp O/overscreening_breakdown_ker_var_prec.o $(NVOPENMP) -o test_ob_newton_var_prec.bin 2>results.make

test_ob_newton_deflation_var_prec:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/test_newton_deflation_var_prec.cpp O/overscreening_breakdown_ker_var_prec.o $(NVOPENMP) -o test_ob_newton_deflation_var_prec.bin 2>results.make


ob_json:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/overscreening_breakdown/ob_bd_json.cpp O/overscreening_breakdown_ker.o O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o $(LCUDA) $(LIBS3) $(LLAPACK) $(LBOOST) $(LIBBOOST) -o ob_bd_json.bin 2>results.make

# test small problems time stepper:
test_vdp_time_stepping:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/time_stepper/tests/vdp_1.cpp $(OPENMP) -o test_vdp.bin 2>results.make
test_rossler_time_stepping:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/time_stepper/tests/rossler_1.cpp $(OPENMP) -o test_rossler.bin 2>results.make

# test periodic orbits:
test_rossler_to_section:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_to_section.cpp $(OPENMP) -o test_rossler_to_section.bin 2>results.make

test_glued_nonlinear_operator:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/glued_nonlinear_operator_and_jacobian.cpp $(OPENMP) -o test_glued_nonlinear_operator_and_jacobian.bin 2>results.make

test_glued_poincare_map_linear_operator:
	$(G++) $(G++FLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/glued_poincare_map_linear_operator.cpp $(OPENMP) -o test_glued_poincare_map_linear_operator.bin 2>results.make

test_rossler_periodic_orbit_cpu:
	$(G++)  $(G++FLAGS)  $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_periodic_cpu.cpp $(OPENMP) -o test_rossler_periodic_cpu.bin 2>results.make

test_rossler_periodic_orbit_gpu:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_operator_ker.cu -c -o O/rossler_operator_kernels.o 2>results.make
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) $(ICUDA) source/periodic_orbit/tests/rossler_periodic_gpu.cpp O/gpu_reduction_ogita_kernels.o O/gpu_vector_operations_kernels.o O/gpu_matrix_vector_operations_kernels.o O/rossler_operator_kernels.o -o test_rossler_periodic_gpu.bin $(LCUDA) $(LIBS3) $(LLAPACK) 2>results.make

# make all common kernels
ker:
	make gpu_matrix_vector_operations_ker gpu_vector_operations_ker gpu_reduction_ogita_ker abc_flow_ker Kolmogorov_3D_ker Taylor_Green_ker

