#once i was framed by autoconf (found two different gcc's for different libraries, cost me 2 days), so now i use all by hand
#set these variables for your local machine
SCALAR_TYPE = -DSCALAR_TYPE=double
NVCC = /usr/local/cuda/bin/nvcc
GCC = /usr/bin/gcc

NVCCFLAGS = -Wno-deprecated-gpu-targets -g -arch=sm_20 -std=c++11 
GCCFLAGS = -g -std=c++11
ICUDA = -I/usr/local/cuda
IPROJECT = -I source/

LCUDA = -L/usr/local/cuda/lib64
LIBS1 = -lcublas -lcurand 
LIBS2 = -lcufft -lcublas -lcurand 

#all targets
cpu_1st_call_alloc_test: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	$(GCC) $(GCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o cpu_1st_call_alloc_test_float.bin

SM_test: source/SM_test.cpp
	$(GCC) $(GCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/SM_test.cpp -o SM_test.bin 2>results_make.txt

lin_solvers_test: source/linear_solvers_test.cpp
	$(GCC) $(GCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/linear_solvers_test.cpp -o linear_solvers_test.bin

cufft_test: source/cufft_test.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -g source/cufft_test.cpp O/cufft_test_kernels.o $(LIBS2) -o cufft_test.bin 2>results_make.txt

cufft_test_2D: source/cufft_test_2D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/cufft_test_2D.cpp O/cufft_test_kernels.o $(LIBS2) -o cufft_test_2D.bin 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/cufft_test_2D_1.cpp O/cufft_test_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o cufft_test_2D_1.bin 2>results_make.txt

gpu_vector_operations:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/test_vector_operations.cpp O/gpu_vector_operations_kernels.o $(LIBS2) -o test_vector_operations.bin 2>results_make.txt

gpu_vector_operations_ker:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt

KS_2D_ker:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt

KS_2D: source/Kuramoto_Sivashinskiy_2D_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/Kuramoto_Sivashinskiy_2D_RHS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_RHS.bin 2>results_make.txt

deflation_KS_2D: source/test_deflation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/test_deflation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_deflation.bin 2>results_make.txt

cont_def_KS_2D: source/test_deflation_continuation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/test_deflation_continuation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_deflation_continuation.bin 2>results_make.txt

circle_ker: source/nonlinear_operators/circle/circle_ker.cu
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/nonlinear_operators/circle/circle_ker.cu -c -o O/circle_ker.o 2>results_make.txt

cont_def_circle: source/circle_test_deflation_continuation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/circle_test_deflation_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o $(LIBS1) -o circle_test_deflation_continuation.bin 2>results_make.txt	

circle_continuation: source/circle_continuation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/circle_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o $(LIBS1) -o circle_continuation.bin 2>results_make.txt	

KS_continuation: source/KS_continuation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/KS_continuation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o KS_continuation.bin 2>results_make.txt	