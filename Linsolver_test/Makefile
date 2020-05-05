#once i was framed by autoconf (found two different gcc's for different libraries, cost me 2 days), 
#so now i use all by hand

#set these variables for your local machine
SCALAR_TYPE =-DSCALAR_TYPE=double
NVCC = /usr/local/cuda/bin/nvcc
GCC = /usr/bin/gcc

NVCCFLAGS = -Wno-deprecated-gpu-targets -g -arch=sm_20 -std=c++11 
GCCFLAGS = -g -std=c++11
ICUDA = -I/usr/local/cuda
IPROJECT = -I source/
IBOOST = -I/usr/local/boost_1_66_0

LCUDA = -L/usr/local/cuda/lib64
LBOOST = -L/usr/local/boost_1_66_0/lib
LIBS1 = -lcublas -lcurand 
LIBS2 = -lcufft -lcublas -lcurand 
LIBBOOST = -lboost_serialization

#all targets
cpu_1st_call_alloc_test: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	$(GCC) $(GCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o cpu_1st_call_alloc_test_float.bin

SM_test: source/models/tests/SM_test.cpp
	$(GCC) $(GCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/tests/SM_test.cpp -o SM_test.bin 2>results_make.txt

lin_solvers_test: source/models/tests/linear_solvers_test.cpp
	$(GCC) $(GCCFLAGS) $(SCALAR_TYPE) $(IPROJECT) source/models/tests/linear_solvers_test.cpp -o linear_solvers_test.bin

cufft_test: source/models/tests/cufft_test.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) -g source/models/tests/cufft_test.cpp O/cufft_test_kernels.o $(LIBS2) -o cufft_test.bin 2>results_make.txt

cufft_test_2D: source/models/tests/cufft_test_2D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_2D.cpp O/cufft_test_kernels.o $(LIBS2) -o cufft_test_2D.bin 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/cufft_test_2D_1.cpp O/cufft_test_kernels.o O/gpu_vector_operations_kernels.o $(LIBS2) -o cufft_test_2D_1.bin 2>results_make.txt

gpu_vector_operations:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/tests/test_vector_operations.cpp O/gpu_vector_operations_kernels.o $(LIBS2) -o test_vector_operations.bin 2>results_make.txt

gpu_vector_operations_ker:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/common/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt

KS_2D_ker:
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt

KS_2D: source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_Kuramoto_Sivashinskiy_2D_RHS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_Kuramoto_Sivashinskiy_2D_RHS.bin 2>results_make.txt

deflation_KS_2D: source/models/KS_2D/test_deflation_KS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_KS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_deflation.bin 2>results_make.txt

cont_def_KS_2D: source/models/KS_2D/test_deflation_continuation_KS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KS_2D/test_deflation_continuation_KS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_deflation_continuation.bin 2>results_make.txt

circle_ker: source/nonlinear_operators/circle/circle_ker.cu
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/nonlinear_operators/circle/circle_ker.cu -c -o O/circle_ker.o 2>results_make.txt

cont_def_circle: source/circle_test_deflation_continuation.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/circle_test_deflation_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o $(LIBS1) -o circle_test_deflation_continuation.bin 2>results_make.txt	

circle_bd: source/models/circle/circle_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/circle/circle_bd.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o $(LIBS1) -o circle_bd.bin 2>results_make.txt	

KS_bd: source/models/KS_2D/KS_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KS_2D/KS_bd.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o $(LCUDA) $(LIBBOOST) $(LIBS2)  -o KS_bd.bin 2>results_make.txt	

Kolmogorov_3D_ker: source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.cu -c -o O/Kolmogorov_3D_ker.o 2>results_make.txt

Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_RHS.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_RHS.cpp O/Kolmogorov_3D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o test_Kolmogorov_3D_RHS.bin 2>results_make.txt

Kolmogorov_3D_all: Kolmogorov_3D_ker Kolmogorov_3D

newton_Kolmogorov_3D: source/models/KF_3D/test_Kolmogorov_3D_Newton.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_Kolmogorov_3D_Newton.cpp O/Kolmogorov_3D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o newton_Kolmogorov_3D.bin 2>results_make.txt

deflation_Kolmogorov_3D: source/models/KF_3D/test_deflation_Kolmogorov_3D.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) source/models/KF_3D/test_deflation_Kolmogorov_3D.cpp O/Kolmogorov_3D_ker.o O/gpu_vector_operations_kernels.o $(LIBS2) -o deflation_Kolmogorov_3D.bin 2>results_make.txt

KF3D_bd: source/models/KF_3D/KF3D_bd.cpp
	$(NVCC) $(NVCCFLAGS) $(SCALAR_TYPE) $(ICUDA) $(IPROJECT) $(IBOOST) source/models/KF_3D/KF3D_bd.cpp O/Kolmogorov_3D_ker.o O/gpu_vector_operations_kernels.o $(LCUDA) $(LIBBOOST) $(LIBS2)  -o KF3D_bd.bin 2>results_make.txt	
