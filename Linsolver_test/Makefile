cpu_1st_call_alloc_test_float: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	g++ -I source/ -DSCALAR_TYPE=float -std=c++11 -O3 source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o cpu_1st_call_alloc_test_float.bin
cpu_1st_call_alloc_test_double: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	g++ -I source/ -DSCALAR_TYPE=double -std=c++11 -O3 source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o cpu_1st_call_alloc_test_double.bin
SM_test_rel: source/SM_test.cpp
	g++ -I source/ -DSCALAR_TYPE=double -std=c++11 -O3 source/SM_test.cpp -o SM_test.bin 2>results_make.txt
SM_test_deb: source/SM_test.cpp
	g++ -I source/ -DSCALAR_TYPE=double -std=c++11 -G -g source/SM_test.cpp -o SM_test.bin 2>results_make.txt
lin_solvers_test: source/linear_solvers_test.cpp
	g++ -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/linear_solvers_test.cpp -o linear_solvers_test.bin
cufft_test_double: source/cufft_test.cpp
	nvcc -I source/ -std=c++11 -g source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/cufft_test.cpp O/cufft_test_kernels.o -lcufft -o cufft_test.bin 2>results_make.txt
cufft_test_float: source/cufft_test.cpp
	nvcc -I source/ -std=c++11 -g source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=float -std=c++11 -g source/cufft_test.cpp O/cufft_test_kernels.o -lcufft -o cufft_test.bin 2>results_make.txt	
cufft_test_2D_double: source/cufft_test_2D.cpp
	nvcc -I source/ -std=c++11 -g source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/cufft_test_2D.cpp O/cufft_test_kernels.o -lcufft -lcublas -o cufft_test_2D.bin 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/cufft_test_2D_1.cpp O/cufft_test_kernels.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -o cufft_test_2D.bin 2>results_make.txt
gpu_vector_operations:
	nvcc -I source/ -std=c++11 -g -G source/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -g -G -lcurand source/test_vector_operations.cpp O/gpu_vector_operations_kernels.o -lcublas -o test_vector_operations.bin 2>results_make.txt
gpu_vector_operations_ker:
	nvcc -I source/ -std=c++11 -O3 source/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt
gpu_vector_operations_ker_deb:
	nvcc -I source/ -std=c++11 -g -G source/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt
KS_2D_ker_deb:
	nvcc -I source/ -std=c++11 -g -G source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt
KS_2D_ker:
	nvcc -I source/ -std=c++11 -O3 source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt
KS_2D_double: source/Kuramoto_Sivashinskiy_2D_RHS.cpp
	nvcc -I source/ -std=c++11 -g source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/Kuramoto_Sivashinskiy_2D_RHS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_RHS.bin 2>results_make.txt
KS_2D_float: source/Kuramoto_Sivashinskiy_2D_RHS.cpp
	nvcc -I source/ -std=c++11 -g source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=float -std=c++11 -g source/Kuramoto_Sivashinskiy_2D_RHS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_RHS.bin 2>results_make.txt

deflation_KS_2D_float: source/test_deflation.cpp
	nvcc -I source/ -DSCALAR_TYPE=float -std=c++11 -g source/test_deflation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation.bin 2>results_make.txt
deflation_KS_2D_double: source/test_deflation.cpp
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/test_deflation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation.bin 2>results_make.txt

deflation_KS_2D_float_rel: source/test_deflation.cpp
	nvcc -I source/ -DSCALAR_TYPE=float -std=c++11 -O3 source/test_deflation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation.bin 2>results_make.txt
deflation_KS_2D_double_rel: source/test_deflation.cpp
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -O3 source/test_deflation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation.bin 2>results_make.txt	

cont_def_KS_2D_float: source/test_deflation_continuation.cpp
	nvcc -I source/ -DSCALAR_TYPE=float -std=c++11 -g source/test_deflation_continuation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation_continuation.bin 2>results_make.txt
cont_def_KS_2D_double: source/test_deflation_continuation.cpp
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/test_deflation_continuation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation_continuation.bin 2>results_make.txt

cont_def_KS_2D_float_rel: source/test_deflation_continuation.cpp
	nvcc -I source/ -DSCALAR_TYPE=float -std=c++11 -O3 source/test_deflation_continuation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation_continuation.bin 2>results_make.txt
cont_def_KS_2D_double_rel: source/test_deflation_continuation.cpp
	nvcc -I source/ -DSCALAR_TYPE=double -std=c++11 -O3 source/test_deflation_continuation.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -lcurand -o test_deflation_continuation.bin 2>results_make.txt	

circle_ker_deb:
	nvcc -I source/ -std=c++11 -g -G source/nonlinear_operators/circle/circle_ker.cu -c -o O/circle_ker.o 2>results_make.txt
circle_ker:
	nvcc -I source/ -std=c++11 -O3 source/nonlinear_operators/circle/circle_ker.cu -c -o O/circle_ker.o 2>results_make.txt

cont_def_circle_float: source/circle_test_deflation_continuation.cpp
	nvcc -Wno-deprecated-gpu-targets -I source/ -DSCALAR_TYPE=float -std=c++11 -g  source/circle_test_deflation_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o -lcublas -lcurand -o circle_test_deflation_continuation.bin 2>results_make.txt
cont_def_circle_double: source/circle_test_deflation_continuation.cpp
	nvcc -Wno-deprecated-gpu-targets -I source/ -DSCALAR_TYPE=double -std=c++11 -g source/circle_test_deflation_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o -lcublas -lcurand -o circle_test_deflation_continuation.bin 2>results_make.txt	

cont_def_circle_float_rel: source/circle_test_deflation_continuation.cpp
	nvcc -Wno-deprecated-gpu-targets -I source/ -DSCALAR_TYPE=float -std=c++11 -O3 source/circle_test_deflation_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o -lcublas -lcurand -o circle_test_deflation_continuation.bin 2>results_make.txt
cont_def_circle_double_rel: source/circle_test_deflation_continuation.cpp
	nvcc -Wno-deprecated-gpu-targets -I source/ -DSCALAR_TYPE=double -std=c++11 -O3 source/circle_test_deflation_continuation.cpp O/circle_ker.o O/gpu_vector_operations_kernels.o -lcublas -lcurand -o circle_test_deflation_continuation.bin 2>results_make.txt	
