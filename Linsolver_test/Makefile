cpu_1st_call_alloc_test_float: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	g++ -I source/ -DSCALAR_TYPE=float -O3 source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o cpu_1st_call_alloc_test_float.bin
cpu_1st_call_alloc_test_double: source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp
	g++ -I source/ -DSCALAR_TYPE=double -O3 source/numerical_algos/lin_solvers/tests/cpu_1st_call_alloc_test.cpp -o cpu_1st_call_alloc_test_double.bin
SM_test: source/SM_test.cpp
	g++ -I source/ -DSCALAR_TYPE=double -g source/SM_test.cpp -o SM_test.bin
cufft_test_double: source/cufft_test.cpp
	nvcc -I source/ -g source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=double -g source/cufft_test.cpp O/cufft_test_kernels.o -lcufft -o cufft_test.bin 2>results_make.txt
cufft_test_float: source/cufft_test.cpp
	nvcc -I source/ -g source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=float -g source/cufft_test.cpp O/cufft_test_kernels.o -lcufft -o cufft_test.bin 2>results_make.txt	
cufft_test_2D_double: source/cufft_test_2D.cpp
	nvcc -I source/ -g source/cufft_test_kernels.cu -c -o O/cufft_test_kernels.o 2>results_make.txt
	nvcc -I source/ -DSCALAR_TYPE=double -g source/cufft_test_2D.cpp O/cufft_test_kernels.o -lcufft -o cufft_test_2D.bin 2>results_make.txt
gpu_vector_operations:
	nvcc -I source/ -g source/gpu_vector_operations_kernels.cu -c -o O/gpu_vector_operations_kernels.o 2>results_make.txt
KS_2D_ker:
	nvcc -I source/ -g source/nonlinear_operators/Kuramoto_Sivashinskiy_2D_ker.cu -c -o O/Kuramoto_Sivashinskiy_2D_ker.o 2>results_make.txt
KS_2D_double: source/test_Kuramoto_Sivashinskiy_2D_RHS.cpp
	nvcc -I source/ -DSCALAR_TYPE=double -g source/test_Kuramoto_Sivashinskiy_2D_RHS.cpp O/Kuramoto_Sivashinskiy_2D_ker.o O/gpu_vector_operations_kernels.o -lcufft -lcublas -o test_Kuramoto_Sivashinskiy_2D_RHS.bin 2>results_make.txt
