deb:
	nvcc -std=c++11 -Wno-deprecated-gpu-targets -g -G -I ../../../source/ test_container.cu ../../../O/gpu_vector_operations_kernels.o -lcublas -lcurand -o run.bin 2>results_make.txt