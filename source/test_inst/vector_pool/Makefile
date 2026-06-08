cpp:
	g++ --std=c++11 vector_pool.cpp -o vector_pool.bin 1>results_make.txt
cu:
	nvcc --std=c++11 vector_pool.cu -g -I../../../source/ ../../../O/gpu_vector_operations_kernels.o -lcublas -lcurand -o run.bin 2>results_make.txt