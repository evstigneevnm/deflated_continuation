test_constant_data_cpp: test_constant_data.cpp 
	g++ -I../.. test_constant_data.cpp -o test_constant_data_cpp
test_constant_data_cu: test_constant_data.cu test_constant_data2.cu
	nvcc -I../.. test_constant_data.cu test_constant_data2.cu -o test_constant_data_cu
cuda_test: cuda_test.cu
	nvcc -I../.. cuda_test.cu -o cuda_test
