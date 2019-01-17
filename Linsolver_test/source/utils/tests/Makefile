test_constant_data_cpp: test_constant_data.cpp 
	g++ -I../.. test_constant_data.cpp -o test_constant_data_cpp
test_constant_data_cu: test_constant_data.cu test_constant_data2.cu
	nvcc -I../.. test_constant_data.cu test_constant_data2.cu -o test_constant_data_cu
logged_obj: logged_obj.cpp
	g++ -I../.. logged_obj.cpp -o logged_obj
main_try_catch: main_try_catch.cpp
	g++ -I../.. main_try_catch.cpp -o main_try_catch
cuda_test: cuda_test.cu
	nvcc -I../.. cuda_test.cu -o cuda_test