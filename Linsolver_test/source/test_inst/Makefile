deb:
	nvcc -Wno-deprecated-gpu-targets -g -I ../../source/ class_file_impl.cu -c -o class_file_impl.o
	nvcc -Wno-deprecated-gpu-targets -g -I ../../source/ run.cpp class_file_impl.o -o run.bin