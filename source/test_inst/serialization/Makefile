#REINSTAL boost_1_66 with compiled libraries!

NVCC = /usr/local/cuda/bin/nvcc
GCC = /usr/bin/gcc

CCFLAGS = -g -std=c++11
IPROJECT = -I../../../source -I/usr/local/boost/include
LBOOST = -L/usr/local/boost/lib

LCUDA = -L/usr/local/cuda/lib64
LIBS1 = -lcublas -lcurand -lboost_serialization


ser:
	$(NVCC) ${CCFLAGS} ${IPROJECT} demo1.cpp -o demo1.bin ${LBOOST} ${LCUDA} ${LIBS1}  ../../../O/gpu_reduction_ogita_kernels.o ../../../O/gpu_vector_operations_kernels.o 2>result_make.txt
