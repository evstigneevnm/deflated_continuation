SCALAR_TYPE =-DSCALAR_TYPE=double
CUDA_ROOT_PATH = /usr/local/cuda
GCC_ROOT_PATH = /usr/bin
CUDA_ARCH =  -gencode arch=compute_52,code=sm_52 #-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
CPPSTD = c++14
TARGET = -O3
BOOST_ROOT_PATH = /usr/local/boost
OPENBLAS_ROOT_PATH = /opt/OpenBLAS
