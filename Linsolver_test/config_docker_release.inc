TARGET_GCC = -O3
TARGET_NVCC = -O3
MPI_ROOT_PATH = /usr/lib/x86_64-linux-gnu/openmpi
CUDA_ROOT_PATH = /usr/local/cuda-11.6
BOOST_ROOT_PATH = /opt/boost
CUDA_ARCH = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86


