TARGET_GCC = -O3
TARGET_NVCC = -O3
HYPRE_BUILD_TYPE = release

MPI_ROOT_PATH = /usr/local/mpi
GCC_ROOT_PATH = /usr/bin
CUDA_ROOT_PATH = /usr/local/cuda
BOOST_ROOT_PATH = /usr/local/boost
OPENBLAS_ROOT_PATH = /usr/local/openblas

# Tesla V100 (Volta).
CUDA_ARCH_LIST = 70

CPPSTD = c++17
REAL_TYPE = double
SCALAR_TYPE = -DSCALAR_TYPE=double
BUILD_DIR = ./build
