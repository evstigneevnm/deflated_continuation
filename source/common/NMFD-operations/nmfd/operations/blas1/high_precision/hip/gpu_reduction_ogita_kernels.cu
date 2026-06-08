#if defined(NMFD_HIGH_PRECISION_BLAS1_ENABLE_HIP_NVIDIA) || (defined(__HIPCC__) && (defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)))
#include <nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu>
#else
#error "HIP high-precision Ogita kernels are currently implemented only for the HIP NVIDIA backend."
#endif
