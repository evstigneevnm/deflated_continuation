#ifndef __NMFD_OPERATIONS_BLAS1_HIGH_PRECISION_HIP_GPU_REDUCTION_OGITA_H__
#define __NMFD_OPERATIONS_BLAS1_HIGH_PRECISION_HIP_GPU_REDUCTION_OGITA_H__

#if defined(NMFD_HIGH_PRECISION_BLAS1_ENABLE_HIP_NVIDIA) || (defined(__HIPCC__) && (defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)))
#include <nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita.h>
#else
#error "HIP high-precision Ogita reduction is currently implemented only for the HIP NVIDIA backend."
#endif

#endif
