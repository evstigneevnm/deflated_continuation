#ifndef __KS1D_BACKEND_TYPEDEFS_H__
#define __KS1D_BACKEND_TYPEDEFS_H__

#ifndef Blocks_x_
#define Blocks_x_ 64
#endif

#if defined(KS1D_VECTOR_BACKEND_OMP)
#include <common/scfd_vector_operations.h>
#include <external_libraries/fft_facade_fftw.h>
#include <scfd/backend/omp.h>
#elif defined(KS1D_VECTOR_BACKEND_CUDA)
#include <common/scfd_vector_operations.h>
#include <external_libraries/fft_facade_cufft.h>
#include <scfd/backend/cuda.h>
#else
#error "KS1D first pass supports KS1D_VECTOR_BACKEND_OMP and KS1D_VECTOR_BACKEND_CUDA."
#endif

#include <scfd/utils/log.h>

#if defined(KS1D_VECTOR_BACKEND_OMP)
#define KS1D_BACKEND_NAME "scfd_omp_fftw"
typedef SCALAR_TYPE real;
typedef scfd_vector_operations<scfd::backend::omp, real> vec_ops_real;
typedef external_libraries::fft::fftw_backend fft_backend_t;
#elif defined(KS1D_VECTOR_BACKEND_CUDA)
#define KS1D_BACKEND_NAME "scfd_cuda_cufft"
typedef SCALAR_TYPE real;
typedef scfd_vector_operations<scfd::backend::cuda, real> vec_ops_real;
typedef external_libraries::fft::cufft_backend fft_backend_t;
#endif

typedef scfd::utils::log_std log_t;

#endif
