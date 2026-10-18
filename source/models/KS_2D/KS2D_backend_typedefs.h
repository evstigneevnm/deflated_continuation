#ifndef __KS2D_BACKEND_TYPEDEFS_H__
#define __KS2D_BACKEND_TYPEDEFS_H__

#if defined( KS2D_VECTOR_BACKEND_OMP )
#    include <nmfd/operations/scfd_vector_space.h>
#    include <external_libraries/fft_facade_fftw.h>
#    include <scfd/backend/omp.h>
#elif defined( KS2D_VECTOR_BACKEND_CUDA )
#    include <nmfd/operations/scfd_vector_space.h>
#    include <external_libraries/fft_facade_cufft.h>
#    include <scfd/backend/cuda.h>
#else
#    error "KS2D supports KS2D_VECTOR_BACKEND_OMP and KS2D_VECTOR_BACKEND_CUDA."
#endif

#include <scfd/utils/log.h>

using real = SCALAR_TYPE;

#if defined( KS2D_VECTOR_BACKEND_OMP )
#    define KS2D_BACKEND_NAME "scfd_omp_fftw"
using vec_ops_real  = nmfd::operations::scfd_vector_space<scfd::backend::omp<std::ptrdiff_t>, real>;
using fft_backend_t = external_libraries::fft::fftw_backend;
#elif defined( KS2D_VECTOR_BACKEND_CUDA )
#    define KS2D_BACKEND_NAME "scfd_cuda_cufft"
using vec_ops_real  = nmfd::operations::scfd_vector_space<scfd::backend::cuda<std::ptrdiff_t>, real>;
using fft_backend_t = external_libraries::fft::cufft_backend;
#endif

using log_t = scfd::utils::log_std;

#endif
