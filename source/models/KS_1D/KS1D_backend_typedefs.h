#ifndef __KS1D_BACKEND_TYPEDEFS_H__
#define __KS1D_BACKEND_TYPEDEFS_H__

#ifndef Blocks_x_
#    define Blocks_x_ 64
#endif

#if defined( KS1D_VECTOR_BACKEND_OMP )
#    include <nmfd/operations/scfd_vector_space.h>
#    include <external_libraries/fft_facade_fftw.h>
#    include <scfd/backend/omp.h>
#elif defined( KS1D_VECTOR_BACKEND_CUDA )
#    include <nmfd/operations/scfd_vector_space.h>
#    include <external_libraries/fft_facade_cufft.h>
#    include <scfd/backend/cuda.h>
#elif defined( KS1D_VECTOR_BACKEND_HIP )
#    include <nmfd/operations/scfd_vector_space.h>
#    include <external_libraries/fft_facade_hipfft.h>
#    include <scfd/backend/hip.h>
#else
#    error "Select KS1D_VECTOR_BACKEND_OMP, KS1D_VECTOR_BACKEND_CUDA, or KS1D_VECTOR_BACKEND_HIP."
#endif

#include <scfd/utils/log.h>

#if defined( KS1D_VECTOR_BACKEND_OMP )
#    define KS1D_BACKEND_NAME "scfd_omp_fftw"
typedef SCALAR_TYPE                                                                   real;
typedef nmfd::operations::scfd_vector_space<scfd::backend::omp<std::ptrdiff_t>, real> vec_ops_real;
typedef external_libraries::fft::fftw_backend                                         fft_backend_t;
#elif defined( KS1D_VECTOR_BACKEND_CUDA )
#    define KS1D_BACKEND_NAME "scfd_cuda_cufft"
typedef SCALAR_TYPE                                                                    real;
typedef nmfd::operations::scfd_vector_space<scfd::backend::cuda<std::ptrdiff_t>, real> vec_ops_real;
typedef external_libraries::fft::cufft_backend                                         fft_backend_t;
#elif defined( KS1D_VECTOR_BACKEND_HIP )
#    define KS1D_BACKEND_NAME "scfd_hip_hipfft"
typedef SCALAR_TYPE                                                                   real;
typedef nmfd::operations::scfd_vector_space<scfd::backend::hip<std::ptrdiff_t>, real> vec_ops_real;
typedef external_libraries::fft::hipfft_backend                                       fft_backend_t;
#endif

typedef scfd::utils::log_std log_t;

#endif
