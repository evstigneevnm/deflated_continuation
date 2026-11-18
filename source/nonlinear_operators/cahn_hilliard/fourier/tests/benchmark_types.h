#ifndef CH_FOURIER_BENCHMARK_TYPES_H
#define CH_FOURIER_BENCHMARK_TYPES_H

#include "../diagnostics.h"
#if defined( TEST_VECTOR_BACKEND_CUDA )
#    include <external_libraries/fft_facade_cufft.h>
using ch_fft_backend = external_libraries::fft::cufft_backend;
#elif defined( TEST_VECTOR_BACKEND_HIP )
#    include <external_libraries/fft_facade_hipfft.h>
using ch_fft_backend = external_libraries::fft::hipfft_backend;
#else
#    include <external_libraries/fft_facade_fftw.h>
using ch_fft_backend = external_libraries::fft::fftw_backend;
#endif
#endif
