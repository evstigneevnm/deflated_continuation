#ifndef __LINSOLVER_EXTERNAL_CUFFT_WRAP_H__
#define __LINSOLVER_EXTERNAL_CUFFT_WRAP_H__

#include <scfd/external_libraries/cufft_wrap.h>

template <class T>
using complex_type_hlp = scfd::complex_type_hlp<T>;

template <class T>
using cufft_wrap_C2C = scfd::cufft_wrap_C2C<T>;

template <class T>
using cufft_wrap_R2C = scfd::cufft_wrap_R2C<T>;

#endif
