#ifndef __LINSOLVER_EXTERNAL_CUBLAS_WRAP_H__
#define __LINSOLVER_EXTERNAL_CUBLAS_WRAP_H__

#include <scfd/external_libraries/cublas_wrap.h>

namespace cublas_complex_types
{
template <class T>
using cublas_cuComplex_type_hlp = scfd::cublas_complex_types::cublas_cuComplex_type_hlp<T>;
}

namespace cublas_real_types
{
template <class T>
using cublas_real_type_hlp = scfd::cublas_real_types::cublas_real_type_hlp<T>;
}

class cublas_wrap : public scfd::cublas_wrap
{
public:
    cublas_wrap() : scfd::cublas_wrap( false )
    {
    }

    explicit cublas_wrap( bool plot_info ) : scfd::cublas_wrap( plot_info, false )
    {
    }

    cublas_wrap( bool plot_info, bool do_set_inst ) : scfd::cublas_wrap( plot_info, do_set_inst )
    {
    }
};

#endif
