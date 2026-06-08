#ifndef __LINSOLVER_EXTERNAL_CUSOLVER_WRAP_H__
#define __LINSOLVER_EXTERNAL_CUSOLVER_WRAP_H__

#include <scfd/external_libraries/cusolver_wrap.h>
#include <external_libraries/cublas_wrap.h>

class cusolver_wrap : public scfd::cusolver_wrap
{
public:
    cusolver_wrap() : scfd::cusolver_wrap( false, false )
    {
    }

    explicit cusolver_wrap( bool plot_info ) : scfd::cusolver_wrap( plot_info, false )
    {
    }

    explicit cusolver_wrap( cublas_wrap *cublas ) : scfd::cusolver_wrap( static_cast<scfd::cublas_wrap *>( cublas ), false )
    {
    }

    cusolver_wrap( cublas_wrap *cublas, bool do_set_inst )
        : scfd::cusolver_wrap( static_cast<scfd::cublas_wrap *>( cublas ), do_set_inst )
    {
    }
};

#endif
