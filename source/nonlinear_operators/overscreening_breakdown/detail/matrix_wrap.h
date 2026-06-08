#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_DETAIL_MATRIX_WRAP_H__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_DETAIL_MATRIX_WRAP_H__

#include <contrib/scfd/include/scfd/utils/device_tag.h>

namespace detail
{
//struct under external managment!
template<class T, class T_mat>
struct matrix_wrap
{
    matrix_wrap(size_t rows_p, size_t cols_p):
    rows(rows_p), cols(cols_p)
    {}

    __DEVICE_TAG__ inline T& operator()(size_t j, size_t k) // __attribute__((__always_inline__))
    {
        return data[rows*k+j];
    }
    size_t rows;
    size_t cols;
    T_mat data;        
};

}


#endif