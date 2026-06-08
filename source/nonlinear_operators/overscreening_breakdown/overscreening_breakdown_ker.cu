#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_impl.cuh>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>

using VecD =  gpu_vector_operations<double>;
using MatD = gpu_matrix_vector_operations<typename VecD::scalar_type, typename VecD::vector_type>;

template struct nonlinear_operators::overscreening_breakdown_ker<VecD, MatD>;

using VecF =  gpu_vector_operations<float>;
using MatF = gpu_matrix_vector_operations<typename VecF::scalar_type, typename VecF::vector_type>;
template struct nonlinear_operators::overscreening_breakdown_ker<VecF, MatF>;

