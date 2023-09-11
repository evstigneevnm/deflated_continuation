#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_impl.hpp>
#include <common/cpu_vector_operations_var_prec.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>

using VecD =  cpu_vector_operations_var_prec;
using MatD = cpu_matrix_vector_operations_var_prec<VecD>;
template struct nonlinear_operators::overscreening_breakdown_ker<VecD, MatD>;
