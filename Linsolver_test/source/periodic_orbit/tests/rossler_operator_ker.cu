#include "rossler_operator_impl.cuh"


template struct nonlinear_operators::rossler_operator_ker<float, float* >;
template struct nonlinear_operators::rossler_operator_ker<double, double* >;

