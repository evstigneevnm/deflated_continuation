#include <nonlinear_operators/abc_flow/abc_flow_impl.cuh>
#include <thrust/complex.h>



template struct nonlinear_operators::abc_flow_ker<float, float*, thrust::complex<float>, thrust::complex<float>*>;
template struct nonlinear_operators::abc_flow_ker<double, double*, thrust::complex<double>, thrust::complex<double>* >;

