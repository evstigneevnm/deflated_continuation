#include <nonlinear_operators/Kolmogorov_flow_2D/Kolmogorov_2D_impl.cuh>

#include <thrust/complex.h>



template struct nonlinear_operators::Kolmogorov_2D_ker<float, float*, thrust::complex<float>, thrust::complex<float>*>;
template struct nonlinear_operators::Kolmogorov_2D_ker<double, double*, thrust::complex<double>, thrust::complex<double>* >;

