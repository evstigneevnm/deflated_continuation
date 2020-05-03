#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_impl.cuh>

#include <thrust/complex.h>



template struct nonlinear_operators::Kolmogorov_3D_ker<float, float*, thrust::complex<float>, thrust::complex<float>*>;
template struct nonlinear_operators::Kolmogorov_3D_ker<double, double*, thrust::complex<double>, thrust::complex<double>* >;

