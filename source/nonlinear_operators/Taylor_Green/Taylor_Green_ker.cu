#include <nonlinear_operators/Taylor_Green/Taylor_Green_impl.cuh>
#include <thrust/complex.h>



template struct nonlinear_operators::Taylor_Green_ker<float, float*, thrust::complex<float>, thrust::complex<float>*>;
template struct nonlinear_operators::Taylor_Green_ker<double, double*, thrust::complex<double>, thrust::complex<double>* >;

