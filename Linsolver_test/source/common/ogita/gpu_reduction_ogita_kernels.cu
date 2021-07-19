#include <common/ogita/gpu_reduction_ogita_impl.cuh>
#include <thrust/complex.h>


template class gpu_reduction_ogita<float, float*>;
template class gpu_reduction_ogita<double, double*>;
template class gpu_reduction_ogita< thrust::complex<float>, thrust::complex<float>* >;
template class gpu_reduction_ogita< thrust::complex<double>, thrust::complex<double>* >;