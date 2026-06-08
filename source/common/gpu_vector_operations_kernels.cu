#include <common/gpu_vector_operations_impl.cuh>
#include <thrust/complex.h>

template struct gpu_vector_operations<float>;
template struct gpu_vector_operations<double>;
template struct gpu_vector_operations< thrust::complex<float> >;
template struct gpu_vector_operations< thrust::complex<double> >;
