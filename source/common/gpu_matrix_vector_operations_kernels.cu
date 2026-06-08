#include <common/gpu_matrix_vector_operations_impl.cuh>
#include <thrust/complex.h>

template struct gpu_matrix_vector_operations<float, float*>;
template struct gpu_matrix_vector_operations<double, double*>;
template struct gpu_matrix_vector_operations< thrust::complex<float>, thrust::complex<float>* >;
template struct gpu_matrix_vector_operations< thrust::complex<double>, thrust::complex<double>*  >;
