#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <gpu_vector_operations.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D.h>
#include "macros.h"
#include "file_operations.h"


int main(int argc, char const *argv[])
{
    int N=2560;
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex;
    init_cuda(-1);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    gpu_vector_operations_real *vec_ops_R = new gpu_vector_operations_real(N, CUBLAS);
    gpu_vector_operations_complex *vec_ops_C = new gpu_vector_operations_complex(N, CUBLAS);

    real *u1_d, *u2_d, *u3_d;
    complex *z1_d, *z2_d, *z3_d;
    vec_ops_R->init_vector(u1_d); vec_ops_R->init_vector(u2_d); vec_ops_R->init_vector(u3_d);
    vec_ops_C->init_vector(z1_d); vec_ops_C->init_vector(z2_d); vec_ops_C->init_vector(z3_d);
    vec_ops_R->start_use_vector(u1_d); vec_ops_R->start_use_vector(u2_d); vec_ops_R->start_use_vector(u3_d);
    vec_ops_C->start_use_vector(z1_d); vec_ops_C->start_use_vector(z2_d); vec_ops_C->start_use_vector(z3_d);

    vec_ops_R->assign_scalar(0.5, u1_d); vec_ops_C->assign_scalar(complex(0.5), z1_d);
    printf("is valid? %i, %i\n",vec_ops_R->check_is_valid_number(u1_d),vec_ops_C->check_is_valid_number(z1_d));



    vec_ops_R->free_vector(u1_d); vec_ops_R->free_vector(u2_d); vec_ops_R->free_vector(u3_d); 
    vec_ops_C->free_vector(z1_d); vec_ops_C->free_vector(z2_d); vec_ops_C->free_vector(z3_d); 

    delete CUBLAS;
    delete vec_ops_R;
    return 0;
}