#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <gpu_vector_operations.h>
#include "cufft_test_kernels.h"
#include "macros.h"
#include "file_operations.h"

int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_R_type;
    typedef gpu_vector_operations<complex> vec_ops_C_type;

    init_cuda(-1);
    size_t N=8;
    size_t Nx=N;
    size_t Ny=N;
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    cufft_wrap_R2C<real> *CUFFT2_R = new cufft_wrap_R2C<real>(Nx, Ny);
    size_t My=CUFFT2_R->get_reduced_size();
    vec_ops_C_type *vec_ops_C=new vec_ops_C_type(Nx*My, CUBLAS);
    vec_ops_R_type *vec_ops_R=new vec_ops_R_type(Nx*Ny, CUBLAS);
    vec_ops_C_type *vec_ops_CC=new vec_ops_C_type(Nx*Ny, CUBLAS);

    printf("Nx=%i, Ny=%i, My=%i\n", Nx, Ny, My);


    complex* AC=(complex*)malloc(sizeof(complex)*Nx*My);
    complex* ACC=(complex*)malloc(sizeof(complex)*Nx*Ny);
    real* A=(real*)malloc(sizeof(real)*Nx*Ny);

    real* A_d;
    complex* AC_d;
    complex* GC_d;
    complex* VC_d;
    complex *grad_x, *grad_y;
    complex *z1_d, *z2_d, *z3_d;

    vec_ops_CC->init_vector(z1_d); vec_ops_CC->start_use_vector(z1_d);
    vec_ops_CC->init_vector(z2_d); vec_ops_CC->start_use_vector(z2_d);
    vec_ops_CC->init_vector(z3_d); vec_ops_CC->start_use_vector(z3_d);


    vec_ops_C->init_vector(AC_d); vec_ops_C->start_use_vector(AC_d);
    vec_ops_C->init_vector(GC_d); vec_ops_C->start_use_vector(GC_d);
    vec_ops_C->init_vector(VC_d); vec_ops_C->start_use_vector(VC_d);
    vec_ops_C->init_vector(grad_x); vec_ops_C->start_use_vector(grad_x);
    vec_ops_C->init_vector(grad_y); vec_ops_C->start_use_vector(grad_y);

    vec_ops_R->init_vector(A_d); vec_ops_R->start_use_vector(A_d);


    gradient_Fourier<complex>(Nx, My, Ny, z1_d, z2_d);
    vec_ops_CC->mul_pointwise(1.0, z1_d, 1.0, z2_d, z3_d);
    complexreal_to_real<real,complex>(Nx*Ny, (const complex*&)z3_d, A_d);
    CUFFT2_R->fft(A_d,GC_d);

    device_2_host_cpy<complex>(ACC, z3_d, Nx*Ny);
    file_operations::write_matrix<complex>("AC.dat",  Nx, Ny, ACC,2);
    device_2_host_cpy<complex>(AC, GC_d, Nx*My);
    file_operations::write_matrix<complex>("ACf.dat",  Nx, My, AC, 2);
    device_2_host_cpy<real>(A, A_d, Nx*Ny);
    file_operations::write_matrix<real>("A.dat",  Nx, Ny, A,2);
    
    CUFFT2_R->ifft(GC_d, A_d);
    vec_ops_R->scale(1.0/Nx/Ny, A_d);
    device_2_host_cpy<real>(A, A_d, Nx*Ny);
    file_operations::write_matrix<real>("A1.dat",  Nx, Ny, A,2);
   






    //host_2_device_cpy<real>(A_d, A, Nx*Ny);
    //device_2_host_cpy<real>(AI, AI_d, Nx*My);
    //file_operations::write_matrix<real>("testFFT_R.dat",  My, Nx, AR);
    

    vec_ops_CC->free_vector(z1_d);
    vec_ops_CC->free_vector(z2_d);
    vec_ops_CC->free_vector(z3_d);
    vec_ops_C->free_vector(GC_d);    
    vec_ops_C->free_vector(AC_d);
    vec_ops_C->free_vector(VC_d);
    vec_ops_C->free_vector(grad_x);
    vec_ops_C->free_vector(grad_y);
    vec_ops_R->free_vector(A_d);

    free(A);
    free(AC);
    free(ACC);
    
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops_CC;
    delete CUBLAS;
    delete CUFFT2_R;

    return 0;
}