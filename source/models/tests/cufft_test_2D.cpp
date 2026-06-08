#include <cmath>
#include <limits>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include "cufft_test_kernels.h"
#include <common/macros.h>
#include <common/file_operations.h>
#include <scfd/utils/cuda_ownutils.h>

int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef cufft_wrap_C2C<real>::complex_type complex_type;

    bool flag_ok = true;
    scfd::utils::init_cuda(-1);

    int N = file_operations::read_matrix_size_square("./dat_files/A.dat");//"testFFT.dat"
    size_t Nx=N;
    size_t Ny=N;
    printf("Nx=%i, Ny=%i\n", Nx, Ny);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    cufft_wrap_R2C<real> *CUFFT2_R = new cufft_wrap_R2C<real>(Nx, Ny);
    size_t My=CUFFT2_R->get_reduced_size();

    complex *A_hat_reduced_d;
    real *AR, *AI, *AR_d, *AI_d, *A, *A1, *A_d, *A1_d;
    A_hat_reduced_d = device_allocate<complex>(Nx*My);
    
    A=(real*)malloc(sizeof(real)*Nx*Ny);
    A1=(real*)malloc(sizeof(real)*Nx*Ny);
    AR=(real*)malloc(sizeof(real)*Nx*My);
    AI=(real*)malloc(sizeof(real)*Nx*My);
    
    A_d = device_allocate<real>(Nx*Ny);
    A1_d = device_allocate<real>(Nx*Ny);
    AR_d = device_allocate<real>(Nx*My);
    AI_d = device_allocate<real>(Nx*My);


    file_operations::read_matrix<real>("./dat_files/A.dat",  Nx, Ny, A);

    host_2_device_cpy<real>(A_d, A, Nx*Ny);
   
    CUFFT2_R->fft(A_d, A_hat_reduced_d);
    convert_2R_values<real, complex_type>(Nx*My, (complex_type*)A_hat_reduced_d, AR_d, AI_d);
    CUFFT2_R->ifft(A_hat_reduced_d, A1_d);
    
    device_2_host_cpy<real>(A1, A1_d, Nx*Ny);
    real res_norm_R=0.0;
    for(int j=0;j<Nx*Ny;j++)
    {   
        real diff=A1[j]/Nx/Ny-A[j];
        res_norm_R+=diff*diff;
    }
    res_norm_R = std::sqrt(res_norm_R);
    printf("R2C: %le \n", (double) res_norm_R );
    if(res_norm_R > std::sqrt( std::numeric_limits<real>::epsilon() ) )
    {
        flag_ok = false;
        std::cout << "norm exceeds threshold." << std::endl;
    }


    device_2_host_cpy<real>(AR, AR_d, Nx*My);
    device_2_host_cpy<real>(AI, AI_d, Nx*My);
    file_operations::write_matrix<real*>("testFFT_R.dat",  My, Nx, AR);
    file_operations::write_matrix<real*>("testFFT_I.dat",  My, Nx, AI);
    file_operations::write_matrix<real*>("testFFT_t.dat",  Nx, My, A);

    cudaFree(A_d);
    cudaFree(A1_d);
    cudaFree(AR_d);
    cudaFree(AI_d);
    cudaFree(A_hat_reduced_d);
    
    free(A);
    free(A1);
    free(AR);
    free(AI);

    delete CUBLAS;
    delete CUFFT2_R;
    if(flag_ok)
    {
        std::cout << "PASS" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
}