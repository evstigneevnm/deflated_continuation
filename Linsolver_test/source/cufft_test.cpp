#include <cmath>
#include <iostream>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <cufft_test_kernels.h>


int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    int Nx=1000;
    typedef cufft_wrap_C2C<real>::complex_type complex_type;

    init_cuda(-1);

    cufft_wrap_C2C<real> *CUFFT_C = new cufft_wrap_C2C<real>(Nx);
    cufft_wrap_R2C<real> *CUFFT_R = new cufft_wrap_R2C<real>(Nx);
    size_t Nf=CUFFT_R->get_reduced_size();

    complex_type *u_d, *u1_d, *u_hat_d, *u_hat_reduced_d;
    real *uR_d, *uI_d;
    u_d = device_allocate<complex_type>(Nx);
    u1_d = device_allocate<complex_type>(Nx);
    u_hat_d = device_allocate<complex_type>(Nx);
    u_hat_reduced_d = device_allocate<complex_type>(Nf);

    uR_d = device_allocate<real>(Nx);
    uI_d = device_allocate<real>(Nx);


    set_values<real,complex_type>(Nx, u_d);

    CUFFT_C->fft(u_d,u_hat_d);
    CUFFT_C->ifft(u_hat_d, u1_d);
    check_values<real,complex_type>(Nx, u_d, u1_d);
    convert_2R_values<real,complex_type>(Nx, u_d, uR_d, uI_d);

    real *uR=(real*)malloc(sizeof(real)*Nx);
    real *uI=(real*)malloc(sizeof(real)*Nx);
    
    device_2_host_cpy<real>(uR, uR_d, Nx);
    device_2_host_cpy<real>(uI, uI_d, Nx);
    real res_norm_R=0.0;
    real res_norm_I=0.0;
    for(int j=0;j<Nx;j++)
    {
        res_norm_R+=sqrt(uR[j]*uR[j]);
        res_norm_I+=sqrt(uI[j]*uI[j]);
    }

    std::cout << "C2C:" << res_norm_R << " " << res_norm_I << " " << std::endl;
    

    set_values<real,complex_type>(Nx, u_d);
    convert_2R_values<real,complex_type>(Nx, u_d, uR_d, uI_d);
    CUFFT_R->fft(uR_d, u_hat_reduced_d);
    CUFFT_R->ifft(u_hat_reduced_d, uI_d);
    device_2_host_cpy<real>(uR, uR_d, Nx);
    device_2_host_cpy<real>(uI, uI_d, Nx);
    res_norm_R=0.0;
    for(int j=0;j<Nx;j++)
    {   
        real diff=uI[j]/Nx-uR[j];
        res_norm_R+=sqrt(diff*diff);
    }
    std::cout << "R2C2R:" << res_norm_R << " " << std::endl;

    free(uR);
    free(uI);
    cudaFree(u_d);
    cudaFree(u1_d);
    cudaFree(u_hat_d);
    cudaFree(u_hat_reduced_d);
    cudaFree(uR_d);
    cudaFree(uI_d);
    delete CUFFT_C;
    delete CUFFT_R;
    return 0;
}