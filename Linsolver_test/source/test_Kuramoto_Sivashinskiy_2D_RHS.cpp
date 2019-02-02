#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D.h>
#include "macros.h"
#include "file_operations.h"
#include <gpu_vector_operations.h>

#define Blocks_x_ 64
#define Blocks_y_ 16

int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> thrust_complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real;
    typedef gpu_vector_operations<thrust_complex> gpu_vector_operations_complex;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef Kuramoto_Sivashinskiy_2D<gpu_vector_operations_real, gpu_vector_operations_complex, Blocks_x_, Blocks_y_> KS_2D;


    init_cuda(-1);
    size_t Nx=8;
    size_t Ny=8;
    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny);
    size_t My=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    gpu_vector_operations_real *vec_ops_R = new gpu_vector_operations_real(Nx*Ny, CUBLAS);
    gpu_vector_operations_complex *vec_ops_C = new gpu_vector_operations_complex(Nx*My, CUBLAS);

    
    KS_2D *KS2D = new KS_2D(2.0, 4.0, Nx, Ny, vec_ops_R, vec_ops_C, CUFFT_C2R);
    
    dim3 Blocks;
    dim3 Grids;
    dim3 Grids_F;
    KS2D->get_cuda_grid(Grids, Grids_F, Blocks);

    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);

    thrust_complex *u_in_hat = device_allocate<thrust_complex>(Nx*My);
    thrust_complex *u_out_hat = device_allocate<thrust_complex>(Nx*My);
    vec_ops_C->assign_scalar(thrust_complex(1.0,0.0), u_in_hat);
    thrust_complex* u_hat_h=(thrust_complex*)malloc(sizeof(thrust_complex)*Nx*Ny);
    device_2_host_cpy<thrust_complex>(u_hat_h, u_in_hat, Nx*My);
    file_operations::write_matrix<thrust_complex>("u_in_hat.dat",Nx,My,u_hat_h, 3);


    KS2D->F((const thrust_complex*&)u_in_hat, 1.0, u_out_hat);


    
    bool input_finit =vec_ops_C->check_is_valid_number(u_in_hat);
    bool output_finit =vec_ops_C->check_is_valid_number(u_out_hat);
    std::string test_in(true==(bool)input_finit?"Ok":"fail");
    std::string test_out(true==(bool)output_finit?"Ok":"fail");
    std::cout << test_in << " " << test_out << std::endl;
    device_2_host_cpy<thrust_complex>(u_hat_h, u_out_hat, Nx*My);
    file_operations::write_matrix<thrust_complex>("u_out_hat.dat",Nx,My,u_hat_h, 3);

    free(u_hat_h);
    cudaFree(u_in_hat);
    cudaFree(u_out_hat);

    delete KS2D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete CUFFT_C2R;
    delete CUBLAS;

    return 0;
}