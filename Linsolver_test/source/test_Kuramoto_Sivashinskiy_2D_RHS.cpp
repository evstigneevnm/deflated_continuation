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
    typedef gpu_vector_operations<real,real> gpu_vector_operations_real;
    typedef gpu_vector_operations<thrust_complex,real> gpu_vector_operations_complex;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef Kuramoto_Sivashinskiy_2D<gpu_vector_operations_real, gpu_vector_operations_complex, Blocks_x_, Blocks_y_> KS_2D;


    init_cuda(-1);
    size_t Nx=256;
    size_t Ny=256;
    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny);
    size_t My=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    gpu_vector_operations_real *vec_ops_R = new gpu_vector_operations_real(Nx*My, CUBLAS);
    gpu_vector_operations_complex *vec_ops_C = new gpu_vector_operations_complex(Nx*My, CUBLAS);

    
    KS_2D *KS2D = new KS_2D(Nx, Ny, vec_ops_R, vec_ops_C, CUFFT_C2R);
    
    dim3 Blocks;
    dim3 Grids;
    dim3 Grids_F;
    KS2D->get_cuda_grid(Grids, Grids_F, Blocks);
    KS2D->tests();

    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);


    delete KS2D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete CUFFT_C2R;
    delete CUBLAS;

    return 0;
}