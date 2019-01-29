#include <cmath>
#include <iostream>
#include <cstdio>
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
    typedef gpu_vector_operations<real> gpu_vector_operations_real;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef Kuramoto_Sivashinskiy_2D<gpu_vector_operations_real, Blocks_x_, Blocks_y_> KS_2D;


    init_cuda(-1);
    size_t Nx=256;
    size_t Ny=256;
    cufft_type *CUFFT2_R = new cufft_type(Nx, Ny);
    size_t My=CUFFT2_R->get_reduced_size();
    gpu_vector_operations_real vec_ops(Nx*My);
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    
    KS_2D *KS2D = new KS_2D(Nx, Ny, &vec_ops, CUFFT2_R);
    
    dim3 Blocks;
    dim3 Grids;
    dim3 Grids_F;
    KS2D->get_cuda_grid(Grids, Grids_F, Blocks);
    KS2D->tests();

    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);


    delete KS2D;
    delete CUFFT2_R;
    delete CUBLAS;

    return 0;
}