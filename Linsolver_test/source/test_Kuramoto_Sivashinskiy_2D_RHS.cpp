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
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_reduced;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef Kuramoto_Sivashinskiy_2D<cufft_type, 
            gpu_vector_operations_real, 
            gpu_vector_operations_complex, 
            gpu_vector_operations_real_reduced,
            Blocks_x_, Blocks_y_> KS_2D;
    typedef typename gpu_vector_operations_real::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex::vector_type complex_vec;

    init_cuda(-1);
    size_t Nx=256;
    size_t Ny=256;
    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny);
    size_t My=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    gpu_vector_operations_real *vec_ops_R = new gpu_vector_operations_real(Nx*Ny, CUBLAS);
    gpu_vector_operations_complex *vec_ops_C = new gpu_vector_operations_complex(Nx*My, CUBLAS);
    gpu_vector_operations_real_reduced *vec_ops_R_im = new gpu_vector_operations_real_reduced(Nx*My-1, CUBLAS);
    //CUDA GRIDS
    dim3 Blocks; dim3 Grids; dim3 Grids_F;
    KS_2D *KS2D = new KS_2D(2.0, 4.0, Nx, Ny, vec_ops_R, vec_ops_C, vec_ops_R_im, CUFFT_C2R);
    KS2D->get_cuda_grid(Grids, Grids_F, Blocks);
    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);

    real_vec u_in, u_out;
    vec_ops_R_im->init_vector(u_in); vec_ops_R_im->start_use_vector(u_in);
    vec_ops_R_im->init_vector(u_out); vec_ops_R_im->start_use_vector(u_out);



    
    delete KS2D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops_R_im;
    delete CUFFT_C2R;
    delete CUBLAS;

    return 0;
}