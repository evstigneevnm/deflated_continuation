#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D.h>
#include "macros.h"
#include "gpu_file_operations.h"
#include "gpu_vector_operations.h"

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
    typedef nonlinear_operators::Kuramoto_Sivashinskiy_2D<cufft_type, 
            gpu_vector_operations_real, 
            gpu_vector_operations_complex, 
            gpu_vector_operations_real_reduced,
            Blocks_x_, Blocks_y_> KS_2D;
    typedef typename gpu_vector_operations_real::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex::vector_type complex_vec;
    typedef typename gpu_vector_operations_real_reduced::vector_type real_im_vec;

    init_cuda(-1);
    size_t Nx=32;
    size_t Ny=32;
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
    real lambda_0 = 8.5;

    real_im_vec u_in, u_out;
    vec_ops_R_im->init_vector(u_in); vec_ops_R_im->start_use_vector(u_in);
    vec_ops_R_im->init_vector(u_out); vec_ops_R_im->start_use_vector(u_out);
    vec_ops_R_im->assign_scalar(1.0, u_in);

    real_vec u_out_ph;
    vec_ops_R->init_vector(u_out_ph); vec_ops_R->start_use_vector(u_out_ph);

    complex_vec uC_in, uC_out;
    vec_ops_C->init_vector(uC_in); vec_ops_C->start_use_vector(uC_in);
    vec_ops_C->init_vector(uC_out); vec_ops_C->start_use_vector(uC_out);
    vec_ops_C->assign_scalar(complex(0.0,1.0), uC_in);

    KS2D->F(u_in, lambda_0, u_out);
    KS2D->F(uC_in, lambda_0, uC_out);
    KS2D->set_linearization_point(u_in, lambda_0);
    
    KS2D->jacobian_u(u_in, u_out);
    KS2D->jacobian_u(uC_in, uC_out);

    KS2D->preconditioner_jacobian_u(u_out);
    KS2D->preconditioner_jacobian_u(uC_out);
    
    KS2D->physical_solution(u_out, u_out_ph);

    // gpu_file_operations::write_matrix<complex>("uC_out_M.dat",  My, Nx, uC_out, 3);
    gpu_file_operations::write_vector<complex>("uC_out.dat",  My*Nx, uC_out, 3);
    gpu_file_operations::write_vector<real>("u_im_out.dat", Nx*My-1, u_out, 3);
    gpu_file_operations::write_matrix<real>("u_out.dat", Nx, Ny, u_out_ph, 3);

    vec_ops_R->stop_use_vector(u_out_ph); vec_ops_R->free_vector(u_out_ph);
    vec_ops_R_im->stop_use_vector(u_in); vec_ops_R_im->free_vector(u_in);
    vec_ops_R_im->stop_use_vector(u_out); vec_ops_R_im->free_vector(u_out);
    vec_ops_C->stop_use_vector(uC_in); vec_ops_C->free_vector(uC_in);
    vec_ops_C->stop_use_vector(uC_out); vec_ops_C->free_vector(uC_out);
    
    delete KS2D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops_R_im;
    delete CUFFT_C2R;
    delete CUBLAS;

    return 0;
}