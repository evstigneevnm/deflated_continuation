#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D.h>
#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>

#define Blocks_x_ 64
#define Blocks_y_ 16



int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_reduced;
    using gpu_file_operations_reduced_t = gpu_file_operations<gpu_vector_operations_real_reduced>;

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
    gpu_file_operations_reduced_t gpu_file_operations_reduced(vec_ops_R_im);
    //CUDA GRIDS
    dim3 Blocks; dim3 Grids; dim3 Grids_F;
    KS_2D *KS2D = new KS_2D(2.0, 4.0, Nx, Ny, vec_ops_R, vec_ops_C, vec_ops_R_im, CUFFT_C2R);
    KS2D->get_cuda_grid(Grids, Grids_F, Blocks);
    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);
    real lambda_0 = 7.6;
    
    real_im_vec u_in, u_out;
    vec_ops_R_im->init_vector(u_in); vec_ops_R_im->start_use_vector(u_in);
    vec_ops_R_im->init_vector(u_out); vec_ops_R_im->start_use_vector(u_out);
    //vec_ops_R_im->assign_scalar(1.0, u_in);
    KS2D->randomize_vector(u_in);

    real_vec u_in_ph, u_out_ph;
    vec_ops_R->init_vectors(u_in_ph, u_out_ph); vec_ops_R->start_use_vectors(u_in_ph, u_out_ph);

    complex_vec vec_C_on_host, vec_C_on_device;

    size_t NNN = vec_ops_C->get_vector_size();
    vec_C_on_host = (complex_vec) malloc( NNN*sizeof(complex) );
    vec_ops_C->init_vector(vec_C_on_device); vec_ops_C->start_use_vector(vec_C_on_device);
    for(int j=0;j<NNN;j++)
    {
        vec_C_on_host[j] = complex(0,-j);
    }

    vec_ops_C->set(vec_C_on_host, vec_C_on_device);

    for(int j=NNN-10;j<NNN;j++)
    {
        complex_vec val_vec_l = vec_ops_C->view(vec_C_on_device);

        printf("%le ", double(vec_C_on_host[j].imag() - val_vec_l[j].imag()) );
    }

    vec_ops_C->stop_use_vector(vec_C_on_device); vec_ops_C->free_vector(vec_C_on_device);

    free(vec_C_on_host);


    complex_vec uC_in, uC_out;
    vec_ops_C->init_vector(uC_in); vec_ops_C->start_use_vector(uC_in);
    vec_ops_C->init_vector(uC_out); vec_ops_C->start_use_vector(uC_out);
    vec_ops_C->assign_scalar(complex(0.0,1.0*Nx*Ny), uC_in);

    vec_ops_R_im->assign_scalar(1.0, u_in);
    KS2D->F(u_in, lambda_0, u_out);
    gpu_file_operations_reduced.write_vector("u_im_ones.dat", u_out, 3);

    auto exp_func = [](int j, int k, int shift, real muu)
    {
        return exp( -muu*((j-shift)*(j-shift)+(k-shift)*(k-shift)) );
    };

    std::vector<real> uR_in_h(Nx*Ny,0);
    for(int j=0;j<Nx;j++)
    {
        for(int k=0;k<Ny;k++)
        {
            real muu = 0.05;
            uR_in_h[I2(j,k,Ny)] = exp_func(j,k,10,muu)-exp_func(j,k,20,muu);
        }
    }
    host_2_device_cpy(u_in_ph, uR_in_h.data(), Nx*Ny);
    // vec_ops_R->assign_scalar(1, u_in_ph);
    KS2D->fft_test(u_in_ph, uC_in);
    std::vector<complex> uC_out_h(Nx*My,0);
    std::cout << std::endl;
    device_2_host_cpy(uC_out_h.data(), uC_in, Nx*My);
    for(int j = 0;j<Nx;j++)
    {
        for(int k = 0;k<My;k++)
        {
            std::cout << uC_out_h[I2(j,k,My)] << " ";
        }
        std::cout << std::endl;
    }

    KS2D->ifft_test(uC_in, u_out_ph);
    vec_ops_R->add_mul(-1, u_in_ph, u_out_ph);
    std::cout << "FFT test = " << vec_ops_R->norm_l2(u_out_ph) << std::endl;
    // device_2_host_cpy(uC_out_h.data(), uC_out, Nx*My);
    // for(int j = 0;j<Nx;j++)
    // {
    //     for(int k = 0;k<My;k++)
    //     {
    //         std::cout << uC_out_h[I2(j,k,My)] << " ";
    //     }
    //     std::cout << std::endl;
    // }    
    std::cout << std::endl;
    device_2_host_cpy(uC_out_h.data(), uC_in, Nx*My);
    for(int j = 0;j<Nx;j++)
    {
        for(int k = 0;k<My;k++)
        {
            std::cout << uC_out_h[I2(j,k,My)] << " ";
        }
        std::cout << std::endl;
    }
    
    KS2D->F(uC_in, lambda_0, uC_out);
    


    KS2D->randomize_vector(u_in);
    KS2D->F(u_in, lambda_0, u_out);
    

    KS2D->set_linearization_point(u_in, lambda_0);
    
    KS2D->jacobian_u(u_in, u_out);
    KS2D->jacobian_u(uC_in, uC_out);

    KS2D->preconditioner_jacobian_u(u_out);
    KS2D->preconditioner_jacobian_u(uC_out);
    
    KS2D->physical_solution(u_out, u_out_ph);




    // gpu_file_operations::write_matrix<complex>("uC_out_M.dat",  My, Nx, uC_out, 3);
    // gpu_file_operations::write_vector<complex>("uC_out.dat",  My*Nx, uC_out, 3);
    // gpu_file_operations::write_vector<real>("u_im_out.dat", Nx*My-1, u_out, 3);
    // gpu_file_operations::write_matrix<real>("u_out.dat", Nx, Ny, u_out_ph, 3);

    vec_ops_R->stop_use_vectors(u_in_ph, u_out_ph); vec_ops_R->free_vectors(u_in_ph, u_out_ph);
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