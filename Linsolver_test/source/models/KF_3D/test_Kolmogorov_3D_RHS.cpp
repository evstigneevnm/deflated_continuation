#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>

#define Blocks_x_ 32
#define Blocks_y_ 16



int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_t;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex_t;
    typedef gpu_vector_operations<real> gpu_vector_operations_t;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> KF_3D_t;
    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;

    if(argc != 4)
    {
        std::cout << argv[0] << " alpha R N:\n 0<alpha<=1, R is the Reynolds number, N = 2^n- discretization in one direction\n";
        return(0);       
    }
    
    real alpha = std::atof(argv[1]);
    real R = std::atof(argv[2]);
    size_t N = std::atoi(argv[3]);
    int one_over_alpha = int(1/alpha);

    init_cuda(-1);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Using alpha = " << alpha << ", Reynolds = " << R << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(3*(Nx*Ny*Mz-1), CUBLAS);
    
    KF_3D_t *KF_3D = new KF_3D_t(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);
    

    vec x0, x1, dx, dy;
    real_vec ux, uy, uz, ua;

    vec_ops_R->init_vector(ux); vec_ops_R->start_use_vector(ux);
    vec_ops_R->init_vector(uy); vec_ops_R->start_use_vector(uy);
    vec_ops_R->init_vector(uz); vec_ops_R->start_use_vector(uz);
    vec_ops_R->init_vector(ua); vec_ops_R->start_use_vector(ua);

    vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
    vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
    vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
    vec_ops->init_vector(dy); vec_ops->start_use_vector(dy);

    KF_3D->randomize_vector(x1, 3);
    KF_3D->randomize_vector(dx, 5);

    KF_3D->B_ABC_exact_solution(x1);
    // KF_3D->write_solution_abs("test_exact_A_ABC.pos", x1);
    // KF_3D->write_solution_vec("test_exact_V_ABC.pos", x1);

    KF_3D->B_ABC_approx_solution(x0);
    // KF_3D->write_solution_abs("test_approx_A_ABC.pos", x1);
    // KF_3D->write_solution_vec("test_approx_V_ABC.pos", x1);


    vec_ops->add_mul(-1.0, x1, x0);
    printf("check B norm = %le\n", vec_ops->norm(x0));

//check RHS
    KF_3D->randomize_vector(x1, 5);
    printf("Divergence rand vec = %le\n", vec_ops->norm(x1) );

    KF_3D->F(x1, R, x0);
    real div_norm = KF_3D->div_norm(x0);
    printf("Divergence norm F = %le\n", double(div_norm) );

//  check Jacobianx
    KF_3D->set_linearization_point(x0, R);
    KF_3D->jacobian_u(dx, dy);
    div_norm = KF_3D->div_norm(dy);
    printf("Divergence norm J_u = %le\n", double(div_norm) );
    // KF_3D->write_solution_abs("testA.pos", dy);
    // KF_3D->write_solution_vec("testV.pos", dy);


    KF_3D->jacobian_alpha(dx);
    div_norm = KF_3D->div_norm(dx);
    printf("Divergence norm J_alpha = %le\n", double(div_norm) );

//  check preconditioner

    KF_3D->preconditioner_jacobian_u(dy);

    div_norm = KF_3D->div_norm(dy);
    printf("Divergence norm inv(M)J_u = %le\n", double(div_norm) );
    // KF_3D->write_solution_abs("testMA.pos", dy);
    // KF_3D->write_solution_vec("testMV.pos", dy);


    vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
    vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
    vec_ops->stop_use_vector(dy); vec_ops->free_vector(dy);
    vec_ops_R->stop_use_vector(ux); vec_ops_R->free_vector(ux);
    vec_ops_R->stop_use_vector(uy); vec_ops_R->free_vector(uy);
    vec_ops_R->stop_use_vector(uz); vec_ops_R->free_vector(uz);
    vec_ops_R->stop_use_vector(ua); vec_ops_R->free_vector(ua);

    delete KF_3D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;
    delete CUFFT_C2R;
    delete CUBLAS;
    
    return 0;
}