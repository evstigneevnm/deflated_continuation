#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/Taylor_Green/Taylor_Green.h>
#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <scfd/utils/cuda_ownutils.h>

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
    typedef nonlinear_operators::Taylor_Green<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> TG_t;
    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;

    if(argc != 4)
    {
        std::cout << argv[0] << " L R N:\n    L is the size of the domain in all dims, R is the Reynolds number, N = 2^n- discretization in one direction\n";
        return(0);       
    }
    
    real L = std::atof(argv[1]);
    real R = std::atof(argv[2]);
    size_t N = std::atoi(argv[3]);
    
    scfd::utils::init_cuda(-1);
    size_t Nx = N;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Using Reynolds = " << R << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(3*(Nx*Ny*Mz-1), CUBLAS);
    
    TG_t *TG = new TG_t(L, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);
    

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



    TG->exact_solution(R, x1);
    TG->write_solution_vec("exact_solution_vec.pos", x1);
    TG->write_solution_abs("exact_solution_abs.pos", x1);
    real div_norm_exact = TG->div_norm(x1);
    printf("exact solution norm = %le\n", vec_ops->norm_l2(x1));
    printf("Divergence norm exact solution = %le\n", double(div_norm_exact) );


    TG->randomize_vector(x1, 3);
    TG->randomize_vector(dx, 5);




    TG->exact_solution(R, x1);
    // TG->write_solution_abs("test_exact_A_ABC.pos", x1);
    // TG->write_solution_vec("test_exact_V_ABC.pos", x1);

    TG->exact_solution_sin_cos(R, x0);
    // TG->write_solution_abs("test_approx_A_ABC.pos", x1);
    // TG->write_solution_vec("test_approx_V_ABC.pos", x1);


    vec_ops->add_mul(-1.0, x1, x0);
    printf("check B norm = %le\n", vec_ops->norm(x0));

//check RHS
    TG->randomize_vector(x1, 5);
    printf("rand vec = %le\n", vec_ops->norm(x1) );
    printf("Divergence rand vec = %le\n", TG->div_norm(x1) );

    TG->F(x1, R, x0);
    printf("F(x) norm = %le\n", vec_ops->norm(x0));
    real div_norm = TG->div_norm(x0);
    printf("Divergence norm F = %le\n", double(div_norm) );

//  check Jacobianx
    TG->set_linearization_point(x0, R);
    TG->jacobian_u(dx, dy);
    div_norm = TG->div_norm(dy);
    printf("Divergence norm J_u = %le\n", double(div_norm) );
    // TG->write_solution_abs("testA.pos", dy);
    // TG->write_solution_vec("testV.pos", dy);


    TG->jacobian_alpha(dx);
    div_norm = TG->div_norm(dx);
    printf("Divergence norm J_alpha = %le\n", double(div_norm) );

//  check preconditioner

    TG->preconditioner_jacobian_u(dy);

    div_norm = TG->div_norm(dy);
    printf("Divergence norm inv(M)J_u = %le\n", double(div_norm) );
    // TG->write_solution_abs("testMA.pos", dy);
    // TG->write_solution_vec("testMV.pos", dy);


    vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
    vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
    vec_ops->stop_use_vector(dy); vec_ops->free_vector(dy);
    vec_ops_R->stop_use_vector(ux); vec_ops_R->free_vector(ux);
    vec_ops_R->stop_use_vector(uy); vec_ops_R->free_vector(uy);
    vec_ops_R->stop_use_vector(uz); vec_ops_R->free_vector(uz);
    vec_ops_R->stop_use_vector(ua); vec_ops_R->free_vector(ua);

    delete TG;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;
    delete CUFFT_C2R;
    delete CUBLAS;
    
    return 0;
}