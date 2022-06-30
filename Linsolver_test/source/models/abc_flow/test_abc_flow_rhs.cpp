#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/abc_flow/abc_flow.h>
#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <string>
#include <type_traits>

#define Blocks_x_ 32
#define Blocks_y_ 16


template<class T>
std::string check_test_to_eps(const T val)
{
    T eps_ = 0;
    if(std::is_same<T, double>::value)
    {
        eps_ = 1.0e-11;
    }
    else if(std::is_same<T, float>::value)
    {
        eps_ = 1.0e-7;
    }
    else
    {
        std::cout << "WARNING: eps is set to 0 for unsupported floating point type" << std::endl;
    }
    if(std::abs(val)>eps_)
    {
        return "FAIL";
    }
    else
    {
        return "PASS";
    }
}


int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_t;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex_t;
    typedef gpu_vector_operations<real> gpu_vector_operations_t;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef nonlinear_operators::abc_flow<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> abc_flow_t;
    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;

    if(argc != 3)
    {
        std::cout << argv[0] << " R N:\n R is the Reynolds number, N = 2^n- discretization in one direction\n";
        return(0);       
    }
    
    real R = std::atof(argv[1]);
    size_t N = std::atoi(argv[2]);

    init_cuda(-1);
    size_t Nx = N;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Reynolds = " << R << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(6*(Nx*Ny*Mz-1), CUBLAS);
    
    abc_flow_t *abc_flow = new abc_flow_t(Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);
    

    vec x0, x1, x2, x3, dx, dy;
    real_vec ux, uy, uz, ua;

    vec_ops_R->init_vector(ux); vec_ops_R->start_use_vector(ux);
    vec_ops_R->init_vector(uy); vec_ops_R->start_use_vector(uy);
    vec_ops_R->init_vector(uz); vec_ops_R->start_use_vector(uz);
    vec_ops_R->init_vector(ua); vec_ops_R->start_use_vector(ua);

    vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
    vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
    vec_ops->init_vector(x2); vec_ops->start_use_vector(x2);
    vec_ops->init_vector(x3); vec_ops->start_use_vector(x3);    
    vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
    vec_ops->init_vector(dy); vec_ops->start_use_vector(dy);

    abc_flow->exact_solution(R, x1);
    abc_flow->write_solution_vec("exact_solution_vec.pos", x1);
    abc_flow->write_solution_abs("exact_solution_abs.pos", x1);
    real div_norm_exact = abc_flow->div_norm(x1);
    printf("exact solution norm = %le\n", vec_ops->norm_l2(x1));
    printf("Divergence norm exact solution = %le, %s\n", double(div_norm_exact), check_test_to_eps(div_norm_exact).c_str() );


    abc_flow->randomize_vector(x1, 3);
    abc_flow->randomize_vector(dx, 5);




    // abc_flow->B_ABC_exact_solution(x1);
    // // abc_flow->write_solution_abs("test_exact_A_ABC.pos", x1);
    // // abc_flow->write_solution_vec("test_exact_V_ABC.pos", x1);

    // abc_flow->B_ABC_approx_solution(x0);
    // // abc_flow->write_solution_abs("test_approx_A_ABC.pos", x1);
    // // abc_flow->write_solution_vec("test_approx_V_ABC.pos", x1);


    // vec_ops->add_mul(-1.0, x1, x0);
    // printf("check B norm = %le\n", vec_ops->norm(x0));

//check RHS
    abc_flow->randomize_vector(x1, 5);
    printf("rand vec = %le\n", vec_ops->norm(x1) );
    printf("Divergence rand vec = %le, %s\n", abc_flow->div_norm(x1),  check_test_to_eps(abc_flow->div_norm(x1)).c_str() );

    abc_flow->F(x1, R, x0);
    printf("F(x) norm = %le\n", vec_ops->norm(x0));
    real div_norm = abc_flow->div_norm(x0);
    printf("Divergence norm F = %le, %s\n", double(div_norm), check_test_to_eps(div_norm).c_str() );

//  check Jacobianx
    abc_flow->set_linearization_point(x0, R);
    abc_flow->jacobian_u(dx, dy);
    div_norm = abc_flow->div_norm(dy);
    printf("Divergence norm J_u = %le, %s\n", double(div_norm), check_test_to_eps(div_norm).c_str() );
    // abc_flow->write_solution_abs("testA.pos", dy);
    // abc_flow->write_solution_vec("testV.pos", dy);


    abc_flow->jacobian_alpha(dx);
    div_norm = abc_flow->div_norm(dx);
    printf("Divergence norm J_alpha = %le, %s\n", double(div_norm), check_test_to_eps(div_norm).c_str()  );

//  check preconditioner

    abc_flow->preconditioner_jacobian_u(dy);

    div_norm = abc_flow->div_norm(dy);
    printf("Divergence norm inv(M)J_u = %le, %s\n", double(div_norm), check_test_to_eps(div_norm).c_str()  );
    // abc_flow->write_solution_abs("testMA.pos", dy);
    // abc_flow->write_solution_vec("testMV.pos", dy);


//  check solution Hermitian conjugate
    abc_flow->randomize_vector(x0, 5);
    vec_ops->assign(x0, x1);
    abc_flow->hermitian_symmetry(x1, x1);
    vec_ops->add_mul( -1.0, x0, x1);
    printf("Hermitian symmetry test  = %le, %s\n", double( vec_ops->norm(x1) ), check_test_to_eps(vec_ops->norm(x1)).c_str()  );

//  check translation with there and back again
    abc_flow->exact_solution(R, x0);
    real varphi_x = 3.14;
    real varphi_y = -0.425;
    real varphi_z = 2.134;
    abc_flow->translate_solution(x0, varphi_x, varphi_y, varphi_z, x1);
    abc_flow->write_solution_vec("exact_solution_vec_translate.pos", x1);
    abc_flow->write_solution_abs("exact_solution_abs_translate.pos", x1);
    abc_flow->translate_solution(x1, -varphi_x, -varphi_y, -varphi_z, x2);
    vec_ops->add_mul( -1.0, x0, x2);
    printf("translation symmetry test  = %le, %s\n", double( vec_ops->norm_l2(x2) ), check_test_to_eps(vec_ops->norm_l2(x2)).c_str()  );

//  check translation fix
    abc_flow->exact_solution(R, x0);
    abc_flow->translate_solution(x0, varphi_x, varphi_y, varphi_z, x1);
    abc_flow->translate_solution(x0, -varphi_x/10.0, varphi_y/12.0, -varphi_z/8.0, x2);
    abc_flow->translation_fix(x1, x0);
    abc_flow->translation_fix(x2, x3);
    vec_ops->add_mul( -1.0, x0, x3);
    printf("translation symmetry fix test  = %le, %s\n", double( vec_ops->norm_l2(x3) ), check_test_to_eps(vec_ops->norm_l2(x3)).c_str()  );



    vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
    vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    vec_ops->stop_use_vector(x2); vec_ops->free_vector(x2);
    vec_ops->stop_use_vector(x3); vec_ops->free_vector(x3);    
    vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
    vec_ops->stop_use_vector(dy); vec_ops->free_vector(dy);
    vec_ops_R->stop_use_vector(ux); vec_ops_R->free_vector(ux);
    vec_ops_R->stop_use_vector(uy); vec_ops_R->free_vector(uy);
    vec_ops_R->stop_use_vector(uz); vec_ops_R->free_vector(uz);
    vec_ops_R->stop_use_vector(ua); vec_ops_R->free_vector(ua);

    delete abc_flow;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;
    delete CUFFT_C2R;
    delete CUBLAS;
    
    return 0;
}