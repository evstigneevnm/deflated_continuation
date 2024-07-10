#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <scfd/utils/cuda_ownutils.h>

#define Blocks_x_ 32
#define Blocks_y_ 16




template<class T>
std::pair<std::string, bool> check_test_to_eps(const T val)
{

    if((std::abs(val)>std::sqrt(std::numeric_limits<T>::epsilon()) )||( !std::isfinite(val) ))
    {
        return {"\x1B[31mFAIL\033[0m", false};
    }
    else
    {
        return {"\x1B[32mPASS\033[0m", true};
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
    typedef nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> KF_3D_t;
    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;

    using KF_3D_full_t = nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_, false>;

    if(argc != 4)
    {
        std::cout << argv[0] << " alpha R N:\n 0<alpha<=1, R is the Reynolds number, N = 2^n- discretization in one direction\n";
        return(0);       
    }

    real alpha = std::atof(argv[1]);
    real R = std::atof(argv[2]);
    size_t N = std::atoi(argv[3]);
    int one_over_alpha = int(1/alpha);

    size_t mem_about = static_cast<size_t>(std::ceil(N*N*N*one_over_alpha*82*8/1.0E6));
    scfd::utils::init_cuda_persistent(mem_about);
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
    
    vec x0, x1, x2, x3, dx, dy;
    real_vec ux, uy, uz, ua;


    vec_ops_R->init_vector(ux); vec_ops_R->start_use_vector(ux);
    vec_ops_R->init_vector(uy); vec_ops_R->start_use_vector(uy);
    vec_ops_R->init_vector(uz); vec_ops_R->start_use_vector(uz);
    vec_ops_R->init_vector(ua); vec_ops_R->start_use_vector(ua);    
    
    bool flag_ok = true;
    std::cout << "testing reduced set..." << std::endl;
    {
        gpu_vector_operations_t* vec_ops = new gpu_vector_operations_t(3*(Nx*Ny*Mz-1), CUBLAS);
        KF_3D_t KF_3D(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);


        vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
        vec_ops->init_vector(dy); vec_ops->start_use_vector(dy);

        KF_3D.exact_solution(R, x1);
        if(N < 64)
        {
            KF_3D.write_solution_vec("exact_solution_vec_reduced.pos", x1);
            KF_3D.write_solution_abs("exact_solution_abs_reduced.pos", x1);
        }
        real div_norm_exact = KF_3D.div_norm(x1);
        printf("exact solution norm = %le\n", vec_ops->norm_l2(x1));
        printf("Divergence norm exact solution = %le\n", double(div_norm_exact) );
        if(div_norm_exact > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }


        KF_3D.randomize_vector(x1, 3);
        KF_3D.randomize_vector(dx, 5);
        KF_3D.B_ABC_exact_solution(x1);
        // KF_3D.write_solution_abs("test_exact_A_ABC.pos", x1);
        // KF_3D.write_solution_vec("test_exact_V_ABC.pos", x1);
        KF_3D.B_ABC_approx_solution(x0);
        // KF_3D.write_solution_abs("test_approx_A_ABC.pos", x1);
        // KF_3D.write_solution_vec("test_approx_V_ABC.pos", x1);
        vec_ops->add_mul(-1.0, x1, x0);
        printf("check B norm = %le\n", vec_ops->norm(x0));
    //check RHS
        KF_3D.randomize_vector(x1, 5);
        printf("rand vec = %le\n", vec_ops->norm(x1) );
        auto div_norm_x1 = KF_3D.div_norm(x1);
        printf("Divergence rand vec = %le\n",  div_norm_x1);
        if(div_norm_x1 > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }

        KF_3D.F(x1, R, x0);
        printf("F(x) norm = %le\n", vec_ops->norm(x0));
        real div_norm = KF_3D.div_norm(x0);
        printf("Divergence norm F = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
    //  check Jacobianx
        KF_3D.set_linearization_point(x0, R);
        KF_3D.jacobian_u(dx, dy);
        div_norm = KF_3D.div_norm(dy);
        printf("Divergence norm J_u = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
        // KF_3D.write_solution_abs("testA.pos", dy);
        // KF_3D.write_solution_vec("testV.pos", dy);
        KF_3D.jacobian_alpha(dx);
        div_norm = KF_3D.div_norm(dx);
        printf("Divergence norm J_alpha = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
    //  check preconditioner
        KF_3D.preconditioner_jacobian_u(dy);
        div_norm = KF_3D.div_norm(dy);
        printf("Divergence norm inv(M)J_u = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
        // KF_3D.write_solution_abs("testMA.pos", dy);
        // KF_3D.write_solution_vec("testMV.pos", dy);
        vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
        vec_ops->stop_use_vector(dy); vec_ops->free_vector(dy);
    }

    std::cout << "testing full set..." << std::endl;
    {
        auto vec_ops = std::make_shared<gpu_vector_operations_t>(6*(Nx*Ny*Mz-1), CUBLAS);
        KF_3D_full_t KF_3D(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops.get(), CUFFT_C2R);

        vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(x2); vec_ops->start_use_vector(x2);
        vec_ops->init_vector(x3); vec_ops->start_use_vector(x3);
        vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
        vec_ops->init_vector(dy); vec_ops->start_use_vector(dy);

        KF_3D.exact_solution(R, x1);
        if(N < 64)
        {
            KF_3D.write_solution_vec("exact_solution_vec_full.pos", x1);
            KF_3D.write_solution_abs("exact_solution_abs_full.pos", x1);
        }
        real div_norm_exact = KF_3D.div_norm(x1);
        printf("exact solution norm = %le\n", vec_ops->norm_l2(x1));
        printf("Divergence norm exact solution = %le\n", double(div_norm_exact) );
        if(div_norm_exact > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }

        KF_3D.randomize_vector(x1, 3);
        KF_3D.randomize_vector(dx, 5);
        KF_3D.B_ABC_exact_solution(x1);
        // KF_3D.write_solution_abs("test_exact_A_ABC.pos", x1);
        // KF_3D.write_solution_vec("test_exact_V_ABC.pos", x1);
        KF_3D.B_ABC_approx_solution(x0);
        // KF_3D.write_solution_abs("test_approx_A_ABC.pos", x1);
        // KF_3D.write_solution_vec("test_approx_V_ABC.pos", x1);
        vec_ops->add_mul(-1.0, x1, x0);
        printf("check B norm = %le\n", vec_ops->norm(x0));
    //check RHS
        KF_3D.randomize_vector(x1, 5);
        real div_norm = KF_3D.div_norm(x1);
        printf("rand vec = %le\n", vec_ops->norm(x1) );
        printf("Divergence rand vec = %le\n", div_norm);
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }
        KF_3D.F(x1, R, x0);
        printf("F(x) norm = %le\n", vec_ops->norm(x0));
        div_norm = KF_3D.div_norm(x0);
        printf("Divergence norm F = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
    //  check Jacobianx
        KF_3D.set_linearization_point(x0, R);
        KF_3D.jacobian_u(dx, dy);
        div_norm = KF_3D.div_norm(dy);
        printf("Divergence norm J_u = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
        // KF_3D.write_solution_abs("testA.pos", dy);
        // KF_3D.write_solution_vec("testV.pos", dy);
        KF_3D.jacobian_alpha(dx);
        div_norm = KF_3D.div_norm(dx);
        printf("Divergence norm J_alpha = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
    //  check preconditioner
        KF_3D.preconditioner_jacobian_u(dy);
        div_norm = KF_3D.div_norm(dy);
        printf("Divergence norm inv(M)J_u = %le\n", double(div_norm) );
        if(div_norm > std::sqrt(std::numeric_limits<real>::epsilon() ) )
        {
            std::cout << "FAILED" << std::endl;
            flag_ok = false;
        }        
        // KF_3D.write_solution_abs("testMA.pos", dy);
        // KF_3D.write_solution_vec("testMV.pos", dy);

    //  check solution Hermitian conjugate
        KF_3D.randomize_vector(x0, 5);
        vec_ops->assign(x0, x1);
        KF_3D.hermitian_symmetry(x1, x1);
        vec_ops->add_mul( -1.0, x0, x1);
        div_norm = vec_ops->norm(x1);
        printf("Hermitian symmetry test  = %le, %s\n", double(div_norm), check_test_to_eps(div_norm).first.c_str()  );
        if(!check_test_to_eps(div_norm).second)
        {
            flag_ok = false;
        }
    //  check translation with there and back again
        // KF_3D.exact_solution(R, x0);
        KF_3D.randomize_vector(x0, 4);
        real varphi_x = 3.14;
        real varphi_y = 0;
        real varphi_z = 2.134;
        KF_3D.translate_solution(x0, varphi_x, varphi_y, varphi_z, x1);
        // KF_3D.write_solution_vec("exact_solution_vec_translate.pos", x1);
        // KF_3D.write_solution_abs("exact_solution_abs_translate.pos", x1);
        KF_3D.translate_solution(x1, -varphi_x, -varphi_y, -varphi_z, x2);
        vec_ops->add_mul( -1.0, x0, x2);
        div_norm = vec_ops->norm_l2(x2);
        printf("translation symmetry test  = %le, %s\n", double( div_norm ), check_test_to_eps(div_norm).first.c_str()  );
        if(!check_test_to_eps(div_norm).second)
        {
            flag_ok = false;
        }
        KF_3D.translate_solution(x0, 2*M_PI, 0, 2*M_PI, x2);
        vec_ops->add_mul( -1.0, x0, x2);
        div_norm = vec_ops->norm_l2(x2);
        printf("translation return map test  = %le, %s\n", double( div_norm ), check_test_to_eps(div_norm).first.c_str()  );
        if(!check_test_to_eps(div_norm).second)
        {
            flag_ok = false;
        }

    //  check translation fix
        // KF_3D.exact_solution(R, x0);
        KF_3D.randomize_vector(x0, 4);
        // KF_3D.translate_solution(x0, varphi_x, varphi_y, varphi_z, x1);
        KF_3D.translation_fix(x0, x1);
        KF_3D.translate_solution(x1, -varphi_x/10.0, varphi_y/15.0, varphi_z/17.0, x2);
        // KF_3D.translation_fix(x1, x0);
        KF_3D.translation_fix(x2, x3);
        vec_ops->add_mul( -1.0, x1, x3);
        div_norm = vec_ops->norm_l2(x3);
        printf("translation symmetry fix test  = %le, %s\n", double( div_norm ), check_test_to_eps(div_norm).first.c_str()  );
        if(!check_test_to_eps(div_norm).second)
        {
            flag_ok = false;
        }

        vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(x2); vec_ops->free_vector(x2);
        vec_ops->stop_use_vector(x3); vec_ops->free_vector(x3);
        vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
        vec_ops->stop_use_vector(dy); vec_ops->free_vector(dy);
    }


    vec_ops_R->stop_use_vector(ux); vec_ops_R->free_vector(ux);
    vec_ops_R->stop_use_vector(uy); vec_ops_R->free_vector(uy);
    vec_ops_R->stop_use_vector(uz); vec_ops_R->free_vector(uz);
    vec_ops_R->stop_use_vector(ua); vec_ops_R->free_vector(ua);





    delete vec_ops_R;
    delete vec_ops_C;
    delete CUFFT_C2R;
    delete CUBLAS;
    if(flag_ok)
    {
        std::cout << "PASSED" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "FAILED" << std::endl;
        return 1;        
    }
}