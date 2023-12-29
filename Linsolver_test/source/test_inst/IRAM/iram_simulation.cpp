#include  <sstream>
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <common/file_operations.h>
#include <numerical_algos/arnolid_process/arnoldi_process.h>
#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>
#include <stability/IRAM/iram_container.h>
#include <stability/IRAM/shift_bulge_chase.h>
#include <common/gpu_file_operations_functions.h>
#include <test_inst/IRAM/system_operator_test.h>
#include <stability/IRAM/schur_select.h>

template<class T>
void print_matrix(int Nrows, int Ncols, T* A)
{
    T* A_h = new T[Nrows*Nrows];
    device_2_host_cpy(A_h, A, Nrows*Nrows);

    for(int j =0;j<Nrows;j++)
    {
        for(int k=0;k<Ncols;k++)
        {
            std::cout << A_h[j+Nrows*k] << " ";
        }
        std::cout << std::endl;
    }

    delete [] A_h;
}
template<class T>
void print_matrix(int Nl, T* A)
{
    T* A_h = new T[Nl*Nl];
    device_2_host_cpy(A_h, A, Nl*Nl);    
    for(int j =0;j<Nl;j++)
    {
        for(int k=0;k<Nl;k++)
        {
            std::cout << A_h[j+Nl*k] << " ";
        }
        std::cout << std::endl;
    }
    delete [] A_h;
}   

template<class T>
void print_vector(int Ncols, T* v)
{
    T* v_h = new T[Ncols];
    device_2_host_cpy(v_h, v, Ncols);
    for(int k=0;k<Ncols;k++)
    {
        std::cout << v[k] << std::endl;
    }
    delete [] v_h;

}



inline void press_enter_2_cont()
{
    std::string foo;
    std::cout << "Enter to continue..." << std::endl;
    std::getline(std::cin, foo);
}


int main(int argc, char const *argv[])
{
    using real = SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<real>;
    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using mat_files_t = gpu_matrix_file_operations<mat_ops_t>;
    using log_t = utils::log_std;
    using sys_op_t = stability::system_operator_stability<vec_ops_t, mat_ops_t, log_t>;
    using arnoldi_t = numerical_algos::eigen_solvers::arnoldi_process<vec_ops_t, mat_ops_t, sys_op_t, log_t>;
    using lapack_wrap_t = lapack_wrap<real>;
    using container_t = stability::IRAM::iram_container<vec_ops_t,mat_ops_t,log_t>;
    // using bulge_t = stability::IRAM::shift_bulge_chase<vec_ops_t, mat_ops_t, lapack_wrap_t, log_t>;
    using schur_select_t = stability::IRAM::schur_select<vec_ops_t, mat_ops_t, lapack_wrap_t, log_t>;

    if((argc != 2)&&(argc != 3)&&(argc != 4))
    {
        std::cout << "Usage: " << argv[0] << " matrix_file (initial_krylov_file) (pause[y/n])" << std::endl;  
        std::cout << "( ... ) <- optional" << std::endl;  
        return 0; 
    }


    std::string A_file_name(argv[1]);
    std::string v_file_name("none");
    if(argc == 3)
    {
        v_file_name = std::string(argv[2]);    
    }
    
    char pause = 'n';
    if(argc == 4)
    {
        pause = argv[3][0];
    }
    auto NN = file_operations::read_matrix_size(A_file_name);
    size_t N = NN.first;

    if(init_cuda(-1) == 0)
    {
        return 0;
    }

    unsigned int m = 20;
    unsigned int k0 = 6;
    k0 = k0<N?k0:N;
    m = m<N?m:N;


    real ritz_norm = 1;
    int max_iterations = 100;
    real ritz_eps = 1.0e-8;
    int iters = 0;
    size_t k = 0;
    bool debug = false;

    cublas_wrap CUBLAS(true);
    lapack_wrap_t blas(m);
    log_t log;
    vec_ops_t vec_ops_N(N, &CUBLAS);
    vec_ops_t vec_ops_m(m, &CUBLAS);
    mat_ops_t mat_ops_A(N, N, &CUBLAS);
    mat_ops_t mat_ops_N(N, m, &CUBLAS);
    mat_ops_t mat_ops_m(m, m, &CUBLAS);
    mat_files_t mat_f_m(&mat_ops_m);
    mat_files_t mat_f_N(&mat_ops_N);
    mat_files_t mat_f_A(&mat_ops_A);
    vec_file_ops_t vec_file_ops(&vec_ops_N);

    T_mat A;
    T_vec v;
    mat_ops_A.init_matrix(A); mat_ops_A.start_use_matrix(A);
    vec_ops_N.init_vector(v); vec_ops_N.start_use_vector(v);
    mat_f_A.read_matrix(A_file_name, A );
    if(v_file_name != "none")
    {
        vec_file_ops.read_vector(v_file_name, v);
    }


    sys_op_t sys_op(&vec_ops_N, &mat_ops_A, &log);
    arnoldi_t arnoldi(&vec_ops_N, &vec_ops_m, &mat_ops_N, &mat_ops_m, &sys_op, &log);

    container_t container(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &log, ritz_eps, k0, debug);
    // bulge_t bulge(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &log, &blas);
    schur_select_t schur_select(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &log, &blas);

    // bulge.set_number_of_desired_eigenvalues(k0);
    schur_select.set_number_of_desired_eigenvalues(k0);
    container.force_gpu();
    container.init_f();
    container.to_cpu();
    container.to_gpu();
    // bulge.set_target("LM");
    schur_select.set_target("LM");


    sys_op.set_matrix_ptr(A);

    if(v_file_name != "none")
    {
        container.set_f(v);
    }


    real ritz_norm_prev = ritz_norm;

    while( (ritz_norm>ritz_eps)&&(iters<max_iterations) )
    {
        container.reset_ritz();
        // auto ritz_value = arnoldi.execute_arnoldi(k, container.ref_V(), container.ref_H()/*, container.ref_f() */);
        auto ritz_value = arnoldi.execute_arnoldi_schur(k, container.ref_V(), container.ref_H(), container.ref_f() );
        
        
        std::stringstream ss;
        if(debug)
        {
            ss << "V_" << iters << ".dat";
            mat_f_N.write_matrix(ss.str(), container.ref_V() );
            ss.str("");
            ss.clear();
            ss << "H0_" << iters << ".dat";
            mat_f_m.write_matrix(ss.str(), container.ref_H() );
        }

        // bulge.execute(container);
        schur_select.execute(container, ritz_value);

        container.to_gpu();
        if(debug)
        {
            ss.str("");
            ss.clear();
            ss << "H_" << iters << ".dat";
            mat_f_m.write_matrix(ss.str(), container.ref_H() );
        }
        ritz_norm = container.ritz_norm();
        k = container.K;
        if(iters == 0)
            log.info_f("iteration = %i, k = %i, Ritz norm = %le", iters, k, ritz_norm);
        else
            log.info_f("iteration = %i, k = %i, Ritz norm = %le, reduction rate = %.02lf", iters, k, ritz_norm, ritz_norm/ritz_norm_prev);

        ritz_norm_prev = ritz_norm;
        iters++;

        if(pause == 'y')
        {
            press_enter_2_cont();
        }

    }

    
    if(debug)
    {
        mat_f_N.write_matrix("V1.dat", container.ref_V() );
        mat_f_m.write_matrix("H1.dat", container.ref_H() );
        vec_ops_N.debug_view(container.ref_f(), "f1.dat");
        vec_ops_N.stop_use_vector(v); vec_ops_N.free_vector(v);
        mat_ops_A.stop_use_matrix(A); mat_ops_A.free_matrix(A);
    }


    return 0;
}