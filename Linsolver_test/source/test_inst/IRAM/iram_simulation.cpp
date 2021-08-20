#include <common/gpu_vector_operations.h>
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



int main(int argc, char const *argv[])
{
    using real = SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<real>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using mat_files_t = gpu_matrix_file_operations<mat_ops_t>;
    using log_t = utils::log_std;
    using sys_op_t = stability::system_operator_stability<vec_ops_t, mat_ops_t, log_t>;
    using arnoldi_t = numerical_algos::eigen_solvers::arnoldi_process<vec_ops_t, mat_ops_t, sys_op_t, log_t>;
    using lapack_wrap_t = lapack_wrap<real>;
    using container_t = stability::IRAM::iram_container<vec_ops_t,mat_ops_t,log_t>;
    using bulge_t = stability::IRAM::shift_bulge_chase<vec_ops_t, mat_ops_t, lapack_wrap_t, log_t>;

    if(argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " matrix_file" << std::endl;  
        return 0; 
    }
    std::string A_file_name(argv[1]);
    size_t N = file_operations::read_matrix_size(A_file_name);
    if(init_cuda(4) == 0)
    {
        return 0;
    }

    unsigned int m = 30;
    unsigned int k0 = 10;


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
    
    T_mat A;
    mat_ops_A.init_matrix(A); mat_ops_A.start_use_matrix(A);
    mat_f_A.read_matrix(A_file_name, A );

    sys_op_t sys_op(&vec_ops_N, &mat_ops_A, &log);
    arnoldi_t arnoldi(&vec_ops_N, &vec_ops_m, &mat_ops_N, &mat_ops_m, &sys_op, &log);

    container_t container(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &log);
    bulge_t bulge(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &log, &blas);
    bulge.set_number_of_desired_eigenvalues(k0);

    container.force_gpu();
    container.set_f();
    container.to_cpu();
    container.to_gpu();
    bulge.set_target("LM");


    real ritz_norm = 1;
    int max_iterations = 100;
    real ritz_eps = 1.0e-8;
    int iters = 0;
    size_t k = 0;
    sys_op.set_matrix_ptr(A);

    while( (ritz_norm>ritz_eps)&&(iters<max_iterations) )
    {
        container.reset_ritz();
        arnoldi.execute_arnoldi(k, container.ref_V(), container.ref_H(), container.ref_f() );
        
        // break;
        bulge.execute(container);
        container.to_gpu();
        ritz_norm = container.ritz_norm();
        k = container.K;
        log.info_f("iteration = %i, k = %i, Ritz norm = %le", iters, k, ritz_norm);
        iters++;
        
    }

    

    mat_f_N.write_matrix("V1.dat", container.ref_V() );
    mat_f_m.write_matrix("H1.dat", container.ref_H() );
    vec_ops_N.debug_view(container.ref_f(), "f1.dat");
    mat_ops_A.stop_use_matrix(A); mat_ops_A.free_matrix(A);



    return 0;
}