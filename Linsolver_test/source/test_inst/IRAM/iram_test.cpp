#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <common/file_operations.h>
#include <numerical_algos/arnolid_process/arnoldi_process.h>
#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>
#include <common/gpu_file_operations_functions.h>
#include <test_inst/IRAM/system_operator_test.h>
#include <test_inst/IRAM/linear_operator.h>
#include <stability/IRAM/iram_process.hpp>

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
    using lin_op_t = stability::linear_operator<vec_ops_t, mat_ops_t>;
    using arnoldi_t = numerical_algos::eigen_solvers::arnoldi_process<vec_ops_t, mat_ops_t, sys_op_t, log_t>;
    using lapack_wrap_t = lapack_wrap<real>;
    using iram_t = stability::IRAM::iram_process<vec_ops_t, mat_ops_t, lapack_wrap_t, arnoldi_t, sys_op_t, lin_op_t, log_t>;

    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " matrix_file_A matrix_file_P(A)" << std::endl;  
        return 0; 
    }
    std::string A_file_name(argv[1]);
    std::string PA_file_name(argv[2]);
    size_t N = file_operations::read_matrix_size(A_file_name);
    if(init_cuda(4) == 0)
    {
        return 0;
    }

    unsigned int m = 20;
    unsigned int k0 = 8;


    cublas_wrap CUBLAS(true);
    lapack_wrap_t lapack(m);
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
    T_mat PA;
    mat_ops_A.init_matrix(A); mat_ops_A.start_use_matrix(A);
    mat_ops_A.init_matrix(PA); mat_ops_A.start_use_matrix(PA);
    mat_f_A.read_matrix(A_file_name, A );
    mat_f_A.read_matrix(PA_file_name, PA );

    lin_op_t lin_op(&vec_ops_N, &mat_ops_A);
    sys_op_t sys_op(&vec_ops_N, &mat_ops_A, &log);
    arnoldi_t arnoldi(&vec_ops_N, &vec_ops_m, &mat_ops_N, &mat_ops_m, &sys_op, &log);
    lin_op.set_matrix_ptr(A);
    sys_op.set_matrix_ptr(PA);

    iram_t IRAM(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &lapack, &arnoldi, &sys_op, &lin_op, &log);
        
    IRAM.set_target_eigs("LR");
    IRAM.set_number_of_desired_eigenvalues(k0);
    IRAM.set_tolerance(1.0e-10);
    IRAM.set_max_iterations(100);

    IRAM.execute();

    mat_ops_A.stop_use_matrix(PA); mat_ops_A.free_matrix(PA);
    mat_ops_A.stop_use_matrix(A); mat_ops_A.free_matrix(A);



    return 0;
}