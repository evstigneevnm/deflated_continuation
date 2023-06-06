#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>

#include <common/gpu_file_operations.h>
#include <common/gpu_matrix_file_operations.h>

int main(int argc, char const *argv[])
{
    using real =  SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<real>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;

    using ob_ker_t = nonlinear_operators::overscreening_breakdown_ker<vec_ops_t, mat_ops_t>;

    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;

    struct params_s
    {
        real L = 1.0;
        real gamma = 1.0;
        real delta = 1.0;    
        real mu = 1.0;
        real u0 = 1.0;
    };
    size_t N = 10;
    params_s params;
    cublas_wrap cublas(true);
    vec_ops_t vec_ops(N, &cublas);
    mat_ops_t mat_ops(vec_ops.get_vector_size(), vec_ops.get_vector_size(), vec_ops.get_cublas_ref() );
    vec_file_ops_t vec_file_ops(&vec_ops);
    
    ob_ker_t ob_ker( &vec_ops, &mat_ops, params );
    T_vec u_sol, u_coeff;
    vec_ops.init_vectors(u_sol, u_coeff);
    vec_ops.start_use_vectors(u_sol, u_coeff);
    

    
    ob_ker.calucalte_function_at_basis(u_coeff);
    ob_ker.expend_function(u_coeff);
    ob_ker.form_jacobian_operator(u_coeff, 2.0);
    ob_ker.form_operator(u_coeff, 2.0, u_sol);
    ob_ker.form_operator_parameter_derivative(u_coeff, 2.0, u_sol);

    vec_ops.stop_use_vectors(u_sol, u_coeff);
    vec_ops.free_vectors(u_sol, u_coeff);


    return 0;
}