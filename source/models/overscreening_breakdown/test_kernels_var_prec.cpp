#include <common/cpu_vector_operations_var_prec.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_params.h>

int main(int argc, char const *argv[])
{
    const unsigned int fp_bits = 100;
    using vec_ops_t = cpu_vector_operations_var_prec<fp_bits>;
    using T = typename vec_ops_t::scalar_type;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = cpu_matrix_vector_operations_var_prec<vec_ops_t>;
    using T_mat = typename mat_ops_t::matrix_type;
    using ob_ker_t = nonlinear_operators::overscreening_breakdown_ker<vec_ops_t, mat_ops_t>;
    using params_t = params_s<T>;

    size_t N = 10;
    params_t params;
    vec_ops_t vec_ops(N);
    mat_ops_t mat_ops(vec_ops.get_vector_size(), vec_ops.get_vector_size(), &vec_ops );
    
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