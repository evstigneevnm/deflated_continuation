#include <iostream>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown.h>

int main(int argc, char const *argv[])
{
    using real =  SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<real>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using ob_prob_t = nonlinear_operators::overscreening_breakdown<vec_ops_t, mat_ops_t>;
    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;
    using mat_file_ops_t = gpu_matrix_file_operations<mat_ops_t>;

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
    ob_prob_t ob_prob( &vec_ops, &mat_ops, params );
    vec_file_ops_t vec_file_ops(&vec_ops);
    mat_file_ops_t mat_file_ops(&mat_ops);

    T_vec u_sol, u_coeff, f_u_coeff, df_alpha_coeff;
    vec_ops.init_vectors(u_sol, u_coeff, f_u_coeff, df_alpha_coeff);
    vec_ops.start_use_vectors(u_sol, u_coeff, f_u_coeff, df_alpha_coeff);
    
    ob_prob.randomize_vector(u_coeff);
    vec_file_ops.write_vector("initial_coeffs.dat", u_coeff);

    ob_prob.F(u_coeff, 2.0, f_u_coeff);
    vec_file_ops.write_vector("fu.dat", u_coeff);
    ob_prob.set_linearization_point(u_coeff, 2.0);
    
    auto J = ob_prob.jacobian_u();
    mat_file_ops.write_matrix("J.dat", J);



    ob_prob.jacobian_alpha(df_alpha_coeff);
    vec_file_ops.write_vector("df_alpha_coeff.dat", df_alpha_coeff);
    ob_prob.jacobian_alpha(u_coeff, 2.0, df_alpha_coeff);
    vec_file_ops.write_vector("df_alpha_coeff_1.dat", df_alpha_coeff);

    std::vector<real> bd_norms;
    ob_prob.norm_bifurcation_diagram(u_coeff, bd_norms);
    for(auto &x: bd_norms)
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    // ob_prob.calucalte_func tion_at_basis(u_coeff);
    // ob_prob.expend_function(u_coeff);
    // ob_prob.form_jacobian_operator(u_coeff, 2.0);
    // ob_prob.form_operator(u_coeff, 2.0, u_sol);
    // ob_prob.form_operator_parameter_derivative(u_coeff, 2.0, u_sol);

    vec_ops.stop_use_vectors(u_sol, u_coeff, f_u_coeff, df_alpha_coeff);
    vec_ops.free_vectors(u_sol, u_coeff, f_u_coeff, df_alpha_coeff);


    return 0;
}