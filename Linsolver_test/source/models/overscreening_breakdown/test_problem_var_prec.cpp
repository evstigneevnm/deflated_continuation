#include <iostream>
#include <string>
#include <sstream>
#include <common/cpu_vector_operations_var_prec.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_params.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown.h>
#include <common/cpu_matrix_file_operations.h>
#include <common/cpu_file_operations.h>
#include <contrib/scfd/include/scfd/utils/device_tag.h>
#include <contrib/scfd/include/scfd/utils/system_timer_event.h>



int main(int argc, char const *argv[])
{
    using vec_ops_t = cpu_vector_operations_var_prec;
    using T_vec = typename vec_ops_t::vector_type;
    using T = typename vec_ops_t::scalar_type;
    using mat_ops_t = cpu_matrix_vector_operations_var_prec<vec_ops_t>;
    using T_mat = typename mat_ops_t::matrix_type;
    using vec_file_ops_t = cpu_file_operations<vec_ops_t>;
    using mat_file_ops_t = cpu_matrix_file_operations<mat_ops_t>;
    using ob_prob_ker_t = nonlinear_operators::overscreening_breakdown_ker<vec_ops_t, mat_ops_t>;
    using ob_prob_t = nonlinear_operators::overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_ker_t, vec_file_ops_t >;
    using params_t = params_s<T>;
    using timer_t = scfd::utils::system_timer_event;

    size_t N = 150;
    params_t params(N, 0, {1.0, 5.0, 0.5, 5.0, 1.0, 0.1});
    vec_ops_t vec_ops(N);
    mat_ops_t mat_ops(vec_ops.get_vector_size(), vec_ops.get_vector_size(), &vec_ops );
    
    timer_t init_s, init_e;
    init_s.record();
    ob_prob_t ob_prob( &vec_ops, &mat_ops, params );
    init_e.record();
    std::cout << "init elapsed time = " << init_e.elapsed_time(init_s)/1000.0 << " s." << std::endl;
    
    
    vec_file_ops_t vec_file_ops(&vec_ops);
    mat_file_ops_t mat_file_ops(&mat_ops);

    T_vec u_sol, u_coeff, f_u_coeff, df_alpha_coeff;
    vec_ops.init_vectors(u_sol, u_coeff, f_u_coeff, df_alpha_coeff);
    vec_ops.start_use_vectors(u_sol, u_coeff, f_u_coeff, df_alpha_coeff);
    
    
    std::stringstream ss1,ss2;
    for(int j=0;j<10;j++)
    {    
        ss1 << "initial_coeffs_" << j << ".dat";
        ss2 << "initial_u_" << j << ".dat";
        ob_prob.randomize_vector(u_coeff);
        ob_prob.write_solution_basis(ss2.str(), u_coeff);
        vec_file_ops.write_vector(ss1.str(), u_coeff);
        ss1.str(std::string());
        ss2.str(std::string());
    }

    timer_t F_s, F_e;
    F_s.record();
    ob_prob.F(u_coeff, 2.0, f_u_coeff);
    F_e.record();
    std::cout << "F elapsed time = " << F_e.elapsed_time(F_s)/1000.0 << " s." << std::endl;

    vec_file_ops.write_vector("fu.dat", u_coeff);
    ob_prob.set_linearization_point(u_coeff, 2.0);
    
    timer_t J_s, J_e;
    J_s.record();    
    auto J = ob_prob.jacobian_u();
    J_e.record();
    std::cout << "J elapsed time = " << J_e.elapsed_time(J_s)/1000.0 << " s." << std::endl;

    mat_file_ops.write_matrix("J.dat", J);



    ob_prob.jacobian_alpha(df_alpha_coeff);
    vec_file_ops.write_vector("df_alpha_coeff.dat", df_alpha_coeff);
    ob_prob.jacobian_alpha(u_coeff, 2.0, df_alpha_coeff);
    vec_file_ops.write_vector("df_alpha_coeff_1.dat", df_alpha_coeff);

    std::vector<T> bd_norms;
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