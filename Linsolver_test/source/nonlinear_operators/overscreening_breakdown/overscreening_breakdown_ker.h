#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_H__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_H__

#include <cuda_runtime.h>
#include <iostream>
#include <contrib/scfd/include/scfd/utils/device_tag.h>
#include "detail/matrix_wrap.h"
#include "detail/overscreening_breakdown_problem.h"

namespace nonlinear_operators
{

template<class VectorOperations, class MatrixOperations>
class overscreening_breakdown_ker
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using mat_ops_t = MatrixOperations;
    using T_mat = typename MatrixOperations::matrix_type;
    using matrix_wrap = detail::matrix_wrap<T, T_mat>;
    using problem_type = detail::overscreening_breakdown_problem<T>;

    
    template<class Params>
    overscreening_breakdown_ker(VectorOperations* vec_ops, MatrixOperations* mat_ops, const Params& params):
    vec_ops_(vec_ops), mat_ops_(mat_ops),
    apply_inverse_diag_(false)
    {
        N = vec_ops_->get_vector_size();

        linear_operator = new matrix_wrap( N, N );
        mass_matrix = new matrix_wrap( N, N );
        stiffness_matrix = new matrix_wrap( N, N );
        shifted_linear_operator = new matrix_wrap( N, N );
        eye = new matrix_wrap(N, N);
        iD = new matrix_wrap(N, N);
        iDA = new matrix_wrap(N, N);

        problem = new problem_type(N, params.param_number, params.sigma, params.L, params.delta, params.gamma, params.mu, params.u0);
        
        mat_ops_->init_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        mat_ops_->start_use_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        
        vec_ops_->init_vectors(u_solution, u_randomized, iDb, basis_points);
        vec_ops_->start_use_vectors(u_solution, u_randomized, iDb, basis_points);

        form_mass_matrix(N, *problem, *mass_matrix);
        form_stiffness_matrix(N, *problem, *stiffness_matrix, *iD);
        form_rhs_linearization_matrix(N, *problem, u_solution, *linear_operator);
        set_identity_matrix(N, *eye);
        
    }
    ~overscreening_breakdown_ker()
    {
        mat_ops_->stop_use_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        mat_ops_->free_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        vec_ops_->stop_use_vectors(u_solution, u_randomized, iDb, basis_points);
        vec_ops_->free_vectors(u_solution, u_randomized, iDb, basis_points); 
        delete iDA;
        delete iD;
        delete eye;
        delete shifted_linear_operator;
        delete stiffness_matrix;
        delete mass_matrix;
        delete linear_operator;
        delete problem;
    }

    void get_solution_at_basis_points(const T_vec& u, T_vec& u_solution_p)
    {
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution_p);
    }

//  jacobian w.r.t. the expansion coefficients.
    void form_jacobian_operator(const T_vec& u, const T sigma)
    {
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution);
        reformulate_bcs(N, *problem, u_solution);
        problem->set_state(sigma);
        form_rhs_linearization_matrix(N, *problem, u_solution, *linear_operator);
        mat_ops_->geam('N', N, N, 1.0, stiffness_matrix->data, -1.0, linear_operator->data, linear_operator->data);
    }
    T_mat& get_jacobi_matrix()
    {
        return linear_operator->data;
    }
    void apply_jacobi_matrix(const T_vec& x, T_vec& f) const
    {
        mat_ops_->gemv('N', linear_operator->data, 1.0, x, 0.0, f);
    }


// returns F(u, sigma), where F is the whole problem.
    void form_operator(const T_vec& u, const T sigma_p, T_vec& res)
    {
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution);
        reformulate_bcs(N, *problem, u_solution);
        problem->set_state(sigma_p);  
        form_right_hand_side(N, *problem, u_solution, res);
        mat_ops_->gemv('N', stiffness_matrix->data, 1.0, u, 1.0, res);
    }

//  derivative of the RHS w.r.t. sigma:
//  -((E^(u-x^2/(2 \[Sigma]^2)) x^2 \[Mu])/(2 Sqrt[2 \[Pi]] \[Sigma]^3 (1+2 \[Gamma] Sinh[u/2]^2)))
//  -((exp(u-x^2/(2*sigma^2))*x^2*mu)/(2*sqrt(2*M_PI)*sigma^3*(1+2*gamma*sinh(u/2)^2)))
//  -((exp(u-x*x/(2.0*sigma*sigma))*x*x*mu)/(2.0*sqrt(2.0*M_PI)*sigma*sigma*sigma*(1.0+2.0*gamma*sinh(0.5*u)*sinh(0.5*u))))
    void form_operator_parameter_derivative(const T_vec& u, const T sigma_p, T_vec& res)
    {
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution);
        reformulate_bcs(N, *problem, u_solution);
        problem->set_state(sigma_p);
        form_right_hand_parameter_derivative(N, *problem, u_solution, res);
    }

    void calucalte_function_at_basis(T_vec& coeffs, bool mult = false)
    {
        calucalte_function_at_basis(N, *problem, mult, coeffs);
    }
    void expend_function(const T_vec& u_at_points, T_vec& coeffs)
    {
        mat_ops_->gesv(N, mass_matrix->data, u_at_points, coeffs); // mass matrix is not changed.
    }
    void expend_function(T_vec& coeffs)
    {
        mat_ops_->gesv(N, mass_matrix->data, coeffs); // mass matrix is changed.
        form_mass_matrix(N, *problem, *mass_matrix);   //regenerate matrix
    }

//  apply inverse diags in these functions
    void solve_linear_system(const T_mat& matrix, const T_vec& b, T_vec& x)
    {
        if(apply_inverse_diag_)
        {
            // iDA, iDb
            mat_ops_->mat2column_mult_mat(iD->data, matrix, N, 1.0, 0.0, iDA->data);
            mat_ops_->gemv('N', iD->data, 1.0, b, 0.0, iDb);
            mat_ops_->gesv(N, iDA->data, iDb, x);
        }
        else
        {    
            mat_ops_->gesv(N, matrix, b, x);
        }
    }
    void solve_linear_system(const T alpha, const T_mat& matrix, const T beta, const T_vec& b, T_vec& x)
    {         
        mat_ops_->geam('N', N, N, alpha, matrix, beta, eye->data, shifted_linear_operator->data);
        mat_ops_->gesv(N, shifted_linear_operator->data, b, x);
    }

    void solve_linear_system(T_mat& matrix, T_vec& b2x)
    {
        if(apply_inverse_diag_)
        {
            // iDA, iDb
            mat_ops_->mat2column_mult_mat(iD->data, matrix, N, 1.0, 0.0, iDA->data);
            mat_ops_->gemv('N', iD->data, 1.0, b2x, 0.0, iDb);
            mat_ops_->gesv(N, iDA->data, iDb);
        }
        else
        {  
            mat_ops_->gesv(N, matrix, b2x);
        }
    }
    void solve_linear_system(const T alpha, T_mat& matrix, const T beta, T_vec& b2x)
    {        
        mat_ops_->geam('N', N, N, alpha, matrix, beta, eye->data, shifted_linear_operator->data);
        mat_ops_->gesv(N, shifted_linear_operator->data, b2x);
    }

    void set_random_smoothed_data(T_vec& coeffs, int steps)
    {
        vec_ops_->assign_random(coeffs, -10.0, 10.0);
        // vec_ops_->assign_scalar(1.0, coeffs);
        // smooth_random_data(N, coeffs, steps);
        calucalte_function_at_basis(coeffs, true);
        expend_function(coeffs);
        problem->rotate_initial_function(); //switch to another predefined function
    }


    void fill_points_at_basis(T_vec& res)
    {
        fill_points_at_basis(N, *problem, res);
    }
    void fill_points_at_domain(T_vec& res)
    {
        fill_points_at_domain(N, *problem, res);
    }    



    void get_right_hand_side(const T_vec& u, T_vec& rhs_p)
    {
        get_solution_at_basis_points(u, u_solution);
        get_right_hand_side(N, *problem, u_solution, rhs_p);
    }



    T integrate_rhs(const T_vec& u)
    {
        
        T res_all = integrate_solution(N, *problem, u);

    }

    T integrate_rhs_to(const T_vec& u, T x)
    {
        // auto val = *problem.right_hand_side(T x, T u);
        T u_of_x = get_solution_value_at_point(N, x, *problem, u, u_solution, true);
        return u_of_x;
    }

    void exact_solution(T_vec& u)
    {
        get_exact_solution(N, *problem, u);
        expend_function(u);
    }


private:
    problem_type* problem;
    matrix_wrap* stiffness_matrix;
    matrix_wrap* linear_operator;
    matrix_wrap* mass_matrix;
    matrix_wrap* shifted_linear_operator;
    matrix_wrap* eye;
    matrix_wrap* iD;
    matrix_wrap* iDA;

    bool apply_inverse_diag_;
    T_vec u_solution;
    T_vec u_randomized;
    T_vec basis_points;
    T_vec iDb;
    VectorOperations* vec_ops_;
    mat_ops_t* mat_ops_;
    size_t N;

    void form_stiffness_matrix(size_t N, problem_type& problem, matrix_wrap& A, matrix_wrap& iD);
    void form_rhs_linearization_matrix(size_t N, problem_type& problem, T_vec& u_solution, matrix_wrap& B);
    void form_mass_matrix(size_t N, problem_type& problem, matrix_wrap& mass_matrix);
    void form_jacobian_matrix(size_t N, problem_type& problem, matrix_wrap& A);
    void calucalte_function_at_basis(size_t N, problem_type& problem, bool mult, T_vec& res);
    void reformulate_bcs(size_t N, problem_type& problem, T_vec& res);
    void form_right_hand_parameter_derivative(size_t N, problem_type& problem, T_vec& u, T_vec& res);
    void form_right_hand_side(size_t N, problem_type& problem, T_vec& u, T_vec& res);
    void smooth_random_data(size_t N, T_vec& res, int smooth_steps);
    void fill_points_at_basis(size_t N,  problem_type& problem, T_vec& res);
    void fill_points_at_domain(size_t N,  problem_type& problem, T_vec& res);
    void set_shifted_matrix(size_t N, T a, matrix_wrap& A, T b, matrix_wrap& aApbE);
    void set_identity_matrix(size_t N, matrix_wrap& E);
    void get_right_hand_side(size_t N, problem_type& problem, const T_vec& u, T_vec& res);
    T get_solution_value_at_point(size_t N, T x, problem_type& problem, const T_vec& u_coeffs, T_vec& temp_storage, bool point_in_domain);
    T integrate_solution(size_t N, problem_type& problem, const T_vec& u_coeffs);
    void get_exact_solution(size_t N, problem_type& problem, T_vec& res);


};

}

#endif