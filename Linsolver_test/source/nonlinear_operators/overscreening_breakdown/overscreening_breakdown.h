#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_H__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_H__


#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include <common/macros.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
// #include <nonlinear_operators/overscreening_breakdown/plot_solution.h>
#include <common/gpu_file_operations.h>

#include <tuple>
#include <memory>

namespace nonlinear_operators
{

/**
*    Problem class for the overscreening breakdown
*    (1)    u_{xx}-delta^2u_{xxxx} - f(x,u) = 0, 0<=x<=+\infty
*    (2)    u(0) = u_0, u_{xxx}(0) = 0
*    (3)    g(x) = 1.0/(sqrt(2*pi))*exp(-(1/2)*((x)/sigma)**2 ) 
*    (4)    f(u,x) = num/din,
*    (5)    num(x,u) = sinh(u) - g(x)*0.5*mu*exp(u),
*    (6)    din(x,u) = 1 + 2.0*gamma*sinh(u/2.0)**2
*    
*    Parameters:
*      sigma \in [0,10],
*      gamma = 1/2,
*      delta = 1,
*      u_0 = {0.1,0.5,1}
*    
*    MAIN PARAMETER: \lambda = sigma \in [0,10]
*       For sigma=0 we have g(x) = 1/sqrt(2pi)
*/

template<class VecOps, class MatOps, class Kern = overscreening_breakdown_ker<VecOps, MatOps>, class FileOperations = gpu_file_operations<VecOps> >
class overscreening_breakdown
{
private:
    using vec_ops_t = VecOps;
    using mat_ops_t = MatOps;
    using T = typename VecOps::scalar_type;
    using T_vec = typename VecOps::vector_type;
    using T_mat = typename MatOps::matrix_type;

    //class for plotting
    // using plot_t = plot_solution<vec_ops_t>;

    //class for low level CUDA kern_els
    using kern_t = Kern;//overscreening_breakdown_ker<vec_ops_t, mat_ops_t>;
    using file_ops_t = FileOperations;// gpu_file_operations<VecOps>;



private:

    vec_ops_t *vec_ops_;
    mat_ops_t *mat_ops_;
    file_ops_t *file_ops_ = nullptr;
    size_t N_; 
    kern_t* kern_;
    // plot_t* plot_;
    T_vec u_0_, u_solution_, x_points_;
    T delta_0_;


public:
    template<class Params>
    overscreening_breakdown(vec_ops_t* vec_ops, mat_ops_t* mat_ops, const Params& params): 
    vec_ops_(vec_ops), 
    mat_ops_(mat_ops)
    {
        
        N_ = vec_ops_->get_vector_size();
        kern_ = new kern_t(vec_ops, mat_ops_, params);
        // plot = new plot_t();
        vec_ops_->init_vectors(u_0_, u_solution_, x_points_);
        vec_ops_->start_use_vectors(u_0_, u_solution_, x_points_);
        file_ops_ = new file_ops_t(vec_ops_);
        std::srand(unsigned(std::time(0)));
    }

    ~overscreening_breakdown()
    {
        delete file_ops_;
        vec_ops_->stop_use_vectors(u_0_, u_solution_, x_points_);
        vec_ops_->free_vectors(u_0_, u_solution_, x_points_);
        delete kern_;
        // delete plot;
    }

    vec_ops_t* get_vec_ops_ref() const
    {
        return vec_ops_;
    }
    //   F(u,alpha)=v
    void F(const T_vec& u, const T param_p, T_vec& v)
    {
        kern_->form_operator(u, param_p, v);
    }

    //sets (u_0, alpha_0) for jacobian linearization
    //stores alpha_0, u_0  MUST NOT BE CHANGED!!!
    void set_linearization_point(const T_vec& u_0, const T param_p)
    {
        vec_ops_->assign(u_0, u_0_);
        delta_0_ = param_p;
    }

    //Jacobian as a dense matrix
    //the matrix is stored in the 'kern_' class
    T_mat& jacobian_u() const
    {
        kern_->form_jacobian_operator(u_0_, delta_0_);
        return kern_->get_jacobi_matrix();
    }
    void jacobian_u(const T_vec& x, T_vec& f) const
    {
        kern_->form_jacobian_operator(u_0_, delta_0_);
        kern_->apply_jacobi_matrix(x, f);
    }

    void exact_solution(const T lambda, T_vec& u_out)
    {
        kern_->exact_solution(u_out);
    }
    void project(T_vec& u_)
    {
        //we don't need to projet to the
    }    
    //solves exactly a linear system.
    void solve_linear_system(const T_mat& matrix, const T_vec& b, T_vec& x) const
    {
        kern_->solve_linear_system(matrix, b, x);
    }
    //solves exactly a linear system. In this case the matrix is corrupted    
    void solve_linear_system(T_mat& matrix, T_vec& b2x) const
    {
        kern_->solve_linear_system(matrix, b2x);
    }
    //solves shiftde linear system in the form (alpha A +beta E)x = b
    void solve_linear_system(const T alpha, const T_mat& matrix, const T beta, const T_vec& b, T_vec& x) const
    {
        kern_->solve_linear_system(alpha, matrix, beta, b, x);
    }    
    void solve_linear_system(const T alpha, T_mat& matrix, const T beta, T_vec& b2x) const
    {
        kern_->solve_linear_system(alpha, matrix, beta, b2x);
    }
    //variational jacobian J=dF/dalpha

    void jacobian_alpha(T_vec& dv)
    {
        kern_->form_operator_parameter_derivative(u_0_, delta_0_, dv); 
    }

    void jacobian_alpha(const T_vec& u0, const T& param_p, T_vec& dv)
    {
        kern_->form_operator_parameter_derivative(u0, param_p, dv);
    }

    //funciton that returns std::vector with different bifurcatoin norms

    void norm_bifurcation_diagram(const T_vec& u_in, std::vector<T>& res)
    {
        
        kern_->get_solution_at_basis_points(u_in, u_solution_);

        T val0 = vec_ops_->norm_l2(u_in);
        T val1 = vec_ops_->norm_l2(u_solution_);

        res.clear();
        res.reserve(2);
        res.push_back(val0); 
        res.push_back(val1); 
    }


    void physical_solution(const T_vec& u_in, T_vec& u_out) 
    {
        kern_->get_solution_at_basis_points(u_in, u_out);
    }
    void rhs_physical_solution(const T_vec& u_in, T_vec& rhs_out) 
    {
        kern_->get_right_hand_side(u_in, rhs_out);
    }


    void write_solution_basis(const std::string& f_name, const T_vec& u_in) 
    {   
        
        kern_->fill_points_at_basis(x_points_);
        physical_solution(u_in, u_solution_);
        file_ops_->write_2_vectors_by_side(f_name, x_points_, u_solution_);
        // file_ops_->write_vector(f_name, u_solution_);

    }
    void write_rhs_solution_basis(const std::string& f_name, const T_vec& u_in)
    {
        kern_->fill_points_at_basis(x_points_);
        rhs_physical_solution(u_in, u_solution_);
        file_ops_->write_2_vectors_by_side(f_name, x_points_, u_solution_);        
    }
    void write_solution_domain(const std::string& f_name, const T_vec& u_in) 
    {   
        kern_->fill_points_at_domain(x_points_);
        physical_solution(u_in, u_solution_);
        file_ops_->write_2_vectors_by_side(f_name, x_points_, u_solution_);
    }
    void write_rhs_solution_domain(const std::string& f_name, const T_vec& u_in)
    {
        kern_->fill_points_at_domain(x_points_);
        rhs_physical_solution(u_in, u_solution_);
        file_ops_->write_2_vectors_by_side(f_name, x_points_, u_solution_);
    }



    void randomize_vector(T_vec& u_out, int steps_p = -1)
    {
        int steps = steps_p;
        if(steps_p == -1)
        {
            steps = std::rand()%10+3; 
        }
        // std::cout << "steps = " << steps << std::endl;
        kern_->set_random_smoothed_data(u_out, steps);
    }


    T rhs_integral_from_0_to_infty(const T_vec& vec)
    {

    }

    void write_rhs_domain_integral_from_0(const T_vec& vec, const std::vector<T>& points_at_domain)const
    {
        kern_->integrate_rhs(vec);
        
    }
    

    T check_solution_quality(const T_vec& u_in)
    {
        return isfinite(vec_ops_->norm(u_in));
    }

    T norm(const T_vec& u_in_)
    {
        return vec_ops_->norm(u_in_);
    }
    


};

}

#endif
