#ifndef __STABILITY__SYSTEM_OPERATOR_SHIFT_AND_INVERSE_H__
#define __STABILITY__SYSTEM_OPERATOR_SHIFT_AND_INVERSE_H__
    


#include <stdexcept>
#include <cmath>

namespace stability
{

/**
 * @brief      System operator for shift-inverse tranformation (A-sigma*E)u^{n+1}=(A+mu*E)u^{n} Eigenvalues at (r,*) are mapped to unit circle, where r = (sigma - mu)/2
 *
 * @tparam     VectorOperations    { description }
 * @tparam     NonlinearOperator   { description }
 * @tparam     LinearOperator      { supposed to have a and b in (b*A+a*E) }
 * @tparam     LinearSystemSolver  { description }
 * @tparam     Log                 { description }
 */
template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSystemSolver, class Log>
class system_operator_shift_inverse
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    system_operator_shift_inverse(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_, LinearOperator* lin_op_, LinearSystemSolver* lin_solv_, Log* log_):
    vec_ops(vec_ops_),
    nonlin_op(nonlin_op_),
    lin_op(lin_op_),
    lin_solv(lin_solv_),
    log(log_)
    {
        tolerance = 1.0e-5;
        max_iterations = 5000;
        vec_ops->init_vector(r_v); vec_ops->start_use_vector(r_v);
    }

    ~system_operator_shift_inverse()
    {
        vec_ops->stop_use_vector(r_v); vec_ops->free_vector(r_v);
    }

    void set_sigma(const T sigma_)
    {
        sigma = sigma_;
    }

    void set_tolerance(T tolerance_)
    {
        tolerance = tolerance_;
    }
    void set_max_iterations(unsigned int max_iter_)
    {
        max_iterations = max_iter_;
    }

    bool solve(const T_vec& v_in, T_vec& v_out)
    {
        // if(!linearization_set) throw std::runtime_error("system_operator_shift_inverse: linearization point not set");

        T tolerance_local = tolerance*vec_ops->get_l2_size();
        lin_solv->monitor().set_temp_max_iterations(max_iterations);

        bool linear_system_converged = false;
        int iter = 0;
        T factor = T(1.0);

        chech_nan_norm(v_in, "input vector v_in is nan.");
        lin_op->set_bA_plus_a(T(1.0), -sigma);
        
        while((!linear_system_converged)&&(iter<max_retries))
        {
            vec_ops->assign_scalar(T(0.0), v_out);  //THIS IS A MUST for an inexact arnoldi process
            lin_solv->monitor().set_temp_tolerance(tolerance_local*factor);
                        
            linear_system_converged = lin_solv->solve((*lin_op), v_in, v_out);   

            T minimum_resid = lin_solv->monitor().resid_norm_out();
            int iters_performed = lin_solv->monitor().iters_performed();
            
            lin_solv->monitor().restore_max_iterations();
            lin_solv->monitor().restore_tolerance(); 

            if(!linear_system_converged)
                log->warning_f("system_operator_shift_inverse: linear solver failed, desired residual = %le, minimum attained residual = %le with %i iterations. Retrying...", tolerance_local*factor, minimum_resid, iters_performed); 
            iter++;
            factor*=T(5.0);  
        }
        if(!linear_system_converged)
        {
            throw std::runtime_error("system_operator_shift_inverse: linear solver failed to converge."); 
        }
      

        return linear_system_converged;
    }

    void set_linerization_point(const T_vec& v_0, const T lambda)
    {
        nonlin_op->set_linearization_point(v_0, lambda);
        if(vec_ops->norm(v_0) == T(0.0))
            zero_linearization = true;
        else
            zero_linearization = false;

        linearization_set = true;
    }

    std::string target_eigs()
    {
        return "LM";  //eigenvalues desired for the transormed spectrum
    }


private:
    VectorOperations* vec_ops;
    NonlinearOperator* nonlin_op;
    LinearOperator* lin_op;
    LinearSystemSolver* lin_solv;
    Log* log;

    T tolerance;
    T sigma = -T(10.0);
    T mu = T(10.0);
    unsigned int max_iterations;
    unsigned int max_retries = 4;
    bool linearization_set = false;
    bool zero_linearization = false;
    T_vec r_v;
    T_vec dx_v;


    void chech_nan_norm(const T_vec v_, const std::string& msg_)
    {
        T norm_ = vec_ops->norm(v_);
        if(std::isnan(norm_) )
        {
            throw std::runtime_error("system_operator_shift_inverse:: chech_nan_norm: " + msg_);
        }
    }

    /*
    //This approach doesn't work any better than the plain system solution

    // call to this funciton: linear_system_converged = iterate_lin_solver(v_in, v_out);


    bool iterate_lin_solver(const T_vec& u_in, T_vec& u_out)
    {
        T rhs_norm = 1.0;
        int iter = 0;
        bool iterations_converged = false;
        vec_ops->assign_scalar(T(0.0), dx_v);
        while((rhs_norm>1.0e-6)&&(iter<20))
        {
            form_rhs(u_in, u_out, r_v);
            lin_solv->solve((*lin_op), r_v, dx_v);
            vec_ops->add_mul(T(1.0), dx_v, T(1.0), u_out); 
            rhs_norm = vec_ops->norm(r_v);
            
            log->info_f("system_operator_stability.iterate_lin_solver: resid = %le, iter = %i", double(rhs_norm), iter);

            iter++;
            
        }

        if(rhs_norm <= 1.0e-6)
            iterations_converged = true;

        return iterations_converged;

    }

    void form_rhs(const T_vec& u_in, const T_vec& u_out, T_vec& r_out) const
    {
        lin_op->apply(u_out, r_out);
        //y := mul_x*x + mul_y*y
        vec_ops->add_mul(T(1.0), u_in, T(-1.0), r_out);
    }
    */
};

}

#endif