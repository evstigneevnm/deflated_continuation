#ifndef __CONTINUATION__INITIAL_TANGENT_H__
#define __CONTINUATION__INITIAL_TANGENT_H__

/**
    Class to get the initial tangent space

*/

#include <string>
#include <stdexcept>
#include <cmath>

#include <iostream>

namespace continuation
{

template<class VectorOperations, class Loggin, class NewtonMethod, class NonlinearOperator, class LinearOperator, class LinearSystemSolver>
class initial_tangent
{

public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;


    initial_tangent(VectorOperations*& vec_ops_,  Loggin* log_, NewtonMethod* newton_, LinearOperator*& lin_op_, LinearSystemSolver*& lin_solv_, bool verbose_=true):
    vec_ops(vec_ops_),
    log(log_),
    newton(newton_),
    lin_op(lin_op_),
    lin_solv(lin_solv_),
    verbose(verbose_)
    {
        vec_ops->init_vector(f); vec_ops->start_use_vector(f);
        vec_ops->init_vector(f1); vec_ops->start_use_vector(f1);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
    }
    ~initial_tangent()
    {
        vec_ops->stop_use_vector(f); vec_ops->free_vector(f);
        vec_ops->stop_use_vector(f1); vec_ops->free_vector(f1);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    }

    bool execute(NonlinearOperator*& nonlin_op, const T sign, const T_vec& x, const T& lambda, T_vec& x_s, T& lambda_s)
    {
        log->info("continuation::initial_tangent: execute starts.");

        bool linear_system_converged = false;
    
        nonlin_op->set_linearization_point(x, lambda);
        nonlin_op->jacobian_alpha(f);
        
        
        //This is important!!!
        vec_ops->assign_scalar(T(0.0), x_s);
        //

        vec_ops->add_mul_scalar(T(0.0), T(-1.0), f);

        T tolerance_local = T(1.0e-8)*vec_ops->get_l2_size();

        lin_solv->get_linsolver_handle_original()->monitor().set_temp_tolerance(tolerance_local);
        lin_solv->get_linsolver_handle_original()->monitor().set_temp_max_iterations(10000);
        linear_system_converged = lin_solv->solve((*lin_op), f, x_s);
        
        T minimum_resid = lin_solv->get_linsolver_handle_original()->monitor().resid_norm_out();
        int iters_performed = lin_solv->get_linsolver_handle_original()->monitor().iters_performed();
        log->info_f("desired residual = %le, minimum attained residual = %le with %i iterations.", tolerance_local, minimum_resid, iters_performed);        
        lin_solv->get_linsolver_handle_original()->monitor().restore_max_iterations();
        lin_solv->get_linsolver_handle_original()->monitor().restore_tolerance();

        if(linear_system_converged)
        {
            T z_sq = vec_ops->scalar_prod(x_s, x_s); //(dx,x_0_s)
            lambda_s = sign/std::sqrt(z_sq+T(1.0));
            vec_ops->add_mul_scalar(T(0.0), lambda_s, x_s); 
	    
        //TODO: do smth with the norm
	    //norm differs greatly for large
	    //and small systems

            T norm = vec_ops->norm_rank1(x_s, lambda_s);
            lambda_s/=norm;
            vec_ops->scale(T(1)/norm, x_s);
            //vec_ops->scale(T(1)/T(vec_ops->get_l2_size()), x_s);
            log->info("continuation::initial_tangent: execute ends successfully.");

        }
        else
        {
            log->warning("continuation::initial_tangent: execute falied to converge. Attempting to use approximate tangent solution via the Newton-Raphson method.");
            T x_norm = vec_ops->norm(x);
            T d_lambda = sign*T(1.0)/x_norm;
            T lambda1 = lambda + d_lambda;
            vec_ops->assign(x, x1); //guess for x1 
            bool converged = newton->solve(nonlin_op, x1, lambda1);
            if(!converged)
            {
                //newton method failed to converge!
                throw std::runtime_error(std::string("continuation::initial_tangent " __FILE__ " " __STR(__LINE__) " tangent space couldn't be obtained - Newton method failed to converge.") );
            }
            else
            {
                lambda_s = lambda1 - lambda; //lambda_s = ds*d(lambda)/ds
                //x_s = x1 - x;      
                vec_ops->assign_mul(T(1.0), x1, T(-1.0), x, x_s);  //x_s = ds*d(x)/ds
                T ds_l = vec_ops->norm_rank1(x_s, lambda_s); 
                lambda_s/=ds_l;
                vec_ops->scale(T(1.0)/ds_l, x_s);
                log->info_f("continuation::initial_tangent: estimated local ds = %le", (double) ds_l);
                linear_system_converged = true;
                log->info("continuation::initial_tangent: Newton-Raphson estimate ends successfully.");
            }

        }
        return linear_system_converged;

    }


private:
    VectorOperations* vec_ops;
    Loggin* log;
    NewtonMethod* newton;
    LinearOperator* lin_op;
    LinearSystemSolver* lin_solv;
    bool verbose;
    T_vec f, f1;
    T_vec x1;
    
};

}


#endif
