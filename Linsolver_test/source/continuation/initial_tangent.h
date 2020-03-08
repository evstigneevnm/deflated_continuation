#ifndef __CONTINUATION__INITIAL_TANGENT_H__
#define __CONTINUATION__INITIAL_TANGENT_H__

#include <string>
#include <stdexcept>
#include <cmath>

/**
    Class to get the initial tangent space

*/

namespace continuation
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSystemSolver>
class initial_tangent
{

public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;


    initial_tangent(VectorOperations* vec_ops_, LinearOperator* lin_op_, LinearSystemSolver* lin_solv_, bool verbose_=true):
    vec_ops(vec_ops_),
    lin_op(lin_op_),
    lin_solv(lin_solv_),
    verbose(verbose_)
    {
        vec_ops->init_vector(f); vec_ops->start_use_vector(f);
        vec_ops->init_vector(f1); vec_ops->start_use_vector(f1);
    }
    ~initial_tangent()
    {
        vec_ops->stop_use_vector(f); vec_ops->free_vector(f);
        vec_ops->stop_use_vector(f1); vec_ops->free_vector(f1);
    }

    bool execute(NonlinearOperator* nonlin_op, const T sign, const T_vec& x, const T& lambda, T_vec& x_s, T& lambda_s)
    {
        bool linear_system_converged = false;
        nonlin_op->set_linearization_point(x, lambda);
        nonlin_op->jacobian_alpha(f);
        
        vec_ops->add_mul_scalar(T(0), T(-1), f);

        lin_solv->get_linsolver_handle()->monitor().set_temp_tolerance(T(1.0e-10)*vec_ops->get_l2_size());
        linear_system_converged = lin_solv->solve((*lin_op), f, x_s);
        lin_solv->get_linsolver_handle()->monitor().restore_tolerance();
        if(linear_system_converged)
        {
    	    T z_sq = vec_ops->scalar_prod(x_s, x_s); //(dx,x_0_s)
            lambda_s = sign/std::sqrt(z_sq+T(1));
            vec_ops->add_mul_scalar(T(0), lambda_s, x_s); 
	    
        //TODO: do smth with the norm
	    //norm differs greatly for large
	    //and small systems

            T norm = vec_ops->norm_rank1(x_s,lambda_s);
            lambda_s/=norm;
            vec_ops->scale(T(1)/norm, x_s);
        }
        else
        {
            //linear system failed to converge!
            throw std::runtime_error(std::string("Continuation::initial_tangent " __FILE__ " " __STR(__LINE__) " tangent space couldn't be obtained - linear system failed to converge.") );
        }
        return linear_system_converged;

    }


private:
    VectorOperations* vec_ops;
    LinearOperator* lin_op;
    LinearSystemSolver* lin_solv;
    bool verbose;
    T_vec f, f1;
    
};

}


#endif
