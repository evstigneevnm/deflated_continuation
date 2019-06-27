#ifndef __CONTINUATION__ADVANCE_SOLUTION_H__
#define __CONTINUATION__ADVANCE_SOLUTION_H__

#include <string>
#include <stdexcept>

/**
  continuation of a single solution forward or backward on a single step
  execute SOLVE method to continue solution in one step
*/

namespace continuation
{

template<class VectorOperations, class Loggin, class NewtonMethod, class NonlinearOperator, class SystemOperator, class Predictoror>
class advance_solution
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    
    advance_solution(VectorOperations* vec_ops_, Loggin* log_, SystemOperator* sys_op_, NewtonMethod* newton_, Predictoror* predictor_):
    vec_ops(vec_ops_),
    log(log_),
    sys_op(sys_op_),
    newton(newton_),
    predictor(predictor_)
    {
        vec_ops->init_vector(x_p); vec_ops->start_use_vector(x_p);
        vec_ops->init_vector(x1_l); vec_ops->start_use_vector(x1_l);
        

    }

    ~advance_solution()
    {
        vec_ops->stop_use_vector(x_p); vec_ops->free_vector(x_p);
        vec_ops->stop_use_vector(x1_l); vec_ops->free_vector(x1_l);
    }
    

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x0, const T& lambda0, const T_vec& x0_s, const T& lambda0_s, T_vec& x1, T& lambda1, T_vec& x1_s, T& lambda1_s)
    {
        bool converged = false;
        bool failed = false;
        T lambda_p;
        log->info_f("advance_solution::starting point: ||x0|| = %le, lambda0 = %le, ||x0_s|| = %le, lambda0_s = %le", (double)vec_ops->norm(x0), (double)lambda0, (double)vec_ops->norm(x0_s), (double)lambda0_s);
        predictor->reset_tangent_space(x0, lambda0, x0_s, lambda0_s);
        while((!converged)&&(!failed))
        {
            predictor->apply(x_p, lambda_p, x1, lambda1);
            log->info_f("predict: ||x_p|| = %le, lambda_p = %le, ||x1|| = %le, lambda1 = %le", (double)vec_ops->norm(x0), (double)lambda_p, (double)vec_ops->norm(x1), (double)lambda1);
            sys_op->set_tangent_space(x_p, lambda_p,(T_vec&)x0_s, (T&)lambda0_s);
            converged = newton->solve(nonlin_op, x1, lambda1);
            if(!converged)
            {
                failed = predictor->modify_ds();

                log->info("advance_solution::failed to converged. Modifiying dS.");
            }
        }
        if(converged)
        {
            log->info("advance_solution::corrector Newton step norms:");
            for(auto& x: *newton->get_convergence_strategy_handle()->get_norms_history_handle())
            {
                log->info_f("%le",(double)x);
            }                
        }
        if(failed)
        {
            throw std::runtime_error(std::string("advance_solution::advance_solution (corrector) " __FILE__ " " __STR(__LINE__) " failed to converge.") );
        }
        bool tangent_obtained = false;

	    if(converged)
        {
            tangent_obtained = sys_op->update_tangent_space(nonlin_op, x1, lambda1, x1_s, lambda1_s);
        }
        if(!tangent_obtained)
        {
           // throw std::runtime_error(std::string("advance_solution::advance_solution (tangent) " __FILE__ " " __STR(__LINE__) " linear system failed to converge.") );
           log->info("advance_solution::tangent system failed to converge. Nearing singularity with ker(J)>1. Using FD estimaiton.");  
           T ds = predictor->get_ds();
           T d_ds = T(1.0e-5);
           T ds_l = ds + d_ds;
           vec_ops->assign_mul(T(1)+d_ds, x1, x1_l);
           T lambda1_l = lambda1*(T(1)+d_ds);
           bool converged_l;
           sys_op->set_tangent_space(x1, lambda1,(T_vec&)x0_s, (T&)lambda0_s);
           converged_l = newton->solve(nonlin_op, x1_l, lambda1_l);
           if(!converged_l)
           {
                log->info("advance_solution::newton solver failed for additional point in tangent");
                vec_ops->assign(x0_s, x1_s);
                lambda1_s = lambda0_s;
           }
           else
           {
               lambda1_s = (lambda1_l - lambda1);
               vec_ops->assign_mul(T(1), x1_l, T(-1), x1, x1_s);
               T x1sTx1s = vec_ops->scalar_prod(x1_s, x1_s);
               T lambda1s_sq = lambda1_s*lambda1_s;
               T norm_vec = std::sqrt(x1sTx1s + lambda1s_sq);
               lambda1_s /= norm_vec;
               vec_ops->add_mul_scalar(T(0), T(1/norm_vec), x1_s);
               //log->info_f("advance_solution::||(x_s, l_s)|| = %le", lambda1_s*lambda1_s + vec_ops->scalar_prod(x1_s, x1_s) );
            }
        }


    }


private:
    VectorOperations* vec_ops;
    SystemOperator* sys_op;
    NewtonMethod* newton;
    Predictoror* predictor;
    T_vec x_p, x1_l;
    Loggin* log;

};




}

#endif
