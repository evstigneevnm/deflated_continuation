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

    
    advance_solution(VectorOperations* vec_ops_, Loggin* log_, SystemOperator* sys_op_, NewtonMethod* newton_, Predictoror* predictor_, char continuation_type_ = 'S'):
    vec_ops(vec_ops_),
    log(log_),
    sys_op(sys_op_),
    newton(newton_),
    predictor(predictor_),
    continuation_type(continuation_type_)
    {
        vec_ops->init_vector(x_p); vec_ops->start_use_vector(x_p);
        vec_ops->init_vector(x1_l); vec_ops->start_use_vector(x1_l);
        vec_ops->init_vector(dx10);  vec_ops->start_use_vector(dx10);
        

    }

    ~advance_solution()
    {
        vec_ops->stop_use_vector(x_p); vec_ops->free_vector(x_p);
        vec_ops->stop_use_vector(x1_l); vec_ops->free_vector(x1_l);
        vec_ops->stop_use_vector(dx10); vec_ops->free_vector(dx10);
        
    }
    

    void reset() //can be used to set everything in default state
    {
        predictor->reset_all(); 
    }

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x0, const T& lambda0, const T_vec& x0_s, const T& lambda0_s, T_vec& x1, T& lambda1, T_vec& x1_s, T& lambda1_s)
    {
        bool converged = false;
        bool failed = false;
        T lambda_p;
        log->info("continuation::advance_solution::starting point:");
        log->info_f("   ||x0|| = %le, lambda0 = %le, ||x0_s|| = %le, lambda0_s = %le", (double)vec_ops->norm(x0), (double)lambda0, (double)vec_ops->norm(x0_s), (double)lambda0_s);
        predictor->reset_tangent_space(x0, lambda0, x0_s, lambda0_s);
        while((!converged)&&(!failed))
        {
            predictor->apply(x_p, lambda_p, x1, lambda1);
            T ds_l = predictor->get_ds();
            log->info_f("continuation::predict: ||x_p|| = %le, lambda_p = %le, ||x1|| = %le, lambda1 = %le", (double)vec_ops->norm(x_p), (double)lambda_p, (double)vec_ops->norm(x1), (double)lambda1);
            if(continuation_type == 'S')
            {
                sys_op->set_tangent_space((T_vec&)x0, (T&)lambda0, (T_vec&)x0_s, (T&)lambda0_s, ds_l, continuation_type);
            }
            else if(continuation_type == 'O')
            {
                sys_op->set_tangent_space(x_p, lambda_p, (T_vec&)x0_s, (T&)lambda0_s, ds_l, continuation_type);
            }
            else
            {
                throw std::runtime_error(std::string("continuation::advance_solution (corrector) " __FILE__ " " __STR(__LINE__) " incorrect continuation_type parameter. Only 'S'pherical or 'O'rthogonal can be used") );
            }
            converged = newton->solve(nonlin_op, x1, lambda1);
            if(!converged)
            {
                //failed = predictor->modify_ds();
                //failed = predictor->decrease_ds();
                failed = predictor->decrease_ds_adaptive();

                log->info("continuation::advance_solution failed to converged. Modifiying dS.");
            }
            else
            {
                predictor->increase_ds();
                log->info("continuation::advance_solution converged. Attempting to increase dS.");
            }
        }
        if(converged)
        {
            log->info("continuation::advance_solution::corrector Newton step norms:");
            for(auto& x: *newton->get_convergence_strategy_handle()->get_norms_history_handle())
            {
                
                log->info_f("%le",(double)x);
            }                
        }
        if(failed)
        {
            throw std::runtime_error(std::string("continuation::advance_solution (corrector) " __FILE__ " " __STR(__LINE__) " failed to converge.") );
        }
        bool tangent_obtained = false;

	    if(converged)
        {
            tangent_obtained = sys_op->update_tangent_space(nonlin_op, x1, lambda1, x1_s, lambda1_s);
        }
        if(!tangent_obtained)
        {
            // throw std::runtime_error(std::string("advance_solution::advance_solution (tangent) " __FILE__ " " __STR(__LINE__) " linear system failed to converge.") );
            log->info("continuation::advance_solution::tangent system failed to converge. Nearing singularity with dim(ker(J))>1. Using FD estimaiton.");  
            T ds = predictor->get_ds();
            T d_ds = T(10.0*std::sqrt(2.0)*1.0e-6);
            //T ds_p = ds + d_ds;
            T ds_m = ds - d_ds;
            //T ds_factor_p = ds_p/ds;
            T ds_factor_m = ds_m/ds;
            //x1-x0=dx10
            vec_ops->assign_mul(T(1), x1, T(-1), x0, dx10);
            //minus_point
            vec_ops->assign_mul(ds_factor_m, dx10, T(1), x0, x1_l);
            T lambda1_l = ds_factor_m*(lambda1 - lambda0) + lambda0;

            bool converged_p, converged_m;
            if(continuation_type == 'S')
            {
                sys_op->set_tangent_space((T_vec&)x0, (T&)lambda0, (T_vec&)x0_s, (T&)lambda0_s, ds_m, continuation_type);
            }
            else if(continuation_type == 'O')
            {
                sys_op->set_tangent_space(x1_l, lambda1_l, (T_vec&)x0_s, (T&)lambda0_s, ds_m, continuation_type);
            }
            else
            {
                throw std::runtime_error(std::string("continuation::advance_solution (tnagent) " __FILE__ " " __STR(__LINE__) " incorrect continuation_type parameter. Only 'S'pherical or 'O'rthogonal can be used") );
            }
            converged_m = newton->solve(nonlin_op, x1_l, lambda1_l);
            if(!converged_m)
            {
                log->warning("continuation::advance_solution::newton solver failed for additional point in tangent");
                vec_ops->assign(x0_s, x1_s);
                lambda1_s = lambda0_s;
            }
            else
            {
                lambda1_s = (lambda1 - lambda1_l)/d_ds;
                vec_ops->assign_mul(T(-1)/d_ds, x1_l, T(1)/d_ds, x1, x1_s);    
                T norm = vec_ops->norm_rank1(x1_s, lambda1_s);
                lambda1_s/=norm;
                vec_ops->scale(T(1)/norm, x1_s);
                log->info_f("continuation::advance_solution::||(x_s, l_s)|| = %le", lambda1_s*lambda1_s + vec_ops->scalar_prod(x1_s, x1_s) );
            }
        }


    }


private:
    VectorOperations* vec_ops;
    SystemOperator* sys_op;
    NewtonMethod* newton;
    Predictoror* predictor;
    T_vec x_p, x1_l, dx10;
    Loggin* log;
    char continuation_type;

};




}

#endif
