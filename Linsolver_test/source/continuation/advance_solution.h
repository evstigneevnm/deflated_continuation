#ifndef __CONTINUATION__ADVANCE_SOLUTION_H__
#define __CONTINUATION__ADVANCE_SOLUTION_H__

#include <string>
#include <stdexcept>

namespace continuation
{

template<class VectorOperations, class NewtonMethod, class NonlinearOperator, class SystemOperator, class Predictoror>
class advance_solution
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    
    advance_solution(VectorOperations* vec_ops_, SystemOperator* sys_op_, NewtonMethod* newton_, Predictoror* predictor_, bool verbose_ = true):
    vec_ops(vec_ops_),
    sys_op(sys_op_),
    newton(newton_),
    predictor(predictor_),
    verbose(verbose_)
    {
        vec_ops->init_vector(x_p); vec_ops->start_use_vector(x_p);
    }

    ~advance_solution()
    {
        vec_ops->stop_use_vector(x_p); vec_ops->free_vector(x_p);
    }
    

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x0, const T& lambda0, const T_vec& x0_s, const T& lambda0_s, T_vec& x1, T& lambda1, T_vec& x1_s, T& lambda1_s)
    {
        bool converged = false;
        bool failed = false;
        T lambda_p;
        predictor->reset_tangent_space(x0, lambda0, x0_s, lambda0_s);
        while((!converged)&&(!failed))
        {
            predictor->apply(x_p, lambda_p, x1, lambda1);
            sys_op->set_tangent_space(x_p, lambda_p,(T_vec&)x0_s, (T&)lambda0_s);
            converged = newton->solve(nonlin_op, x1, lambda1);
            if(!converged)
            {
                failed = predictor->modify_ds();
                if(verbose)
                {
                    std::cout << "\nfailed to converged. Modifiying dS.\n";
                }
            }

        }
        if(converged)
        {
            if(verbose)
            {
                std::cout << "corrector Newton step norms:" << std::endl;
                for(auto& x: *newton->get_convergence_strategy_handle()->get_norms_history_handle())
                {
                    std::cout << x << std::endl;
                }                

            }

        }
        if(failed)
        {
            throw std::runtime_error(std::string("Continuation::advance_solution (corrector) " __FILE__ " " __STR(__LINE__) " failed to converge.") );
        }
        bool tangent_obtained = false;
        if(converged)
        {
           
            tangent_obtained = sys_op->update_tangent_space(nonlin_op, x1, lambda1, x1_s, lambda1_s);
        }
        if(!tangent_obtained)
        {
            throw std::runtime_error(std::string("Continuation::advance_solution (tangent) " __FILE__ " " __STR(__LINE__) " linear system failed to converge.") );            
        }


    }


private:
    VectorOperations* vec_ops;
    SystemOperator* sys_op;
    NewtonMethod* newton;
    Predictoror* predictor;
    bool verbose;
    T_vec x_p;
    
};




}

#endif
