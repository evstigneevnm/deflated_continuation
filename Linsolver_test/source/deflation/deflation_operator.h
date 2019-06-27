#ifndef __DEFLATION_OPERATOR_H__
#define __DEFLATION_OPERATOR_H__


namespace deflation
{

template<class VectorOperations, class NewtonMethod, class NonlinearOperator, class SolutionStorage, class Logging>
class deflation_operator
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;


    deflation_operator(VectorOperations* vec_ops_, Logging* log_, NewtonMethod* newton_, unsigned int max_retries_):
    vec_ops(vec_ops_),
    log(log_),
    newton(newton_),
    max_retries(max_retries_)
    {
        number_of_solutions = 0;
        vec_ops->init_vector(u_in); vec_ops->start_use_vector(u_in);
        vec_ops->init_vector(u_out); vec_ops->start_use_vector(u_out);
        //vec_ops->init_vector(u_out_1); vec_ops->start_use_vector(u_out_1);
        
    }
    
    ~deflation_operator()
    {
        vec_ops->stop_use_vector(u_in); vec_ops->free_vector(u_in);
        vec_ops->stop_use_vector(u_out); vec_ops->free_vector(u_out);
        //vec_ops->stop_use_vector(u_out_1); vec_ops->free_vector(u_out_1);
    }
    

    bool find_solution(T lambda_0, NonlinearOperator* nonlin_op, SolutionStorage* sol_storage)
    {
        T lambda = lambda_0;
        bool found_solution = false;
        unsigned int retries = 0;
        while((retries<max_retries)&&(found_solution==false))
        {
            nonlin_op->randomize_vector(u_in);
            found_solution = newton->solve(nonlin_op, u_in, lambda_0, u_out, lambda);
            retries++;
            if(!found_solution)
            {
                log->info_f("deflation::retrying, attempt %i\n", retries);
            }
        }
        if(found_solution)
        {
            sol_storage->push(u_out);

            log->info("deflation::convergence_norms:");
            for(auto& x: *newton->get_convergence_strategy_handle()->get_norms_history_handle())
            {
                
                log->info_f("%le", x);
            }
            // if(verbose) printf("solving with simple Newton solver to increase accuracy\n");
            // newton->solve(nonlin_op, u_out, lambda_0, u_out_1);
        }        

        return found_solution;
    }

    void execute_all(T lambda_0, NonlinearOperator* nonlin_op, SolutionStorage* sol_storage)
    {
        bool found_solution = true;
        
        number_of_solutions = 0;
        while(found_solution)
        {
            found_solution = find_solution(lambda_0, nonlin_op, sol_storage);
            if(found_solution)
            {
                number_of_solutions++;
                log->info_f("deflation::========== found %i solutions ==========", number_of_solutions);
            }
            
        }
        log->info_f("deflation::========== found %i solutions for parameter %lf ======", number_of_solutions, (double)lambda_0);        
    }


private:
    VectorOperations* vec_ops;
    NewtonMethod* newton;
    unsigned int max_retries;
    unsigned int number_of_solutions;
    T_vec u_in, u_out, u_out_1;
    Logging* log;

};

}

#endif