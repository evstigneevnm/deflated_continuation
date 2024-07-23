#ifndef __DEFLATION_OPERATOR_H__
#define __DEFLATION_OPERATOR_H__


/**
*  main deflation operator that implements basic routines to find and deflate solutions:
*  -- find solution - finds a solution using solution storage container to deflate;
*  -- find and strore - calls find solution and adds it to the storage container;
*  -- execute all - calls find and store untill all solutions are found.
*/
#include <exception>
#include <vector>
#include <string>
#include <fstream>

namespace deflation
{

template<class VectorOperations, class NewtonMethod, class NonlinearOperator, class SolutionStorage, class Logging>
class deflation_operator
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;


    deflation_operator(VectorOperations* vec_ops_, Logging* log_, NewtonMethod* newton_, unsigned int max_retries_ = 5):
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
    
    void set_max_retries(unsigned int max_retries_)
    {
        max_retries = max_retries_;
    }

    void get_solution_ref(T_vec& sol_ref_)
    {
        sol_ref_ = u_out;
    }

    bool find_solution(T lambda_0, NonlinearOperator*& nonlin_op)
    {
        T lambda = lambda_0;
        bool found_solution = false;
        unsigned int retries = 0;
        while((retries<max_retries)&&(found_solution==false))
        {
            nonlin_op->randomize_vector(u_in);
            // nonlin_op->exact_solution(lambda_0, u_out);
            // vec_ops->add_mul_scalar(0.0, lambda_0, u_in);
            // vec_ops->add_mul(0.1, u_out, u_in);
            
            try
            {
                found_solution = newton->solve(nonlin_op, u_in, lambda_0, u_out, lambda);
            }
            catch(const std::exception& e)
            {
                log->error_f("deflation::find_solution: exception: %s\n", e.what() );
                found_solution = false;
            }
            retries++;
            if(!found_solution)
            {
                log->info_f("deflation::retrying, attempt %i\n", retries);
            }
            if( !norms_file_name_.empty() )
            {
                all_norms.push_back( *newton->get_convergence_strategy_handle()->get_norms_history_handle() );
            }        
        }
        if(found_solution)
        {
            log->info("deflation::convergence_norms:");
            for(auto& x: *newton->get_convergence_strategy_handle()->get_norms_history_handle())
            {
                log->info_f("%le", static_cast<double>(x)) ;
            }            
        }
        return(found_solution);      
    }


    bool find_add_solution(T lambda_0, NonlinearOperator* nonlin_op, SolutionStorage* sol_storage)
    {
        bool found_solution = find_solution(lambda_0, nonlin_op);
        if(found_solution)
        {
            sol_storage->push_back(u_out);

        }        

        return found_solution;
    }

    void save_norms(const std::string& file_name)
    {
        norms_file_name_ = file_name;
    }

    void execute_all(T lambda_0, NonlinearOperator* nonlin_op, SolutionStorage* sol_storage)
    {
        bool found_solution = true;
        number_of_solutions = 0;
        while(found_solution)
        {
            found_solution = find_add_solution(lambda_0, nonlin_op, sol_storage);
            if(found_solution)
            {
                number_of_solutions++;
            }
            log->info_f("deflation::========== found %i solutions ==========", number_of_solutions);

        }
        log->info_f("deflation::========== found %i solutions for parameter %lf ======", number_of_solutions, (double)lambda_0);        

        if( !norms_file_name_.empty() )
        {
            write_norm_file();
        }


    }


private:
    VectorOperations* vec_ops;
    NewtonMethod* newton;
    unsigned int max_retries;
    unsigned int number_of_solutions;
    T_vec u_in, u_out, u_out_1;
    Logging* log;
    std::string norms_file_name_;
    std::vector<std::vector<T>> all_norms;


    void write_norm_file()
    {
        std::size_t N = all_norms.size();
        std::size_t max_size = 0;
        for(auto& x: all_norms) max_size = x.size()>max_size?x.size():max_size; 
        std::cout << "N = " << N << std::endl;
        std::cout << "max_size = " << max_size << std::endl;
        std::ofstream f(norms_file_name_, std::ofstream::out);
        if (!f) throw std::runtime_error("deflation_operator::write_norm_file: error while opening file " + norms_file_name_);

        for(std::size_t k=0;k<max_size;k++)
        {
            for(std::size_t j=0;j<N;j++)
            {
                auto size_l = all_norms[j].size();
                if(k>=size_l)
                {
                    f << ",";
                }
                else
                {
                    f << all_norms[j][k] << ",";
                }

            }
            f << std::endl;
        }
        f.close();

    }


};

}

#endif