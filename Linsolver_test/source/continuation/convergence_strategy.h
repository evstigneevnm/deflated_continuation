#ifndef __CONVERGENCE_STRATEGY_CONTINUATION_H__
#define __CONVERGENCE_STRATEGY_CONTINUATION_H__
/**
convergence rules for Newton iterator for continuation process
*/
#include <cmath>
#include <vector>
#include <utils/logged_obj_base.h>
#include <algorithm> // std::min_element
#include <iterator>  // std::begin, std::end

namespace continuation
{
namespace newton_method_extended
{


template<class vector_operations, class nonlinear_operator, class logging>
class convergence_strategy
{
private:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;
    typedef utils::logged_obj_base<logging> logged_obj_t;

public:    
    convergence_strategy(vector_operations*& vec_ops_, logging*& log_, T tolerance_ = T(1.0e-6), unsigned int maximum_iterations_ = 100, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true, T maximum_norm_increase = 0.0):
    vec_ops(vec_ops_),
    log(log_),
    iterations(0),
    tolerance(tolerance_),
    maximum_iterations(maximum_iterations_),
    newton_wight(newton_wight_),
    newton_wight_initial(newton_wight_),
    verbose(verbose_),
    store_norms_history(store_norms_history_)
    {
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(x1_storage); vec_ops->start_use_vector(x1_storage);
        lambda1_storage = T(0.0);
        vec_ops->init_vector(Fx); vec_ops->start_use_vector(Fx);
        if(store_norms_history)
        {
            norms_evolution.reserve(maximum_iterations);
        }
        norms_storage.reserve(maximum_iterations+1);
        relaxed_tolerance_reached.reserve(maximum_iterations);
        stagnation_max = 10;
        maximum_norm_increase_ = maximum_norm_increase;
        newton_wight_threshold_ = 1.0e-12;
    }
    ~convergence_strategy()
    {
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(x1_storage); vec_ops->free_vector(x1_storage);
        vec_ops->stop_use_vector(Fx); vec_ops->free_vector(Fx);
    }

    void set_convergence_constants(T tolerance_, unsigned int maximum_iterations_, T relax_tolerance_factor_, int relax_tolerance_steps_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true, unsigned int stagnation_max_ = 10, T maximum_norm_increase_p = 0.0, T newton_wight_threshold_p = 1.0e-12)
    {
        tolerance = tolerance_;
        tolerance_0 = tolerance_;
        maximum_iterations = maximum_iterations_;
        newton_wight = newton_wight_;
        newton_wight_initial = newton_wight_;
        store_norms_history = store_norms_history_;
        verbose = verbose_;
        stagnation_max = stagnation_max_;
        relax_tolerance_factor = relax_tolerance_factor_;
        // relax_tolerance_steps = relax_tolerance_steps_;
        current_relax_step = 0;
        if(store_norms_history)
        {
            norms_evolution.reserve(maximum_iterations);
        } 
        norms_storage.reserve(maximum_iterations+1);
        relaxed_tolerance_reached.reserve(maximum_iterations);
        // T d_step = relax_tolerance_factor/T(relax_tolerance_steps);   
        
        // d_step = std::log10(relax_tolerance_factor)/T(relax_tolerance_steps);
        
        // log->info_f("continuation::convergence: check: relax_tolerance_steps = %i, relax_tolerance_factor = %le, d_step = %le, d_step_exp = %le", relax_tolerance_steps, (double)relax_tolerance_factor, (double)d_step, (double)std::pow<T>(T(10), d_step));
        maximum_norm_increase_ = maximum_norm_increase_p;
        newton_wight_threshold_ = newton_wight_threshold_p;

        log->info_f("continuation::convergence: check: relax_tolerance_factor = %le, maximum_norm_increase = %le, newton_wight_threshold = %le", (double)relax_tolerance_factor, (double)maximum_norm_increase_, double(newton_wight_threshold_) );

    }   

    //updates a solution with a newton wight value provided
    T inline update_solution(nonlinear_operator* nonlin_op, T_vec& x, T& lambda, T_vec& delta_x, T& delta_lambda, T_vec& x1, T& lambda1)
    {
        vec_ops->assign_mul(static_cast<T>(1.0), x, newton_wight, delta_x, x1);
        nonlin_op->project(x1); // project to invariant solution subspace. Should be blank if nothing is needed to be projected.
        lambda1 = lambda + newton_wight*delta_lambda;
        nonlin_op->F(x1, lambda1, Fx);
        T normFx1 = vec_ops->norm_l2(Fx);
        return normFx1;
    }




    // bool check_convergence(nonlinear_operator* nonlin_op, T_vec& x, T& lambda, T_vec& delta_x, T& delta_lambda, int& result_status)
    // {
    //     bool finish = false; //states that the newton process should stop.
    //     // result_status defines on how this process is stoped.
    //     reset_wight();
    //     nonlin_op->F(x, lambda, Fx);
    //     T normFx = vec_ops->norm_l2(Fx);
    //     if(!std::isfinite(normFx)) //set result_status = 2 if the provided vector is inconsistent
    //     {
    //         result_status = 2;
    //         finish = true; 
    //     }
    //     if(normFx < tolerance) //do nothing is my kind of problem =)
    //     {
    //         result_status = 0;
    //         log->info_f("continuation::convergence: iteration %i, residuals n: %le < tolerance: %le => finished.",iterations, (double)normFx, (double)tolerance );            
    //         finish = true;
    //     }
    //     T lambda1 = lambda;
    //     T normFx1 = update_solution(nonlin_op, x, lambda, delta_x, delta_lambda, x1, lambda1);
    //     if(normFx1 < tolerance) //converged
    //     {
    //         norms_storage.push_back(normFx1);
    //         lambda = lambda1;
    //         vec_ops->assign(x1, x);
    //         result_status = 0;
    //         finish = true;
    //     }
    //     else
    //     {
    //         lambda = lambda1;
    //         vec_ops->assign(x1, x);
    //         result_status = 1;
    //         finish = false;
    //     }        
    //     //store norm only if the step is successfull
    //     if(store_norms_history)
    //     {
    //         norms_evolution.push_back(normFx1);
    //     }

    //     return finish;

    // }

    bool check_convergence(nonlinear_operator* nonlin_op, T_vec& x, T& lambda, T_vec& delta_x, T& delta_lambda, int& result_status)
    {
        bool finish = false; //states that the newton process should stop.
        // result_status defines on how this process is stoped.
        reset_wight();
        nonlin_op->F(x, lambda, Fx);
        T normFx = vec_ops->norm_l2(Fx);
        if(!std::isfinite(normFx)) //set result_status = 2 if the provided vector is inconsistent
        {
            result_status = 2;
            finish = true; 
        }
        if(normFx < tolerance) //do nothing is my kind of problem =)
        {
            result_status = 0;
            log->info_f("continuation::convergence: iteration %i, residuals n: %le < tolerance: %le => finished.",iterations, (double)normFx, (double)tolerance );            
            return true;
        }
        if(norms_storage.size() == 0)
        {
            //stores initial norm
            norms_storage.push_back(normFx);
        }
        //update solution
        T lambda1 = lambda;
        T normFx1 = update_solution(nonlin_op, x, lambda, delta_x, delta_lambda, x1, lambda1);
        if(!std::isfinite(normFx1)) //quit if the obtained vector is inconsistent
        {
            result_status = 3;
            finish = true; 
        }     
        if(normFx1 < tolerance) //converged
        {
            norms_storage.push_back(normFx1);
            lambda = lambda1;
            vec_ops->assign(x1, x);
            result_status = 0;
            finish = true;
        }
        else //...
        {
            result_status = 1;
            if(iterations>0)
            {
                while( (normFx1 - normFx) > maximum_norm_increase_*normFx )
                {
                    newton_wight *= 0.7;
                    normFx1 = update_solution(nonlin_op, x, lambda, delta_x, delta_lambda, x1, lambda1);
                    log->info_f("continuation::convergence: increase threshold: %.01f, weight update from %le to %le with weight: %le and weight threshold: %le ", maximum_norm_increase_, normFx, normFx1, newton_wight,  newton_wight_threshold_);
                    if(newton_wight < newton_wight_threshold_)
                    {
                        result_status = 4;
                        finish = true;
                        break;
                    }
                }
            }
            if(result_status == 1) //set up solution, not 4
            {
                lambda = lambda1;
                vec_ops->assign(x1, x);
            }
            if( std::abs(normFx1 - normFx) < 1.0e-6*normFx )
            {
                stagnation++;
            }
            if(stagnation > stagnation_max)
            {
                finish = true;
            }
            if(iterations > maximum_iterations)
            {
                finish = true;         
            }            
        }
        //store norm only if the step is successfull
        if(store_norms_history)
        {
            norms_evolution.push_back(normFx1);
        }

        auto min_value = *std::min_element(norms_storage.begin(),norms_storage.end());
        norms_storage.push_back(normFx1);
        iterations++;
        auto result_status_string = parse_result_status(result_status);
        auto finish_string = parse_bool(finish);
        log->info_f("continuation::convergence: iteration: %i, max_iterations: %i, residuals n: %le, n+1: %le, min_value: %le, result_status: %i => %s, is_finished = %s, newton_wight = %le, stagnation = %u ",iterations, maximum_iterations, (double)normFx, (double)normFx1, double(min_value), result_status,  result_status_string.c_str(), finish_string.c_str(), newton_wight, stagnation );

        // store this solution point if the norm is the smalles of all
        if( (min_value >= normFx1)&&((result_status == 1)||(result_status == 4)) )
        {
            vec_ops->assign(x1, x1_storage);
            lambda1_storage = lambda1;
            //signal that relaxed tolerance converged and put it into vector of signals
            if( normFx1 <= tolerance*relax_tolerance_factor  )
            {
                relaxed_tolerance_reached.push_back(true);
            }
            else
            {
                relaxed_tolerance_reached.push_back(false);
            }
        }


        //this sets minimum norm solution that is bellow relaxed tolerance if finish condition is met
        bool relaxed_tolerance_reached_max = *std::max_element(relaxed_tolerance_reached.begin(),relaxed_tolerance_reached.end());
        if( finish&&relaxed_tolerance_reached_max&&(result_status>0)&&(relaxed_tolerance_reached.size()>0) )
        {
            auto min_value = *std::min_element(norms_storage.begin(),norms_storage.end());
            size_t soluton_num = 0;
            for(int jjj = 0;jjj<relaxed_tolerance_reached.size();jjj++)
            {
                log->warning_f("continuation::convergence: solution %i: norm = %le, flag = %s, relaxed_tol = %le", soluton_num++, norms_storage[jjj], (relaxed_tolerance_reached[jjj]?"true":"false"), tolerance*relax_tolerance_factor  );
            }

            vec_ops->assign(x1_storage, x);
            lambda = lambda1_storage;

            log->warning_f("continuation::convergence: Newton is setting relaxed tolerance = %le,  solution with norm = %le", double(tolerance*relax_tolerance_factor), (double)min_value );
            result_status = 0;     
        }
        //this signals that we couldn't set up the solution with the relaxed tolerance
        else if(finish&&(result_status>0))
        {
            log->error_f("continuation::convergence: newton step failed to finish: relaxed_tolerance_reached.size() = %i, result_status = %i, ||x| = %le, relaxed_tol = %le", relaxed_tolerance_reached.size(), result_status, vec_ops->norm_l2(x), tolerance*relax_tolerance_factor );
                finish = true;
        }

        if(finish)
        {   //checks whaterver is needed for nans, errors or whaterver is considered a quality solution in the nonlinear operator.
            T solution_quality = nonlin_op->check_solution_quality(x);
            log->info_f("continuation::convergence: Newton obtained solution quality = %le.", solution_quality);

        }

        return finish;
    }
    
    unsigned int get_number_of_iterations()
    {
        return iterations;
    }
    void reset_iterations()
    {
        iterations = 0;
        reset_wight();
        norms_evolution.clear();
        norms_storage.clear();
        relaxed_tolerance_reached.clear();
        stagnation = 0;
    }
    void reset_wight()
    {
        newton_wight = newton_wight_initial;
        stagnation = 0;
    }
    std::vector<T>* get_norms_history_handle()
    {
        return &norms_evolution;
    }

private:
    unsigned int stagnation = 0;
    unsigned int stagnation_max = 0;    
    vector_operations* vec_ops;
    logging* log;
    unsigned int maximum_iterations;
    unsigned int iterations;
    T tolerance;
    T tolerance_0;
    T lambda1_storage;
    T_vec x1, x1_storage, Fx;
    T newton_wight, newton_wight_initial;
    bool verbose, store_norms_history;
    std::vector<T> norms_evolution;
    std::vector<bool> relaxed_tolerance_reached;
    std::vector<T> norms_storage;
    T maximum_norm_increase_;
    T newton_wight_threshold_;

    T relax_tolerance_factor;
    int relax_tolerance_steps;
    T d_step;
    int current_relax_step;


    std::string parse_result_status(int result_status)
    {
        switch(result_status)
        {
            case 0:
                return{"converged"};
                break;
            case 1:
                return{"in progress"};
                break;                
            case 2:
                return{"not finite input n"};
                break;
            case 3:
                return{"not finite update n+1"};
                break;   
            case 4:
                return{"too small update wight"};
                break;
            default:
                return{"unknown state!"};
                break;
        }
    }

    std::string parse_bool(bool val)
    {
        return (val?"true":"false");
    }


};


}
}

#endif