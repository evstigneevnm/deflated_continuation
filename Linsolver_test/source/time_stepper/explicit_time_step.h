#ifndef __TIME_STEPPER_EXPLICIT_TIME_STEP_H__
#define __TIME_STEPPER_EXPLICIT_TIME_STEP_H__

#include <string>
#include <vector>
#include <stdexcept>
#include <vector>
#include <utility>
#include <limits>
#include <time_stepper/detail/all_methods_enum.h>
#include <time_stepper/detail/butcher_tables.h>



namespace time_steppers
{

// important! 
template<class VectorOperations, class NonlinearOperator, class Log, class TimeStepAdaptation>
class explicit_time_step
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using method_type = detail::methods;
    using table_t = time_steppers::detail::tableu;

    explicit_time_step(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_, TimeStepAdaptation* time_step_adapt_p, Log* log_, T param_p = 1.0,  method_type method_p = method_type::RKDP45):
    vec_ops_(vec_ops_p), 
    nonlin_op_(nonlin_op_), 
    time_step_adapt_(time_step_adapt_p),
    log(log_), 
    param_(param_p),
    method_(method_p)
    {
        set_table_and_reinit_storage();
        vec_ops_->init_vectors(v1_helper_, f_helper_); vec_ops_->start_use_vectors(v1_helper_, f_helper_);
        numeric_eps_ = std::numeric_limits<T>::epsilon();

    }
    ~explicit_time_step()
    {
        vec_ops_->stop_use_vectors(v1_helper_, f_helper_); vec_ops_->free_vectors(v1_helper_, f_helper_);
        clear_storage();
    }
    
    void scheme(method_type method_p)
    {
        if(method_ != method_p)
        {
            method_ = method_p;
            set_table_and_reinit_storage();
        }
    }

    void set_parameter(const T param_p)     
    {
        param_ = param_p;
    }

    T get_dt()const
    {
        return (time_step_adapt_->get_dt());
    }
    T get_time()const
    {
        return (time_step_adapt_->get_time());
    }
    void pre_execte_step()
    {
        time_step_adapt_->pre_execte_step();
    }

    auto get_iteration()const
    {
        return (time_step_adapt_->get_iteration());
    }

    void init_steps(const T_vec& in_p)
    {
        nonlin_op_->F( get_time(), in_p, param_, f_helper_ );
        time_step_adapt_->init_steps(in_p, f_helper_);
    }
    bool check_reject_step()const
    {
        return time_step_adapt_->check_reject_step();
    }

    bool execute(const T_vec in_p, T_vec out_p) 
    {
        bool finish = false;

        while(true)
        {
            auto dt = time_step_adapt_->get_dt();
            auto t = time_step_adapt_->get_time();
            auto t1 = t;
            nonlin_op_->F(t, in_p, param_, fk_storage_.at(0) );


            for(size_t j=0;j<n_stages_;j++)
            {
                vec_ops_->assign(in_p, v1_helper_);
                if( !table.is_autonomous() )
                {
                    t1+=dt*table.get_c<T>(j);
                }
                
                for(size_t k=0;(1+k)<=j;k++)
                {
                    auto a_jk = table.get_A<T>(j,k);
                    vec_ops_->add_mul(dt*a_jk, fk_storage_.at(k), v1_helper_);
                }
                
                nonlin_op_->F(t1, v1_helper_, param_, fk_storage_.at(j) );
            }
            
            vec_ops_->assign(in_p, v1_helper_);
            for(size_t j=0;j<n_stages_;j++)
            {
                auto b_j = table.get_b<T>(j);
                vec_ops_->add_mul(dt*b_j, fk_storage_.at(j), v1_helper_);
            }

            if(table.is_embedded())
            {
                vec_ops_->assign_scalar(static_cast<T>(0.0), f_helper_);
                for(size_t j=0;j<n_stages_;j++)
                {
                    auto b_err = table.get_err_b<T>(j);
                    vec_ops_->add_mul(b_err, fk_storage_.at(j), f_helper_);
                }
            }

            auto accept_time_step = time_step_adapt_->estimate_timestep(in_p, v1_helper_, f_helper_);

            if(accept_time_step)
            {
                auto res_finish = time_step_adapt_->chech_finished(); //it can either stop because it reached the end of integration time or can stop because the integration process fails.
                if(res_finish.first)  //res_finish.first <-if false: continue (not failed)
                {
                    throw std::runtime_error("explicit_time_step::execute: filed to complete a step.");
                    break;
                }
                else if (res_finish.second)   //res_finish.first <-if false: continue (not finished time reached)
                {
                    finish = true;
                    break;
                }


                ++(*time_step_adapt_);
                vec_ops_->assign(v1_helper_, out_p);
                break;
            }
            

        }
        return(finish);
    }


private:
    T numeric_eps_;
    T f_sign = T(1.0);
    T time_step;  
    T param_;
    T_vec v1_helper_ = nullptr;
    T_vec f_helper_ = nullptr;
    std::vector<T_vec> fk_storage_;
    std::vector<T> tk;

    TimeStepAdaptation* time_step_adapt_;
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;
    Log* log;
    detail::butcher_tables table_generator;
    table_t table;
    method_type method_;
    size_t n_stages_;


    void reinit_storage(size_t n_stages_p)
    {
        clear_storage();
        fk_storage_.resize(n_stages_p);
        for(auto &x: fk_storage_)
        {
            vec_ops_->init_vector(x);
            vec_ops_->start_use_vector(x);            
        }        
    }

    void clear_storage()
    {
        for(auto &x: fk_storage_)
        {
            vec_ops_->stop_use_vector(x);
            vec_ops_->free_vector(x);
            // std::cout << "deleted T_vec" << std::endl;
        }
        fk_storage_.clear();
    }


    void set_table_and_reinit_storage()
    {
        table = table_generator.set_table(method_);
        n_stages_ = table.get_size();
        reinit_storage(n_stages_);
    }

};



}



#endif