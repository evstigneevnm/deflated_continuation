#ifndef __TIME_STEPPER_IMPLICIT_TIME_STEP_H__
#define __TIME_STEPPER_IMPLICIT_TIME_STEP_H__


#include <memory>
#include <time_stepper/system_operator.h>
#include <time_stepper/convergence_strategy.h>
#include <numerical_algos/newton_solvers/newton_solver.h>


namespace time_steppers
{


template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSolver, class Log, class TimeStepAdaptation>
class implicit_time_step
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using table_t = time_steppers::detail::tableu;

    implicit_time_step(VectorOperations* vec_ops_p, TimeStepAdaptation* time_step_adapt_p, Log* log_, NonlinearOperator* nonlin_op_p, LinearOperator* lin_op_p, LinearSolver* lin_solver_p, T param_p = 1.0, const std::string& method_p = "IE"):
    vec_ops_(vec_ops_p), 
    time_step_adapt_(time_step_adapt_p),
    log(log_), 
    nonlin_op_(nonlin_op_p), 
    lin_op_(lin_op_p),
    lin_solver_(lin_solver_p),
    param_(param_p),
    method_(method_p)
    {
        set_table_and_reinit_storage();
        vec_ops_->init_vectors(v1_helper_, f_helper_, v2_helper_, v3_helper_);
        vec_ops_->start_use_vectors(v1_helper_, f_helper_, v2_helper_, v3_helper_);
        numeric_eps_ = std::numeric_limits<T>::epsilon();
        loc_nonlin_op_ = std::make_shared<nonlinear_operator>(vec_ops_, nonlin_op_);
        conv_strat_ = std::make_shared<conv_strategy_t>(vec_ops_, log_);
        sys_op_ = std::make_shared<sys_operator_t>(vec_ops_, lin_op_, lin_solver_);
        newton_ = std::make_shared<newton_t>(vec_ops_, sys_op_.get(), conv_strat_.get());
    }
    ~implicit_time_step()
    {
        vec_ops_->stop_use_vectors(v1_helper_, f_helper_, v2_helper_, v3_helper_);
        vec_ops_->free_vectors(v1_helper_, f_helper_, v2_helper_, v3_helper_);
        clear_storage();
    }
    

private:
    struct nonlinear_operator
    {
        nonlinear_operator(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p):
        vec_ops_(vec_ops_p), nonlin_op_(nonlin_op_p)
        {}

        void set_time(T current_time)
        {
            current_time_ = current_time;
        }

        void ref_to_vector_and_rhs_weight(T_vec& rhs_nonchanged_vec_p, const T dtajj_p)
        {
            rhs_nonchanged_vec_ = rhs_nonchanged_vec_p;
            dtajj_ = dtajj_p;
        }

        void set_linearization_point(const T_vec& u_0_p, const T param_p)
        {
            nonlin_op_->set_linearization_point(u_0_p, param_p);
        }

        void F(const T_vec& u, const T param_p, T_vec& v)const
        {
            // u_n + tau sum(a_jk F(u_k)) + tau F(u_j) - u_j, where u_n + 
            nonlin_op_->F(current_time_, u, param_p, v);
            vec_ops_->add_mul(-1.0, u, 1.0, rhs_nonchanged_vec_, dtajj_, v);
        }

        VectorOperations* vec_ops_;
        NonlinearOperator* nonlin_op_;
        T_vec rhs_nonchanged_vec_;
        T current_time_;
        T dtajj_;
    };

    using conv_strategy_t = nonlinear_operators::newton_method::convergence_strategy<VectorOperations, nonlinear_operator, Log>;
    using sys_operator_t = nonlinear_operators::system_operator<VectorOperations, nonlinear_operator, LinearOperator, LinearSolver>;
    using newton_t = numerical_algos::newton_method::newton_solver<VectorOperations, nonlinear_operator, sys_operator_t, conv_strategy_t>;
    std::shared_ptr<nonlinear_operator> loc_nonlin_op_;
    std::shared_ptr<conv_strategy_t> conv_strat_;
    std::shared_ptr<sys_operator_t> sys_op_;
    std::shared_ptr<newton_t> newton_;


public:

    void scheme(const std::string& method_p)
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
    void force_dt_single_step(const T dt_p)
    {
        // time_step_adapt_->force_dt_single_step()dt_forced_ = dt_p;
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
    void set_initial_time_interval(const std::pair<T,T> time_interval_p)
    {
        time_step_adapt_->set_initial_time_interval(time_interval_p);
    }

    void force_undo_step()
    {
        time_step_adapt_->force_undo_step();
    }

    void reset_steps()
    {
        time_step_adapt_->reset_steps();
    }

    bool execute_forced_dt(const T dt_forced_p, const T_vec& in_p, T_vec& out_p)
    {
        auto t = time_step_adapt_->get_time();
        rk_step(in_p, dt_forced_p, t);
        vec_ops_->assign(v1_helper_, out_p);
        time_step_adapt_->force_set_timestep(dt_forced_p);
        ++(*time_step_adapt_);
        return true; //always accept?
    }


    void set_newton_method(const T tolerance_p, size_t max_iters_p)
    {
        conv_strat_->set_convergence_constants(tolerance_p, max_iters_p);
    }

    bool execute(const T_vec& in_p, T_vec& out_p) 
    {
        bool finish = false;
        pre_execte_step();
        while(true)
        {
            auto dt = time_step_adapt_->get_dt();
            auto t = time_step_adapt_->get_time();
            if( table.get_type() == table_t::type::IRK)
            {
                rk_step_irk(in_p, dt, t);
            }
            else
            {
                rk_step_dirk(in_p, dt, t);
            }

            if(!newton_converged)
            {
                time_step_adapt_->reject_step();
            }
            auto accept_time_step = time_step_adapt_->estimate_timestep(in_p, v1_helper_, f_helper_);

            if(accept_time_step)
            {
                auto res_finish = time_step_adapt_->chech_finished(); //it can either stop because it reached the end of integration time or can stop because the integration process fails.
                if(res_finish.first)  //res_finish.first <-if false: continue (not failed)
                {
                    throw std::runtime_error("implicit_time_step::execute: filed to complete a step.");
                    break;
                }
                else if (res_finish.second)   //res_finish.first <-if false: continue (not finished time reached)
                {
                    finish = true;
                    break;
                }

                if(newton_converged) //update only if timestep converged
                {
                    ++(*time_step_adapt_);
                    vec_ops_->assign(v1_helper_, out_p);
                }
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
    T_vec v1_helper_, v2_helper_, v3_helper_;
    T_vec f_helper_;
    std::vector<T_vec> fk_storage_impl_;
    std::vector<T> tk;

    TimeStepAdaptation* time_step_adapt_;
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;

    LinearOperator* lin_op_;
    LinearSolver* lin_solver_;
    Log* log;
    detail::butcher_tables table_generator;
    table_t table;
    std::string method_;
    size_t n_stages_;
    bool newton_converged;


    void inline rk_step_irk(const T_vec& in_p, const T dt, const T t)
    {
        throw std::logic_error("time_steppers::implicit_time_step: IRK method is not implemented.");
    }
    //
    // system assumed in the form u_t = F(t, u) i.e. F is on the right.
    //
    void inline rk_step_dirk(const T_vec& in_p, const T dt, const T t)
    {
        auto t1 = t;
        
        for(std::size_t j=0;j<n_stages_;j++)
        {
            vec_ops_->assign(in_p, v1_helper_);
            t1 = t + dt*table.get_c<T>(j);
            for(std::size_t k=0;(1+k)<=j;k++)
            {
                auto a_impl_jk = table.get_A<T>(j,k); //L part of the implicit matrix
                vec_ops_->add_mul(dt*a_impl_jk, fk_storage_impl_.at(k), v1_helper_);
            }            
            if(table.get_A<T>(j,j) != 0.0)
            {
                auto current_weight = dt*table.get_A<T>(j,j);
                loc_nonlin_op_->set_time(t1);
                loc_nonlin_op_->ref_to_vector_and_rhs_weight(v1_helper_, current_weight);
                lin_op_->set_aE_plus_bA( {1.0, -current_weight} );
                newton_converged = newton_->solve(loc_nonlin_op_.get(), v1_helper_, param_, v2_helper_); // solves newton part
                if(!newton_converged)
                {
                    log->error("time_steppers::implicit_time_step: Newton solver failed to converge.");
                    break;
                }
            }
            else
            {
                vec_ops_->assign(v1_helper_, v2_helper_);
            }
            nonlin_op_->F(t1, v2_helper_, param_, fk_storage_impl_.at(j) );
        }
        
        vec_ops_->assign(in_p, v1_helper_);
        for(std::size_t j=0;j<n_stages_;j++)
        {
            auto b_impl_j = table.get_b<T>(j);
            vec_ops_->add_mul(dt*b_impl_j, fk_storage_impl_.at(j), v1_helper_);            
        }

        if(table.is_embedded())
        {
            vec_ops_->assign_scalar(static_cast<T>(0.0), f_helper_);
            for(std::size_t j=0;j<n_stages_;j++)
            {
                auto b_err = table.get_err_b<T>(j);
                vec_ops_->add_mul(b_err, fk_storage_impl_.at(j), f_helper_);               
            }
        }  

    }


    void reinit_storage(size_t n_stages_p)
    {
        clear_storage();
        fk_storage_impl_.resize(n_stages_p);
        for(auto &x: fk_storage_impl_)
        {
            vec_ops_->init_vector(x);
            vec_ops_->start_use_vector(x);            
        }            
    }

    void clear_storage()
    {
        for(auto &x: fk_storage_impl_)
        {
            vec_ops_->stop_use_vector(x);
            vec_ops_->free_vector(x);
        }        
        fk_storage_impl_.clear();
    }


    void set_table_and_reinit_storage()
    {
        table = table_generator.set_table_by_name(method_);

        if( !time_step_adapt_->check_globalization() )
        {
            if( time_step_adapt_->is_adaptive() && !table.is_embedded() )
            {
                throw std::logic_error("time_steppers::implicit_time_step: cannot use adaptive timestep selection strategy with a non-embedded Runge-Kutta method.");
            }
        }
        n_stages_ = table.get_size();
        reinit_storage(n_stages_);
    }

};

}


#endif