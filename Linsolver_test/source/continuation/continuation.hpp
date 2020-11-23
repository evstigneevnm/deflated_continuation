#ifndef __CONTINUATION_HPP__
#define __CONTINUATION_HPP__

/**
*    The main part of the deflation-continuation process.
*
*    Continuation class that utilizes single step advance to continue the solution 
*    until it returns back or reaches two boundaries of min and max knots values.
*
*/

#include <string>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>

#include <continuation/predictor_adaptive.h>
#include <continuation/system_operator_continuation.h>
#include <continuation/advance_solution.h>
#include <continuation/initial_tangent.h>
#include <continuation/convergence_strategy.h>



namespace continuation
{

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperations, class LinearOperator,  class Knots, class LinearSolver, class Newton, class Curve>
class continuation
{
protected:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

private:
    typedef std::pair<bool, bool> bools2;


    typedef system_operator_continuation<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        LinearSolver,
        Log
        > system_operator_cont_t;

    typedef newton_method_extended::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_cont_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        VectorOperations, 
        NonlinearOperations,
        system_operator_cont_t, 
        convergence_newton_cont_t, 
        T /* point solution class here instead of real!*/ 
        > newton_cont_t;

    typedef predictor_adaptive<
        VectorOperations,
        Log
        > predictor_cont_t;

    typedef advance_solution<
        VectorOperations,
        Log,
        newton_cont_t,
        Newton,
        NonlinearOperations,
        system_operator_cont_t,
        predictor_cont_t,
        convergence_newton_cont_t
        >advance_step_cont_t;

    typedef initial_tangent<
        VectorOperations,
        Log,
        Newton,
        NonlinearOperations, 
        LinearOperator,
        LinearSolver
        > tangent_0_cont_t;




public:
    continuation(VectorOperations*& vec_ops_, VectorFileOperations*& file_ops_, Log*& log_, NonlinearOperations*& nonlin_op_, LinearOperator*& lin_op_, Knots*& knots_, LinearSolver*& SM_, Newton*& newton_):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    knots(knots_),
    SM(SM_),
    newton(newton_),
    lin_op(lin_op_)
    {
        predict = new predictor_cont_t(vec_ops, log);
        system_operator_cont = new system_operator_cont_t(vec_ops, log, lin_op, SM);
        conv_newton_cont = new convergence_newton_cont_t(vec_ops, log);
        newton_cont = new newton_cont_t(vec_ops, system_operator_cont, conv_newton_cont);
        continuation_step = new advance_step_cont_t(vec_ops, log, system_operator_cont, newton_cont, newton, predict, conv_newton_cont);
        init_tangent = new tangent_0_cont_t(vec_ops, log, newton, lin_op, SM);


        max_S = 100;
        set_all_vectors();
    }
    ~continuation()
    {
        unset_all_vectors();
        delete init_tangent;
        delete continuation_step;
        delete newton_cont;
        delete conv_newton_cont;
        delete system_operator_cont;
        delete predict;

    }

    void set_steps(unsigned int max_S_, T ds_0_, int initial_direciton_ = -1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        max_S = max_S_;
        initial_direciton = initial_direciton_;
        predict->set_steps(ds_0_, step_ds_m_, step_ds_p_, attempts_0_);
        
    }

    void set_newton(T tolerance_, unsigned int maximum_iterations_, T relax_tolerance_factor_, int relax_tolerance_steps_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true)
    {
        conv_newton_cont->set_convergence_constants(tolerance_, maximum_iterations_, relax_tolerance_factor_, relax_tolerance_steps_, newton_wight_, store_norms_history_,  verbose_);
        epsilon = T(5.0)*tolerance_; //tolerance to check distance between vectors in curves.
    }


    void update_knots()
    {
        lambda_min = knots->get_min_value();
        lambda_max = knots->get_max_value();  
        
    }

    void continuate_curve(Curve*& curve_, const T_vec& x0_, const T& lambda0_)
    {
        update_knots();
        bif_diag = curve_;
        direction = initial_direciton;
        
        //make a copy here? or just use the provided reference
        //x0 = x0_, lambda0 = lambda0_;
        vec_ops->assign(x0_, x0);
        lambda0 = lambda0_;
        lambda_start = lambda0_;
        //let's use a copy for start values since we need those to check returning value anyway
        
        vec_ops->assign(x0_, x_start);
        
        break_semicurve = 0;

        while (break_semicurve < 2)
        {
            continue_next_step = true;
            start_semicurve();
            change_direction(); //if we reached the origin, then this is irrelevant. Else, change direction and do it again
            vec_ops->assign(x_start, x0);
            lambda0 = lambda_start;
            if(fail_flag)
            {
                log->info_f("continuation::continuate_curve: previous semicurve returned with fail flag.");
                fail_flag = false;
            }
        }
        bif_diag->print_curve();
    }


protected: //changed to protected for inheritance
    //passed:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperations* nonlin_op;
    Knots* knots;
    LinearSolver* SM;
    Newton* newton;
    LinearOperator* lin_op;
    
    //created localy:
    predictor_cont_t* predict;
    system_operator_cont_t* system_operator_cont;
    convergence_newton_cont_t *conv_newton_cont;
    newton_cont_t* newton_cont;
    advance_step_cont_t* continuation_step;
    tangent_0_cont_t* init_tangent;
    Curve* bif_diag;

    int direction = -1;
    int initial_direciton = 1;
    unsigned int max_S;
    T epsilon = T(1.0e-6);


    void change_direction()
    {
        direction *= -1;
    }

    //vectors and points for continuation

    T lambda_start; T_vec x_start;
    T lambda0, lambda0_s, lambda1, lambda1_s;
    T lambda_min, lambda_max;
    T_vec x0, x0_s, x1, x1_back, x1_s, x_check;
    char break_semicurve = 0;
    bool fail_flag = false;
    bool continue_next_step = true;
    bool just_interpolated = false;
private:
    void set_all_vectors()
    {
        
        vec_ops->init_vector(x_check); vec_ops->start_use_vector(x_check);
        vec_ops->init_vector(x_start); vec_ops->start_use_vector(x_start);
        vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        vec_ops->init_vector(x0_s); vec_ops->start_use_vector(x0_s);
        vec_ops->init_vector(x1_s); vec_ops->start_use_vector(x1_s);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(x1_back); vec_ops->start_use_vector(x1_back);
    }
    void unset_all_vectors()
    {
        vec_ops->stop_use_vector(x_check); vec_ops->free_vector(x_check);
        vec_ops->stop_use_vector(x_start); vec_ops->free_vector(x_start);
        vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        vec_ops->stop_use_vector(x0_s); vec_ops->free_vector(x0_s);
        vec_ops->stop_use_vector(x1_s); vec_ops->free_vector(x1_s);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(x1_back); vec_ops->free_vector(x1_back);
    }


    bool get_solution(const T& lambda_fix, T_vec& x_)
    {
        bool converged;
        converged = newton->solve(nonlin_op, x_, lambda_fix);
        return(converged);
    }
    

    bool interpolate_solutions(const T& lambda_star, const T& lambda_0_, const T_vec& x0_,  T& lambda_1_, T_vec& x1_)
    {
        T w = (lambda_star - lambda_0_)/(lambda_1_ - lambda_0_);
        T _w = T(1) - w;
        vec_ops->add_mul(w, x0_, _w, x1_);
        lambda_1_ = lambda_star;
        bool res = get_solution(lambda_star, x1_);
        if(!res)
        {
            log->error("continuation::interpolate_solutions: newton solver for interpolation failed to converge.");
        }
        return(res);
    }

    bool check_vector_distances()
    {

        vec_ops->assign_mul(T(1), x_start, T(-1), x1, x_check);
        T norm_distance = vec_ops->norm_l2(x_check);
        if(norm_distance < epsilon)
            return(true);
        else
            return(false);
    }

    bools2 check_intersection(T lambda_star) //check current interseciton with the parameter value lambda_star
    {
        if( (lambda_star - lambda1)*(lambda_star - lambda0)<=T(0.0) )
        {
            

            bool ret = interpolate_solutions(lambda_star, lambda0, x0, lambda1, x1);
            if(!ret)
            {
                fail_flag = true;
                return(bools2(true, true));
            }
            bool vectors_coincide = check_vector_distances();

            return(bools2(true, vectors_coincide));
        }
        else
        {
            return(bools2(false, false));
        }
    
    }


    void check_returning()
    {
        bools2 returned = check_intersection(lambda_start);

        if(returned.second) //intersecting a starting point
        {
            if(fail_flag)
            {
                break_semicurve++;
                fail_flag = true;
                continue_next_step = false;

            }
            else
            {
                break_semicurve = 2;
                fail_flag = false;
                continue_next_step = false;

            }
        }
    }

    void check_interval()
    {
        bools2 intersect_min = check_intersection(lambda_min);
        bools2 intersect_max = check_intersection(lambda_max);

        if( intersect_min.first || intersect_max.first )
        {
            break_semicurve++;
            fail_flag = false;
            continue_next_step = false;
        }
    }

    bool interpolate_all_knots()
    {
        bool res = false;
        for(auto &x: *knots)
        {

            bools2 res_l = check_intersection(x);
            if(!fail_flag)
            {
                if(res_l.first)
                {
                    just_interpolated = true;
                    res = res_l.first;
                    break;
                }
            }
            else
            {
                res = false;
                break;                
            }
        }
        return res;
    }


    void start_semicurve()
    {
        //assume that x0 and lambda0 are valid solutions, so that ||F(x_0,lambda_0)||<eps
        try
        {
            log->info_f("continuation::start_semicurve: starting semicurve with direction = %i", direction);
            init_tangent->execute(nonlin_op, T(direction), x0, lambda0, x0_s, lambda0_s);
        }
        catch(const std::exception& e)
        {
//            throw std::runtime_error(std::string("continuation::start_semicurve:") + std::string(e.what()) );
            log->error_f("continuation::start_semicurve exception init_tangent: %s\n", e.what());
            break_semicurve++;
            fail_flag = true;
        }
        if(!fail_flag)
        {        
            bif_diag->add(lambda0, x0, true); //add initial knot, force save data!           
            continuation_step->reset(); //resets all data for initial continuation stepping
            unsigned int s;
            for(s=0;s<max_S;s++)
            {
                try
                {
                    continuation_step->solve(nonlin_op, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
                    bool did_knot_interpolation = false;
                    if((s>1)&&(!just_interpolated))
                    {
                        check_interval();
                        check_returning();
                        
                        //save for restoring if interpolation fails!
                        bool fail_flag_b4_interpolation = fail_flag;
                        vec_ops->assign(x1, x1_back);
                        T lambda1_back = lambda1;
                        
                        did_knot_interpolation = interpolate_all_knots();
                        //if fail flag after the interpolation, restore (x1, lambda1) and continue?
                        if((fail_flag)&&(!fail_flag_b4_interpolation))
                        {
                            vec_ops->assign(x1_back, x1);
                            lambda1 = lambda1_back;
                            fail_flag = false;
                            did_knot_interpolation = false;
                            log->warning("continuation::start_semicurve did_knot_interpolation falied, restoring state. May cause problems during deflation!");
                        }
                    }
                    else
                    {
                        just_interpolated = false;
                    }
                    //if try blocks passes, THIS is executed:
                    bif_diag->add(lambda1, x1, did_knot_interpolation);
                    
                    vec_ops->assign(x1, x0);
                    vec_ops->assign(x1_s, x0_s);
                    lambda0 = lambda1;
                    lambda0_s = lambda1_s;
                }
                catch(const std::exception& e)
                {
                    log->error_f("continuation::start_semicurve exception continuation_step: %s\n", e.what());
                    break_semicurve++;
                    fail_flag = true; 
                    continue_next_step = false;                   
                }
                if(!continue_next_step)
                { 
                   break;
                }

            }
            if(s==max_S)
            {
                log->warning_f("continuation::start_semicurve: reached maximum steps = %i", s);
                continue_next_step = false;
                break_semicurve++;
            }

        }       

    }



};

}




#endif // CONTINUATION_HPP