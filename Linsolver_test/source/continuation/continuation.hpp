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
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#include <numerical_algos/newton_solvers/newton_solver.h>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>



#include <continuation/predictor_adaptive.h>
#include <continuation/system_operator_continuation.h>
#include <continuation/advance_solution.h>
#include <continuation/initial_tangent.h>
#include <continuation/convergence_strategy.h>

#include <containers/bifurcation_diagram_curve.h>



namespace continuation
{

template<class VectorOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, class Knots, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator>
class continuation
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef Monitor monitor_t;

private:
    typedef std::pair<bool, bool> bools2;

    typedef numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        LinearOperator,
        Preconditioner,
        VectorOperations,
        monitor_t,
        Log,
        LinearSolver> sherman_morrison_linear_system_solve_t;

    typedef SystemOperator<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t
        > system_operator_t;
    
    typedef nonlinear_operators::newton_method::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_t;

    typedef numerical_algos::newton_method::newton_solver<
        VectorOperations, 
        NonlinearOperations,
        system_operator_t, 
        convergence_newton_t
        > newton_t;

    typedef system_operator_continuation<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t
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
        NonlinearOperations,
        system_operator_cont_t,
        predictor_cont_t
        >advance_step_cont_t;

    typedef initial_tangent<
        VectorOperations,
        NonlinearOperations, 
        LinearOperator,
        sherman_morrison_linear_system_solve_t
        > tangent_0_cont_t;

    typedef container::bifurcation_diagram_curve<
        VectorOperations,
        VectorFileOperations, 
        Log,
        NonlinearOperations,
        newton_t, std::vector<T_vec>> bif_diag_t;


public:
    continuation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, NonlinearOperations* nonlin_op_, Knots* knots_):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    knots(knots_)
    {
        lin_op = new LinearOperator(nonlin_op);
        precond = new Preconditioner(nonlin_op);
        
        SM = new sherman_morrison_linear_system_solve_t(precond, vec_ops, log); //to be inserted into upper level class

        conv_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, SM);
        newton = new newton_t(vec_ops, system_operator, conv_newton);

        predict = new predictor_cont_t(vec_ops, log);

        system_operator_cont = new system_operator_cont_t(vec_ops, lin_op, SM);
        conv_newton_cont = new convergence_newton_cont_t(vec_ops, log);
        newton_cont = new newton_cont_t(vec_ops, system_operator_cont, conv_newton_cont);
        continuation_step = new advance_step_cont_t(vec_ops, log, system_operator_cont, newton_cont, predict);
        init_tangent = new tangent_0_cont_t(vec_ops, lin_op, SM);
        
        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton);
        max_S = 100;
        set_all_vectors();
        update_knots();
    }
    ~continuation()
    {
        
        unset_all_vectors();
        delete bif_diag;
        delete init_tangent;
        delete continuation_step;
        delete newton_cont;
        delete conv_newton_cont;
        delete system_operator_cont;
        delete predict;
        delete newton;
        delete conv_newton;
        delete SM;
        delete precond;
        delete lin_op;

    }

    void set_steps(unsigned int max_S_, T ds_0_, int initial_direciton_ = -1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        max_S = max_S_;
        initial_direciton = initial_direciton_;
        predict->set_steps(ds_0_, step_ds_m_, step_ds_p_, attempts_0_);
        
    }

    void set_newton(T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true)
    {
        conv_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_,  verbose_);
        conv_newton_cont->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_,  verbose_);
        epsilon = T(5)*tolerance_; //tolerance to check distance between vectors in curves.
    }

    //to be inserted into upper level class
    void set_linsolver(T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = -1, int resid_recalc_freq = -1, int basis_sz = -1)
    {
        //setup linear system:
        mon_orig = &SM->get_linsolver_handle_original()->monitor();
        mon_orig->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon_orig->set_save_convergence_history(true);
        mon_orig->set_divide_out_norms_by_rel_base(true);
        mon_orig->out_min_resid_norm();
        // if(use_precond_resid >= 0)
        //     SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
        // // if(resid_recalc_freq >= 0)
        //     SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
        // if(basis_sz >= 0)
        //     SM->get_linsolver_handle_original()->set_basis_size(basis_sz);  
    }//to be inserted into upper level class
    void set_extended_linsolver(T lin_solver_tol, unsigned int lin_solver_max_it, bool is_small_alpha = false, int use_precond_resid = -1, int resid_recalc_freq = -1, int basis_sz = -1)
    {
        mon = &SM->get_linsolver_handle()->monitor();
        mon->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon->set_save_convergence_history(true);
        mon->set_divide_out_norms_by_rel_base(true);
        mon->out_min_resid_norm();
        // if(use_precond_resid >= 0)
        //     SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
        // if(resid_recalc_freq >= 0)
        //     SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
        // if(basis_sz >= 0)
        //     SM->get_linsolver_handle_original()->set_basis_size(basis_sz); 

        SM->is_small_alpha(is_small_alpha);        
    }

    void update_knots()
    {
        lambda_min = knots->get_min_value();
        lambda_max = knots->get_max_value();  
        
    }

    void continuate_curve(const T_vec& x0_, const T& lambda0_)
    {
        direction = initial_direciton;
        
        //make a copy here? or just use the provided reference
        x0 = x0_, lambda0 = lambda0_;
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
        }
        bif_diag->print_curve();
    }


private:
    //passed:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperations* nonlin_op;
    Knots* knots;

    //created localy:
    LinearOperator* lin_op;
    Preconditioner* precond;
    monitor_t *mon;
    monitor_t *mon_orig; 
    sherman_morrison_linear_system_solve_t *SM;
    convergence_newton_t *conv_newton;
    system_operator_t *system_operator;
    newton_t *newton;
    predictor_cont_t* predict;
    system_operator_cont_t* system_operator_cont;
    convergence_newton_cont_t *conv_newton_cont;
    newton_cont_t* newton_cont;
    advance_step_cont_t* continuation_step;
    tangent_0_cont_t* init_tangent;
    bif_diag_t* bif_diag;

    int direction = -1;
    int initial_direciton = 1;
    unsigned int max_S;
    T epsilon = T(1.0e-6);

    void change_direction()
    {
        if(direction==-1)
            direction = 1;
        else
            direction = -1;

    }

    //vactors and points for continuation

    T lambda_start; T_vec x_start;
    T lambda0, lambda0_s, lambda1, lambda1_s;
    T lambda_min, lambda_max;
    T_vec x0, x0_s, x1, x1_s, x_check;
    char break_semicurve = 0;
    bool fail_flag = false;
    bool continue_next_step = true;

    void set_all_vectors()
    {
        
        vec_ops->init_vector(x_check); vec_ops->start_use_vector(x_check);
        vec_ops->init_vector(x_start); vec_ops->start_use_vector(x_start);
        vec_ops->init_vector(x0_s); vec_ops->start_use_vector(x0_s);
        vec_ops->init_vector(x1_s); vec_ops->start_use_vector(x1_s);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
    }
    void unset_all_vectors()
    {
        vec_ops->stop_use_vector(x_check); vec_ops->free_vector(x_check);
        vec_ops->stop_use_vector(x_start); vec_ops->free_vector(x_start);
        vec_ops->stop_use_vector(x0_s); vec_ops->free_vector(x0_s);
        vec_ops->stop_use_vector(x1_s); vec_ops->free_vector(x1_s);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
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

    bools2 check_intersection(T lambda_star) //check interseciton with the parameter value lambda_star
    {
        if( (lambda_star - lambda1)*(lambda_star - lambda0)<T(0.0) )
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

    void start_semicurve()
    {
        //assume that x0 and lambda0 are valid solutions, so that ||F(x_0,lambda_0)||<eps
        try
        {
            init_tangent->execute(nonlin_op, T(direction), x0, lambda0, x0_s, lambda0_s);
        }
        catch(const std::exception& e)
        {
            log->info_f("%s\n", e.what());
            break_semicurve++;
            fail_flag = true;
        }
        if(!fail_flag)
        {        
            bif_diag->add(lambda0, x0); //add initial knot            
            unsigned int s;
            for(s=0;s<max_S;s++)
            {
                try
                {
                    continuation_step->solve(nonlin_op, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
                    check_returning();
                    check_interval();
                    //if try blocks passes, THIS is executed:
                    bif_diag->add(lambda1, x1);
                    vec_ops->assign(x1, x0);
                    vec_ops->assign(x1_s, x0_s);
                    lambda0 = lambda1;
                    lambda0_s = lambda1_s;
                }
                catch(const std::exception& e)
                {
                    log->info_f("%s\n", e.what());
                    break_semicurve++;
                    fail_flag = true; 
                    continue_next_step = false;                   
                }
                if(!continue_next_step)
                { 
                   break;
                }

            }
            if( (s)==max_S)
            {
                continue_next_step = false;
                break_semicurve++;
            }

        }       

    }



};

}




#endif // CONTINUATION_HPP