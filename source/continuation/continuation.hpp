#ifndef __CONTINUATION_HPP__
#define __CONTINUATION_HPP__

/**
*    The main part of the deflation-continuation process.
*
*    Continuation class that utilizes single step advance to continue the solution 
*    until it returns back or reaches two boundaries of min and max knots values.
*
*/

#include <functional>
#include <string>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>

#include <continuation/predictor_adaptive.h>
#include <continuation/system_operator_continuation.h>
#include <continuation/advance_solution.h>
#include <continuation/initial_tangent.h>
#include <continuation/convergence_strategy.h>
#include <containers/curve_endpoint_reason.h>



namespace continuation
{

namespace detail
{

template<class Curve>
auto start_new_curve_segment_if_available(Curve* curve) -> decltype(curve->start_new_segment(), void())
{
    curve->start_new_segment();
}

inline void start_new_curve_segment_if_available(...)
{
}

}

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperator, class LinearOperator,  class Knots, class LinearSolver, class Newton, class Curve, template<class, class, class, class, class> class SystemOperatorContinuation = system_operator_continuation>
class continuation
{
protected:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef container::curve_endpoint_reason endpoint_reason_t;

private:
    typedef std::pair<bool, bool> bools2;
    typedef std::function<bool(const T& requested_lambda, T& effective_lambda)> knot_resolver_t;
    typedef std::function<bool(
        const T& requested_lambda,
        const T& lambda_left,
        const T_vec& x_left,
        const T& lambda_right,
        const T_vec& x_right,
        T& effective_lambda,
        T_vec& effective_x)> knot_relocator_t;
    typedef std::function<bool(
        const T& lambda_left,
        const T_vec& x_left,
        const T& lambda_right,
        const T_vec& x_right,
        T& hit_lambda,
        T_vec& hit_x,
        std::string& reason)> branch_intersection_checker_t;
    typedef std::function<bool(
        Curve* curve,
        const T& lambda_left,
        const T_vec& x_left,
        const T& lambda_right,
        const T_vec& x_right,
        T& hit_lambda,
        T_vec& hit_x,
        std::string& reason)> self_intersection_checker_t;


    typedef SystemOperatorContinuation<
        VectorOperations, 
        NonlinearOperator,
        LinearOperator,
        LinearSolver,
        Log
        > system_operator_cont_t;

    typedef newton_method_extended::convergence_strategy<
        VectorOperations, 
        NonlinearOperator, 
        Log> convergence_newton_cont_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        VectorOperations, 
        NonlinearOperator,
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
        NonlinearOperator,
        system_operator_cont_t,
        predictor_cont_t,
        convergence_newton_cont_t
        >advance_step_cont_t;

    typedef initial_tangent<
        VectorOperations,
        Log,
        Newton,
        NonlinearOperator, 
        LinearOperator,
        LinearSolver
        > tangent_0_cont_t;




public:
    continuation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, NonlinearOperator* nonlin_op_, LinearOperator* lin_op_, Knots* knots_, LinearSolver* SM_, Newton* newton_):
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

    void set_steps(unsigned int max_S_, T ds_0_, T ds_max_, int initial_direciton_ = -1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        max_S = max_S_;
        initial_direciton = initial_direciton_;
        predict->set_steps(ds_0_, ds_max_, step_ds_m_, step_ds_p_, attempts_0_);
        
    }

    void set_newton(T tolerance_, unsigned int maximum_iterations_, T relax_tolerance_factor_, int relax_tolerance_steps_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true, unsigned int stagnation_max_p = 10, T maximum_norm_increase_p = 0.1, T newton_wight_threshold_p = 1.0e-12)
    {
        conv_newton_cont->set_convergence_constants(tolerance_, maximum_iterations_, relax_tolerance_factor_, relax_tolerance_steps_, newton_wight_, store_norms_history_,  verbose_, stagnation_max_p, maximum_norm_increase_p, newton_wight_threshold_p);
        epsilon = T(100.0)*tolerance_; //tolerance to check distance between vectors in curves.
    }

    void set_solution_postprocessor(std::function<void(T_vec&)> solution_postprocessor_)
    {
        solution_postprocessor = std::move(solution_postprocessor_);
    }

    void set_knot_resolver(knot_resolver_t knot_resolver_)
    {
        knot_resolver = std::move(knot_resolver_);
    }

    void set_knot_relocator(knot_relocator_t knot_relocator_)
    {
        knot_relocator = std::move(knot_relocator_);
    }

    void set_allow_knot_interpolation_failure(const bool allow_)
    {
        allow_knot_interpolation_failure = allow_;
    }

    void set_branch_intersection_checker(branch_intersection_checker_t checker_)
    {
        branch_intersection_checker = std::move(checker_);
    }

    void set_self_intersection_checker(self_intersection_checker_t checker_)
    {
        self_intersection_checker = std::move(checker_);
    }


    void update_knots()
    {
        lambda_min = knots->get_min_value();
        lambda_max = knots->get_max_value();  
        
    }

    bool continuate_curve(Curve*& curve_, const T_vec& x0_, const T& lambda0_)
    {
        update_knots();
        bif_diag = curve_;
        direction = initial_direciton;
        fail_flag = false;
        hard_failure = false;
        incomplete_curve = false;
        pending_endpoint_reason = endpoint_reason_t::none;
        just_interpolated = false;
        continue_next_step = true;
        
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
            detail::start_new_curve_segment_if_available(bif_diag);
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
        return !hard_failure && !incomplete_curve;
    }


protected: //changed to protected for inheritance
    //passed:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperator* nonlin_op;
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
    T epsilon = T(1.0e-5);


    void change_direction()
    {
        direction *= -1;
    }

    //vectors and points for continuation

    T lambda_start; T_vec x_start;
    T lambda0, lambda0_s, lambda1, lambda1_s;
    T lambda_min, lambda_max;
    T_vec x0, x0_s, x1, x1_back, x1_s, x_check, x_output, x_relocated_knot, x_branch_intersection;
    char break_semicurve = 0;
    bool fail_flag = false;
    bool hard_failure = false;
    bool continue_next_step = true;
    bool just_interpolated = false;
    bool incomplete_curve = false;
    endpoint_reason_t pending_endpoint_reason = endpoint_reason_t::none;
    bool allow_knot_interpolation_failure = false;
    bool last_failed_knot_interpolation = false;
    T last_failed_requested_knot = T(0);
    T last_failed_effective_knot = T(0);
    std::function<void(T_vec&)> solution_postprocessor;
    knot_resolver_t knot_resolver;
    knot_relocator_t knot_relocator;
    branch_intersection_checker_t branch_intersection_checker;
    self_intersection_checker_t self_intersection_checker;

    void set_pending_endpoint_reason(endpoint_reason_t reason)
    {
        pending_endpoint_reason = reason;
    }

    void mark_last_curve_point(endpoint_reason_t reason)
    {
        if(bif_diag != nullptr)
        {
            bif_diag->set_last_endpoint_reason(reason);
        }
        if(container::is_incomplete_endpoint(reason))
        {
            incomplete_curve = true;
        }
    }

    void add_solution_to_curve(
        const T& lambda,
        const T_vec& x,
        const bool force_store,
        endpoint_reason_t endpoint_reason = endpoint_reason_t::none)
    {
        const bool should_force_store =
            force_store || container::is_terminal_endpoint(endpoint_reason);
        if(container::is_incomplete_endpoint(endpoint_reason))
        {
            incomplete_curve = true;
        }
        if(solution_postprocessor)
        {
            vec_ops->assign(x, x_output);
            solution_postprocessor(x_output);
            bif_diag->add(lambda, x_output, should_force_store, endpoint_reason);
        }
        else
        {
            bif_diag->add(lambda, x, should_force_store, endpoint_reason);
        }
    }

private:
    void set_all_vectors()
    {
        
        vec_ops->init_vector(x_check); vec_ops->start_use_vector(x_check);
        vec_ops->init_vector(x_output); vec_ops->start_use_vector(x_output);
        vec_ops->init_vector(x_start); vec_ops->start_use_vector(x_start);
        vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        vec_ops->init_vector(x0_s); vec_ops->start_use_vector(x0_s);
        vec_ops->init_vector(x1_s); vec_ops->start_use_vector(x1_s);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(x1_back); vec_ops->start_use_vector(x1_back);
        vec_ops->init_vector(x_relocated_knot); vec_ops->start_use_vector(x_relocated_knot);
        vec_ops->init_vector(x_branch_intersection); vec_ops->start_use_vector(x_branch_intersection);
    }
    void unset_all_vectors()
    {
        vec_ops->stop_use_vector(x_branch_intersection); vec_ops->free_vector(x_branch_intersection);
        vec_ops->stop_use_vector(x_output); vec_ops->free_vector(x_output);
        vec_ops->stop_use_vector(x_check); vec_ops->free_vector(x_check);
        vec_ops->stop_use_vector(x_start); vec_ops->free_vector(x_start);
        vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        vec_ops->stop_use_vector(x0_s); vec_ops->free_vector(x0_s);
        vec_ops->stop_use_vector(x1_s); vec_ops->free_vector(x1_s);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(x1_back); vec_ops->free_vector(x1_back);
        vec_ops->stop_use_vector(x_relocated_knot); vec_ops->free_vector(x_relocated_knot);
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
        vec_ops->add_mul(_w, x0_, w, x1_);
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
        T norm_distance_ref = vec_ops->norm_l2(x_start);
        T norm_distance = vec_ops->norm_l2(x_check);
        T epsilon_l = epsilon*((norm_distance_ref>1.0)?norm_distance_ref:1.0);
        if(norm_distance < epsilon_l)
        {
            log->info_f("continuation::check_intersection::check_vector_distances: vectors coincide with distance = %le, distance reference norm = %le and epsilon = %le", norm_distance, norm_distance_ref, epsilon_l);
            return(true);
        }
        else
        {
            log->warning_f("continuation::check_intersection::check_vector_distances: failed with distance = %le, distance reference norm = %le and epsilon = %le", norm_distance, norm_distance_ref, epsilon_l);
            return(false);
        }
    }

    bools2 check_intersection(
        T lambda_star,
        endpoint_reason_t endpoint_reason = endpoint_reason_t::none) //check current interseciton with the parameter value lambda_star
    {
        if( (lambda_star - lambda1)*(lambda_star - lambda0)<=T(0.0) )
        {
            

            bool ret = interpolate_solutions(lambda_star, lambda0, x0, lambda1, x1);
            if(!ret)
            {
                log->warning_f("continuation::check_intersection::interpolate_solutions: returned failed for lambda_star = %le, lambda_0 = %le, lambda_1 = %le", lambda_star, lambda0, lambda1);
                last_failed_knot_interpolation = true;
                last_failed_requested_knot = lambda_star;
                last_failed_effective_knot = lambda_star;
                fail_flag = true;
                set_pending_endpoint_reason(endpoint_reason_t::knot_interpolation_failure);
                return(bools2(true, true));
            }
            if(endpoint_reason != endpoint_reason_t::none)
            {
                set_pending_endpoint_reason(endpoint_reason);
                if(endpoint_reason == endpoint_reason_t::boundary_min ||
                   endpoint_reason == endpoint_reason_t::boundary_max)
                {
                    return(bools2(true, false));
                }
            }
            bool vectors_coincide = check_vector_distances();

            return(bools2(true, vectors_coincide));
        }
        else if(lambda1>lambda_max) // if we somehow magically sliped out?!
        {
            if(endpoint_reason == endpoint_reason_t::boundary_max)
            {
                log->warning_f("continuation::check_intersection: passed lambda_max = %le without bracketed interpolation, lambda_1 = %le.", lambda_max, lambda1);
                set_pending_endpoint_reason(endpoint_reason_t::boundary_max);
            }
            else
            {
                log->error_f("continuation::check_intersection: slipped out with lambda_max = %le, lambda_1 = %le.", lambda_max, lambda1);
                fail_flag = true;
                set_pending_endpoint_reason(endpoint_reason_t::hard_failure);
            }
            return(bools2(true, true));            
        }
        else if(lambda1<lambda_min) // if we somehow magically sliped out?!
        {
            if(endpoint_reason == endpoint_reason_t::boundary_min)
            {
                log->warning_f("continuation::check_intersection: passed lambda_min = %le without bracketed interpolation, lambda_1 = %le.", lambda_min, lambda1);
                set_pending_endpoint_reason(endpoint_reason_t::boundary_min);
            }
            else
            {
                log->error_f("continuation::check_intersection: slipped out with lambda_min = %le, lambda_1 = %le.", lambda_min, lambda1);
                fail_flag = true;
                set_pending_endpoint_reason(endpoint_reason_t::hard_failure);
            }
            return(bools2(true, true));
        }
        else
        {
            return(bools2(false, false));
        }
    
    }



    void check_returning()
    {
        bools2 returned = check_intersection(lambda_start);
        std::string bool1_l = (returned.first?"true":"false");
        std::string bool2_l = (returned.second?"true":"false");

        log->info_f("continuation::check_returning: returned (%s,%s)", bool1_l.c_str(), bool2_l.c_str() );
        
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
                set_pending_endpoint_reason(endpoint_reason_t::closed_return);
                continue_next_step = false;

            }
        }
    }

    void check_interval()
    {
        bools2 intersect_min(false, false);
        bools2 intersect_max(false, false);

        if(lambda1 < lambda_min)
        {
            intersect_min = check_intersection(lambda_min, endpoint_reason_t::boundary_min);
        }
        else if(lambda1 > lambda_max)
        {
            intersect_max = check_intersection(lambda_max, endpoint_reason_t::boundary_max);
        }
        else
        {
            intersect_min = check_intersection(lambda_min, endpoint_reason_t::boundary_min);
            if(!intersect_min.first)
            {
                intersect_max = check_intersection(lambda_max, endpoint_reason_t::boundary_max);
            }
        }

        if( intersect_min.first || intersect_max.first )
        {
            if(intersect_min.first)
            {
                log->warning_f(
                    "continuation::check_interval: reached lambda_min = %le; stopping semicurve at parameter boundary.",
                    double(lambda_min));
            }
            if(intersect_max.first)
            {
                log->warning_f(
                    "continuation::check_interval: reached lambda_max = %le; stopping semicurve at parameter boundary.",
                    double(lambda_max));
            }
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
            const T requested_lambda = x;
            T effective_lambda = requested_lambda;
            if(knot_resolver)
            {
                knot_resolver(requested_lambda, effective_lambda);
            }

            bools2 res_l = check_intersection(effective_lambda);
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
                if(last_failed_knot_interpolation)
                {
                    last_failed_requested_knot = requested_lambda;
                    last_failed_effective_knot = effective_lambda;
                }
                res = false;
                break;                
            }
        }
        return res;
    }

    bool try_relocate_failed_knot(const T& lambda1_original)
    {
        if(!last_failed_knot_interpolation || !knot_relocator)
        {
            return false;
        }

        T effective_lambda = last_failed_requested_knot;
        const bool relocated = knot_relocator(
            last_failed_requested_knot,
            lambda0,
            x0,
            lambda1_original,
            x1_back,
            effective_lambda,
            x_relocated_knot);
        if(!relocated)
        {
            return false;
        }

        vec_ops->assign(x_relocated_knot, x1);
        lambda1 = effective_lambda;
        fail_flag = false;
        just_interpolated = true;
        last_failed_knot_interpolation = false;
        log->warning_f(
            "continuation::start_semicurve: shifted failed active knot interpolation from requested lambda = %le, effective lambda = %le to validated lambda = %le.",
            double(last_failed_requested_knot),
            double(last_failed_effective_knot),
            double(lambda1));
        return true;
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
            hard_failure = true;
            incomplete_curve = true;
        }
        if(!fail_flag)
        {        
            add_solution_to_curve(lambda0, x0, true); //add initial knot, force save data!
            continuation_step->reset(); //resets all data for initial continuation stepping
            unsigned int s;
            for(s=0;s<max_S;s++)
            {
                pending_endpoint_reason = endpoint_reason_t::none;
                try
                {
                    continuation_step->solve(nonlin_op, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
                    bool did_knot_interpolation = false;
                    if((s>1)&&(!just_interpolated))
                    {
                        check_interval();
                        if(continue_next_step)
                        {
                            check_returning();
                        }
                        if(continue_next_step)
                        {
                            //save for restoring if interpolation fails!
                            bool fail_flag_b4_interpolation = fail_flag;
                            vec_ops->assign(x1, x1_back);
                            T lambda1_back = lambda1;

                            last_failed_knot_interpolation = false;
                            did_knot_interpolation = interpolate_all_knots();
                            //if fail flag after the interpolation, restore (x1, lambda1) and continue?
                            if((fail_flag)&&(!fail_flag_b4_interpolation))
                            {
                                vec_ops->assign(x1_back, x1);
                                lambda1 = lambda1_back;
                                did_knot_interpolation = false;
                                if(try_relocate_failed_knot(lambda1_back))
                                {
                                    did_knot_interpolation = true;
                                }
                                else if(allow_knot_interpolation_failure)
                                {
                                    fail_flag = false;
                                    last_failed_knot_interpolation = false;
                                    log->warning("continuation::start_semicurve did_knot_interpolation failed, restoring state and continuing because policy allows it. May cause problems during deflation!");
                                }
                                else
                                {
                                    log->warning("continuation::start_semicurve did_knot_interpolation failed, restoring state and stopping this curve.");
                                    continue_next_step = false;
                                    break_semicurve++;
                                    hard_failure = true;
                                    incomplete_curve = true;
                                    mark_last_curve_point(endpoint_reason_t::knot_interpolation_failure);
                                    break;
                                }
                            }
                        }
                    }
                    else
                    {
                        just_interpolated = false;
                    }
                    bool branch_intersection_found = false;
                    if(continue_next_step && branch_intersection_checker)
                    {
                        T hit_lambda = lambda1;
                        std::string hit_reason;
                        branch_intersection_found = branch_intersection_checker(
                            lambda0,
                            x0,
                            lambda1,
                            x1,
                            hit_lambda,
                            x_branch_intersection,
                            hit_reason);
                        if(branch_intersection_found)
                        {
                            lambda1 = hit_lambda;
                            vec_ops->assign(x_branch_intersection, x1);
                            did_knot_interpolation = true;
                            continue_next_step = false;
                            break_semicurve++;
                            if(hit_reason == "analytical branch endpoint")
                            {
                                set_pending_endpoint_reason(endpoint_reason_t::analytical_branch);
                            }
                            else
                            {
                                set_pending_endpoint_reason(endpoint_reason_t::known_branch);
                            }
                            log->warning_f(
                                "continuation::start_semicurve: stopped semicurve at lambda = %le due to %s.",
                                double(lambda1),
                                hit_reason.empty() ? "known branch intersection" : hit_reason.c_str());
                        }
                    }
                    bool self_intersection_found = false;
                    if(continue_next_step && self_intersection_checker)
                    {
                        T hit_lambda = lambda1;
                        std::string hit_reason;
                        self_intersection_found = self_intersection_checker(
                            bif_diag,
                            lambda0,
                            x0,
                            lambda1,
                            x1,
                            hit_lambda,
                            x_branch_intersection,
                            hit_reason);
                        if(self_intersection_found)
                        {
                            lambda1 = hit_lambda;
                            vec_ops->assign(x_branch_intersection, x1);
                            did_knot_interpolation = true;
                            continue_next_step = false;
                            break_semicurve++;
                            set_pending_endpoint_reason(endpoint_reason_t::self_intersection);
                            log->warning_f(
                                "continuation::start_semicurve: stopped semicurve at lambda = %le due to %s.",
                                double(lambda1),
                                hit_reason.empty() ? "curve-local self intersection" : hit_reason.c_str());
                        }
                    }
                    //if try blocks passes, THIS is executed:
                    add_solution_to_curve(lambda1, x1, did_knot_interpolation, pending_endpoint_reason);
                    pending_endpoint_reason = endpoint_reason_t::none;
                    
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
                    hard_failure = true;
                    incomplete_curve = true;
                    mark_last_curve_point(endpoint_reason_t::hard_failure);
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
                incomplete_curve = true;
                mark_last_curve_point(endpoint_reason_t::max_steps);
            }

        }       

    }



};

}




#endif // CONTINUATION_HPP
