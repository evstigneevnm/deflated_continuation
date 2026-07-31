#ifndef __CONTINUATION_ANALYTICAL_HPP__
#define __CONTINUATION_ANALYTICAL_HPP__

#include <functional>
#include <sstream>
#include <string>

#include <common/scalar_math.h>
#include <continuation/continuation.hpp>

namespace continuation
{

namespace detail
{

template<class T>
std::string scalar_to_string(const T& value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

} // namespace detail

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperations, class LinearOperator,  class Knots, class LinearSolver, class Newton, class Curve, template<class, class, class, class, class> class SystemOperatorContinuation = system_operator_continuation>
class continuation_analytical: public continuation<VectorOperations, VectorFileOperations, Log, NonlinearOperations, LinearOperator,  Knots, LinearSolver, Newton, Curve, SystemOperatorContinuation>
{
private:
    typedef continuation<VectorOperations, VectorFileOperations, Log, NonlinearOperations, LinearOperator,  Knots, LinearSolver, Newton, Curve, SystemOperatorContinuation> parent_t;

    typedef typename parent_t::T T;
    typedef typename parent_t::T_vec T_vec;

//  local variables
    T ds_0 = T(0);
    T ds_max = T(0);
    std::function<bool(const T&, T_vec&)> exact_solution_provider;


public:
    continuation_analytical(VectorOperations*& vec_ops_, VectorFileOperations*& file_ops_, Log*& log_, NonlinearOperations*& nonlin_op_, LinearOperator*& lin_op_, Knots*& knots_, LinearSolver*& SM_, Newton*& newton_):
    parent_t(vec_ops_, file_ops_, log_, nonlin_op_, lin_op_, knots_, SM_, newton_)
    {

    }
    ~continuation_analytical()
    {

    }


    void set_steps(unsigned int max_S_, T ds_0_, T ds_max_, int initial_direciton_ = -1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        parent_t::max_S = max_S_;
        parent_t::initial_direciton = initial_direciton_;
        parent_t::predict->set_steps(ds_0_, ds_max_, step_ds_m_, step_ds_p_, attempts_0_);
        ds_0 = ds_0_;
        ds_max = ds_max_;
    }

    void set_steps(
        const unsigned int max_S_,
        const T ds_0_,
        const T ds_max_,
        const int initial_direciton_,
        const corrector_retry_policy<T>& retry_policy)
    {
        parent_t::max_S = max_S_;
        parent_t::initial_direciton = initial_direciton_;
        parent_t::predict->set_steps(ds_0_, ds_max_, retry_policy);
        ds_0 = ds_0_;
        ds_max = ds_max_;
    }

    void set_exact_solution_provider(std::function<bool(const T&, T_vec&)> provider)
    {
        exact_solution_provider = std::move(provider);
    }

    void clear_exact_solution_provider()
    {
        exact_solution_provider = {};
    }

    bool continuate_curve(Curve*& curve_, const T_vec& x0_, const T& lambda0_)
    {
        parent_t::update_knots();
        parent_t::bif_diag = curve_;
        parent_t::direction = parent_t::initial_direciton;
        parent_t::fail_flag = false;
        parent_t::hard_failure = false;
        parent_t::endpoint_state.reset_curve();
        parent_t::just_interpolated = false;
        parent_t::continue_next_step = true;
        
        //make a copy here? or just use the provided reference
        //x0 = x0_, lambda0 = lambda0_;
        parent_t::vec_ops->assign(x0_, parent_t::x0);
        parent_t::lambda0 = lambda0_;
        parent_t::lambda_start = lambda0_;
        //let's use a copy for start values since we need those to check returning value anyway
       
        parent_t::vec_ops->assign(x0_, parent_t::x_start);
        
        parent_t::break_semicurve = 0;

        while (parent_t::break_semicurve < 2)
        {
            parent_t::continue_next_step = true;
            start_semicurve();
            parent_t::change_direction(); //if we reached the origin, then this is irrelevant. Else, change direction and do it again
            parent_t::vec_ops->assign(parent_t::x_start, parent_t::x0);
            parent_t::lambda0 = parent_t::lambda_start;
            if(parent_t::fail_flag)
            {
                parent_t::log->info_f("continuation_analytical::continuate_curve: previous semicurve returned with fail flag.");
                parent_t::fail_flag = false;
            }
        }
        parent_t::bif_diag->print_curve();
        return !parent_t::hard_failure &&
               !parent_t::endpoint_state.incomplete();
      
    }

private:

    bool interpolate_all_knots()
    {
        bool res = false;
        for(auto &x: *parent_t::knots)
        {
            const T requested_lambda = x;
            T effective_lambda = requested_lambda;
            if(parent_t::knot_resolver)
            {
                parent_t::knot_resolver(requested_lambda, effective_lambda);
            }

            if( (effective_lambda - parent_t::lambda1)*(effective_lambda - parent_t::lambda0)<=T(0.0) )
            {
                parent_t::lambda1 = effective_lambda;
                if(!evaluate_exact_solution(parent_t::lambda1, parent_t::x1))
                {
                    parent_t::continue_next_step = false;
                    parent_t::break_semicurve++;
                    return res;
                }
                parent_t::just_interpolated = true;
                res = true;
            }
        }
        return res;
    }
    void check_interval()
    {
        bool intersect_min = false;
        bool intersect_max = false;

        if(!common::scalar_math::isfinite(parent_t::lambda0))
        {
            throw std::runtime_error("continuation_analytical::check_interval: fatal nonfinite value of lambda0 = " + detail::scalar_to_string(parent_t::lambda0) );
        }
        if(!common::scalar_math::isfinite(parent_t::lambda1))
        {
            throw std::runtime_error("continuation_analytical::check_interval: fatal nonfinite value of lambda1 = " + detail::scalar_to_string(parent_t::lambda1) );
        }


        if( (parent_t::lambda_min - parent_t::lambda1)*(parent_t::lambda_min - parent_t::lambda0)<=T(0.0) )
        {
            parent_t::vec_ops->assign(
                parent_t::x1,
                parent_t::x1_back);
            const T converged_lambda = parent_t::lambda1;
            parent_t::lambda1 = parent_t::lambda_min;
            if(!evaluate_exact_solution(parent_t::lambda1, parent_t::x1))
            {
                parent_t::vec_ops->assign(
                    parent_t::x1_back,
                    parent_t::x1);
                parent_t::lambda1 = converged_lambda;
                if(parent_t::preserve_last_converged_boundary_point)
                {
                    parent_t::set_pending_endpoint_reason(
                        container::curve_endpoint_reason::
                            boundary_min_approximate);
                    intersect_min = true;
                }
                else
                {
                    parent_t::endpoint_state.mark_incomplete();
                    parent_t::set_pending_endpoint_reason(
                        container::curve_endpoint_reason::
                            knot_interpolation_failure);
                    parent_t::hard_failure = true;
                    parent_t::break_semicurve++;
                    parent_t::continue_next_step = false;
                    return;
                }
            }
            else
            {
                parent_t::set_pending_endpoint_reason(
                    container::curve_endpoint_reason::boundary_min);
                intersect_min = true;
            }
        }
        if( (parent_t::lambda_max - parent_t::lambda1)*(parent_t::lambda_max - parent_t::lambda0)<=T(0.0) )
        {
            parent_t::vec_ops->assign(
                parent_t::x1,
                parent_t::x1_back);
            const T converged_lambda = parent_t::lambda1;
            parent_t::lambda1 = parent_t::lambda_max;
            if(!evaluate_exact_solution(parent_t::lambda1, parent_t::x1))
            {
                parent_t::vec_ops->assign(
                    parent_t::x1_back,
                    parent_t::x1);
                parent_t::lambda1 = converged_lambda;
                if(parent_t::preserve_last_converged_boundary_point)
                {
                    parent_t::set_pending_endpoint_reason(
                        container::curve_endpoint_reason::
                            boundary_max_approximate);
                    intersect_max = true;
                }
                else
                {
                    parent_t::endpoint_state.mark_incomplete();
                    parent_t::set_pending_endpoint_reason(
                        container::curve_endpoint_reason::
                            knot_interpolation_failure);
                    parent_t::hard_failure = true;
                    parent_t::break_semicurve++;
                    parent_t::continue_next_step = false;
                    return;
                }
            }
            else
            {
                parent_t::set_pending_endpoint_reason(
                    container::curve_endpoint_reason::boundary_max);
                intersect_max = true;
            }
        }

        if( intersect_min || intersect_max )
        {
            if(intersect_min)
            {
                parent_t::log->warning_f(
                    "continuation_analytical::check_interval: reached lambda_min = %le; stopping semicurve at parameter boundary.",
                    double(parent_t::lambda_min));
            }
            if(intersect_max)
            {
                parent_t::log->warning_f(
                    "continuation_analytical::check_interval: reached lambda_max = %le; stopping semicurve at parameter boundary.",
                    double(parent_t::lambda_max));
            }
            parent_t::break_semicurve++;
            parent_t::fail_flag = false;
            parent_t::continue_next_step = false;
        }        
    }

    void start_semicurve()
    {

        parent_t::add_solution_to_curve(parent_t::lambda0, parent_t::x0, true); //add initial knot, force save data!
        unsigned int s;
        for(s=0;s<parent_t::max_S;s++)
        {
            parent_t::endpoint_state.reset_step();
            T d_lambda = ds_max;
            if(d_lambda <= T(0) || !common::scalar_math::isfinite(d_lambda))
            {
                d_lambda = ds_0;
            }
            if(d_lambda <= T(0) || !common::scalar_math::isfinite(d_lambda))
            {
                d_lambda = T(1);
            }

            parent_t::lambda1 = parent_t::lambda0 + parent_t::direction*d_lambda;
            if(!evaluate_exact_solution(parent_t::lambda1, parent_t::x1))
            {
                parent_t::log->warning_f(
                    "continuation_analytical::start_semicurve: analytical branch is not defined at lambda = %le; stopping semicurve.",
                    double(parent_t::lambda1));
                parent_t::continue_next_step = false;
                parent_t::break_semicurve++;
                break;
            }
            // continuation_step->solve(nonlin_op, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
            // (x0, lambda0)->(x1, lambda1)
            bool did_knot_interpolation = false;
            if(!parent_t::just_interpolated)
            {
                check_interval();
                //check_returning();
                if(parent_t::continue_next_step)
                {
                    did_knot_interpolation = interpolate_all_knots();
                }
                //if fail flag after the interpolation, restore (x1, lambda1)?!
            }
            else
            {
                parent_t::just_interpolated = false;
            }
            //if try blocks passes, THIS is executed:
            parent_t::add_solution_to_curve(
                parent_t::lambda1,
                parent_t::x1,
                did_knot_interpolation,
                parent_t::endpoint_state.pending_reason());
            parent_t::endpoint_state.reset_step();
                    
            parent_t::vec_ops->assign(parent_t::x1, parent_t::x0);
            //parent_t::vec_ops->assign(parent_t::x1_s, parent_t::x0_s);
            parent_t::lambda0 = parent_t::lambda1;
            //lambda0_s = lambda1_s;
            if(!parent_t::continue_next_step)
            { 
                break;
            }
        }       
        if(parent_t::continue_next_step)
        {
            parent_t::log->warning_f(
                "continuation_analytical::start_semicurve: reached maximum analytical sampling steps = %i.",
                parent_t::max_S);
            parent_t::continue_next_step = false;
            parent_t::break_semicurve++;
        }

    }

    bool evaluate_exact_solution(const T& lambda, T_vec& x)
    {
        if(exact_solution_provider)
        {
            return exact_solution_provider(lambda, x);
        }
        parent_t::nonlin_op->exact_solution(lambda, x);
        return true;
    }



};

}

#endif
