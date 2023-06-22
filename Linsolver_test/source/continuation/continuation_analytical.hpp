#ifndef __CONTINUATION_ANALYTICAL_HPP__
#define __CONTINUATION_ANALYTICAL_HPP__


#include <continuation/continuation.hpp>

namespace continuation
{


template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperations, class LinearOperator,  class Knots, class LinearSolver, class Newton, class Curve>
class continuation_analytical: public continuation<VectorOperations, VectorFileOperations, Log, NonlinearOperations, LinearOperator,  Knots, LinearSolver, Newton, Curve>
{
private:
    typedef continuation<VectorOperations, VectorFileOperations, Log, NonlinearOperations, LinearOperator,  Knots, LinearSolver, Newton, Curve> parent_t;

    typedef typename parent_t::T T;
    typedef typename parent_t::T_vec T_vec;

//  local variables
    T ds_0;


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
    }

    void continuate_curve(Curve*& curve_, const T_vec& x0_, const T& lambda0_)
    {
        parent_t::update_knots();
        parent_t::bif_diag = curve_;
        parent_t::direction = parent_t::initial_direciton;
        
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
      
    }

private:

    bool interpolate_all_knots()
    {
        bool res = false;
        for(auto &x: *parent_t::knots)
        {

            if( (x - parent_t::lambda1)*(x - parent_t::lambda0)<=T(0.0) )
            {
                parent_t::lambda1 = x;
                parent_t::nonlin_op->exact_solution(parent_t::lambda1, parent_t::x1);
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

        if(!std::isfinite(parent_t::lambda0))
        {
            throw std::runtime_error("continuation_analytical::check_interval: fatal nonfinite value of lambda0 = " + std::to_string(parent_t::lambda0) );
        }
        if(!std::isfinite(parent_t::lambda1))
        {
            throw std::runtime_error("continuation_analytical::check_interval: fatal nonfinite value of lambda1 = " + std::to_string(parent_t::lambda1) );
        }


        if( (parent_t::lambda_min - parent_t::lambda1)*(parent_t::lambda_min - parent_t::lambda0)<=T(0.0) )
        {
            parent_t::lambda1 = parent_t::lambda_min;
            parent_t::nonlin_op->exact_solution(parent_t::lambda1, parent_t::x1);
            intersect_min = true;
        }
        if( (parent_t::lambda_max - parent_t::lambda1)*(parent_t::lambda_max - parent_t::lambda0)<=T(0.0) )
        {
            parent_t::lambda1 = parent_t::lambda_max;
            parent_t::nonlin_op->exact_solution(parent_t::lambda1, parent_t::x1);
            intersect_max = true;
        }

        if( intersect_min || intersect_max )
        {
            parent_t::break_semicurve++;
            parent_t::fail_flag = false;
            parent_t::continue_next_step = false;
        }        
    }

    void start_semicurve()
    {

        parent_t::bif_diag->add(parent_t::lambda0, parent_t::x0, true); //add initial knot, force save data!           
        unsigned int s;
        for(s=0;s<parent_t::max_S;s++)
        {
            T norm_vector = parent_t::vec_ops->norm(parent_t::x0);
            T d_lambda = ds_0/norm_vector*10.0*parent_t::lambda0;     //must put finite difference tangent Jacobian instead!

            parent_t::lambda1 = parent_t::lambda0 + parent_t::direction*d_lambda;
            parent_t::nonlin_op->exact_solution(parent_t::lambda1, parent_t::x1);
            // continuation_step->solve(nonlin_op, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
            // (x0, lambda0)->(x1, lambda1)
            bool did_knot_interpolation = false;
            if((s>1)&&(!parent_t::just_interpolated))
            {
                check_interval();
                //check_returning();
                did_knot_interpolation = interpolate_all_knots();
                //if fail flag after the interpolation, restore (x1, lambda1)?!
            }
            else
            {
                parent_t::just_interpolated = false;
            }
            //if try blocks passes, THIS is executed:
            parent_t::bif_diag->add(parent_t::lambda1, parent_t::x1, did_knot_interpolation);
                    
            parent_t::vec_ops->assign(parent_t::x1, parent_t::x0);
            //parent_t::vec_ops->assign(parent_t::x1_s, parent_t::x0_s);
            parent_t::lambda0 = parent_t::lambda1;
            //lambda0_s = lambda1_s;
            if(!parent_t::continue_next_step)
            { 
                break;
            }
            if(s==parent_t::max_S)
            {
                parent_t::continue_next_step = false;
                parent_t::break_semicurve++;
            }

        }       

    }



};

}

#endif