#ifndef __TIME_STEPPER_EXTERNAL_DISTANCE_TO_POINTS_H__
#define __TIME_STEPPER_EXTERNAL_DISTANCE_TO_POINTS_H__

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cstdio>

namespace time_steppers{

template<class VectorOperations, class NonlinearOperator>
struct distance_to_points
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;   
    using distance_type = std::vector< std::vector<T> >;

    distance_to_points(VectorOperations* vec_ops, NonlinearOperator* nonlin_op): 
    vec_ops_(vec_ops),
    nonlin_op_(nonlin_op),
    skip_time_(0),
    internal_period_time_counter_(0),
    internal_counter_(0)
    {
        vec_ops_->init_vector(dx); vec_ops_->start_use_vector(dx);
    };
    ~distance_to_points()
    {
        for(auto &x: solutions_)
        {
            vec_ops_->stop_use_vector(x); vec_ops_->free_vector(x);
        }
        vec_ops_->stop_use_vector(dx); vec_ops_->free_vector(dx);
    }
    
    void copy_and_add(const T_vec& vec)
    {

        T_vec x;
        vec_ops_->init_vector(x); vec_ops_->start_use_vector(x);
        vec_ops_->assign(vec, x);
        solutions_.push_back(x);
    }
    distance_type get_all_distances()
    {
        return distance;
    }

    void set_drop_solutions(T skip_time, const std::string& file_name)
    {
        skip_time_ = skip_time;
        solutions_file_name = file_name;
    }   

    bool apply(T simulated_time, T& dt, T_vec& v_in, T_vec& v_out)
    {
        
        std::vector<T> d_l;
        d_l.push_back(simulated_time);
        if(solutions_.size() > 0)
        {
            for(auto& v: solutions_)
            {
                vec_ops_->assign_mul(1.0, v_out, -1.0, v, dx);
                T dts = vec_ops_->norm_l2(dx);
                d_l.push_back(dts);
            }
            auto min_el = std::min_element(d_l.begin()+1, d_l.end());
            d_l.push_back(*min_el);
            distance.push_back(d_l);
        }
        if(skip_time_>0)
        {
            internal_period_time_counter_ += dt;
            if( internal_period_time_counter_>= skip_time_)
            {
                internal_period_time_counter_ = 0.0;
                // std::stringstream ss, ss1;
                // ss.fill(0);
                // ss.width(6);
                // ss << internal_counter_;
                // ss1 << "_" << solutions_file_name;

                char buf[256];
                std::sprintf(buf, "%04lu_%s", internal_counter_, solutions_file_name.c_str());
                std::string fname(buf);
                nonlin_op_->write_solution_abs(fname, v_out);
                std::cout << "solution file " << fname << " written at " << simulated_time << std::endl;
                ++internal_counter_;
            }

        }

        return false; //returns external control of the finish flag
    }

    void save_results(const std::string& f_name)
    {
        std::ofstream f(f_name, std::ofstream::out);
        if (!f) throw std::runtime_error("error while opening file " + f_name);
        for(auto &v: distance)
        {
            for(auto &x: v)
            {
                f << x << " ";
            }
            f << std::endl;
        }
        f.close();
    }


private:
    T skip_time_;
    T internal_period_time_counter_;
    T_vec dx;
    std::vector< T_vec > solutions_;
    std::vector< std::vector<T> > distance;
    std::string solutions_file_name;
    std::size_t internal_counter_;
    VectorOperations *vec_ops_;
    NonlinearOperator *nonlin_op_;


};

}


#endif