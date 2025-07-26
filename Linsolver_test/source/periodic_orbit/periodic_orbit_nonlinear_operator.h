#ifndef __PERIODOC_ORBIT_PERIODIC_ORBIT_NONLINEAR_OPERATOR_H__
#define __PERIODOC_ORBIT_PERIODIC_ORBIT_NONLINEAR_OPERATOR_H__

#include <utility>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <time_stepper/detail/all_methods_enum.h>
#include <periodic_orbit/hyperplane.h>
#include <periodic_orbit/poincare_map_operator.h>
#include <periodic_orbit/glued_poincare_map_linear_operator.h>

/**
*   main periodic orbit operator that executes main routines to find periodic orbits:
*   - find_orbit() - finds perodic orbit by using provided initial value.
*/

namespace periodic_orbit
{


template<class VectorOperations, class NonlinearOperator, class Log, template<class, class, class>class TimeStepAdaptation, template<class, class, class, class> class SingleStepper>
class periodic_orbit_nonlinear_operator
{

    using hyperplane_t = hyperplane<VectorOperations, NonlinearOperator>;
    //TODO: put an alpha linearization here:
    using glued_poincare_map_linear_op_t = periodic_orbit::glued_poincare_map_linear_operator<VectorOperations, NonlinearOperator,  TimeStepAdaptation, SingleStepper,  hyperplane_t, Log>;

    using poincare_map_operator_t = periodic_orbit::poincare_map_operator<VectorOperations, NonlinearOperator, TimeStepAdaptation, SingleStepper, hyperplane_t, Log>;    

    glued_poincare_map_linear_op_t* poincare_map_x_;
    poincare_map_operator_t* poincare_map_;

    using T_vec = typename VectorOperations::vector_type;
    using T = typename VectorOperations::scalar_type;

public:
    struct is_periodic_orbit_reprojected
    {
        static const bool value = true;
    };

    struct linear_operator_type
    {
        linear_operator_type(glued_poincare_map_linear_op_t*& poincare_map_x_p):
        poincare_map_x_(poincare_map_x_p)
        {}
        ~linear_operator_type() = default;

        bool apply(const T_vec& x, T_vec& f)const
        {
            poincare_map_x_->apply(x, f);
            return true;
        }
    private:
        glued_poincare_map_linear_op_t* poincare_map_x_;
    };
    linear_operator_type* linear_operator;

    struct preconditioner_type
    {
        preconditioner_type() = default;
        ~preconditioner_type() = default;
        void apply(T_vec& f)const
        {}
        void set_operator(const linear_operator_type *op_)const 
        {}
    };
    preconditioner_type* preconditioner;

    periodic_orbit_nonlinear_operator(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p, Log* log_p, T max_time_p, T param_p = 1.0,  const std::string& method_p ="RKDP45", T dt_initial_p = 1.0/700.0):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p),
    log_(log_p) 
    {
        poincare_map_ = new poincare_map_operator_t(vec_ops_, nonlin_op_, log_, max_time_p, param_p, method_p, dt_initial_p);
        poincare_map_x_ = new glued_poincare_map_linear_op_t(vec_ops_, nonlin_op_, log_, max_time_p, param_p, method_p, dt_initial_p);
        linear_operator = new linear_operator_type(poincare_map_x_);
        preconditioner = new preconditioner_type();
    }
    ~periodic_orbit_nonlinear_operator()
    {
        delete preconditioner;
        delete linear_operator;
        delete poincare_map_;
        delete poincare_map_x_;     
    }
    
    template<class VecOfVecs>
    void set_hyperplanes_from_initial_guesses(const VecOfVecs& init_vecs, const std::vector<T>& lambdas_p)
    {
        size_t num = 0;
        for(auto& v: init_vecs)
        {
            all_hyperplanes_.emplace_back(vec_ops_, nonlin_op_, v, lambdas_p[num++]);
        }
    }

    void set_hyperplane_from_initial_guesses(const T_vec& init_vec, const T lambda_p)
    {
        all_hyperplanes_.emplace_back(vec_ops_, nonlin_op_, init_vec, lambda_p);
        std::pair<hyperplane_t*, hyperplane_t*> h_pair{&all_hyperplanes_[0], &all_hyperplanes_[0]};
        poincare_map_->set_hyperplanes(h_pair);
        poincare_map_x_->set_hyperplanes(h_pair);        
    }
    // FIX: update hyperplanes
    void set_linearization_point(const T_vec& init_vec, const T init_lambda)
    {
        
        all_hyperplanes_[0].update(0, init_vec, init_lambda);
    }

    template<class VecOfVecs>
    void set_linearization_point(const VecOfVecs& init_vec, const std::vector<T>& init_lambda)
    {
        throw std::logic_error("set_linearization_point: multiple sections to be implemented!");
    }

    void F(const T_vec& u, const T lambda_p, T_vec& v)const
    {
        if(all_hyperplanes_.size() == 1)
        {
            std::pair<hyperplane_t*, hyperplane_t*> h_pair{&all_hyperplanes_[0], &all_hyperplanes_[0]};
            poincare_map_->set_hyperplanes(h_pair);
            poincare_map_x_->set_hyperplanes(h_pair);
            poincare_map_->F(u, lambda_p, v);
            vec_ops_->add_mul(1.0, u, -1.0, v);
        }
        else
        {
            throw std::runtime_error("MULTIPLE SECTIONS TO BE IMPLEMENTED");
            // to be implemented
        }
    }

    void F_and_jacobian_alpha(T_vec& x_out, T_vec& x_lambda_out)const
    {
        if(all_hyperplanes_.size() == 1)
        {
            std::pair<hyperplane_t*, hyperplane_t*> h_pair{&all_hyperplanes_[0], &all_hyperplanes_[0]};
            poincare_map_->set_hyperplanes(h_pair);
            poincare_map_x_->set_hyperplanes(h_pair);
            // poincare_map_->F(u, lambda_p, v);
            poincare_map_x_->F_and_jacobian_alpha(x_out, x_lambda_out);
            //vec_ops_->add_mul(1.0, u, -1.0, v);
        }
        else
        {
            throw std::runtime_error("MULTIPLE SECTIONS TO BE IMPLEMENTED");
            // to be implemented
        }            
    }

    void F_and_jacobian_alpha(const T_vec& u, const T lambda_p, T_vec& x_out, T_vec& x_lambda_out)const
    {
        if(all_hyperplanes_.size() == 1)
        {
            std::pair<hyperplane_t*, hyperplane_t*> h_pair{&all_hyperplanes_[0], &all_hyperplanes_[0]};
            poincare_map_->set_hyperplanes(h_pair);
            poincare_map_x_->set_hyperplanes(h_pair);
            // poincare_map_->F(u, lambda_p, x_out);
            poincare_map_x_->F_and_jacobian_alpha(u, lambda_p, x_out, x_lambda_out);
            // vec_ops_->add_mul(1.0, u, -1.0, x_lambda_out);
        }
        else
        {
            throw std::runtime_error("MULTIPLE SECTIONS TO BE IMPLEMENTED");
            // to be implemented
        }            
    }


    void time_stepper(T_vec& x_p, const T param_p, const std::pair<T,T> time_interval_p)
    {
        poincare_map_->time_stepper(x_p, param_p, time_interval_p);
    }    
    void save_norms(const std::string& file_name_)
    {
        poincare_map_->save_norms(file_name_);
    }
    void save_period_estmate_norms(const std::string& file_name_)const
    {
        poincare_map_->save_period_estmate_norms(file_name_);
    }

    void reproject(T_vec& x) const
    {
        if(all_hyperplanes_.size() == 1)
        {
            all_hyperplanes_[0].restore_from(x); // R^{n} with zero component -> R^{n}
        }
        else
        {
            throw std::runtime_error("MULTIPLE SECTIONS TO BE IMPLEMENTED");
            //to be implemented
        }        
    }

    void project(T_vec& x1_s) const //not needed!
    {
        // reproject(x1_s); 
    }

    T check_solution_quality(const T_vec& x)const
    {
        return nonlin_op_->check_solution_quality(x);
    }
    void norm_bifurcation_diagram(const T_vec& v_in, std::vector<T>& bif_norms_at_t_)const 
    {
        nonlin_op_->norm_bifurcation_diagram(v_in, bif_norms_at_t_);
        auto T_period = poincare_map_->get_period_estmate_time();
        bif_norms_at_t_.push_back(T_period);
    }

    T get_period_estmate_time() const
    {
        return poincare_map_->get_period_estmate_time();
    }

private:
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;
    Log* log_;

    mutable std::vector<hyperplane_t> all_hyperplanes_;



};


}






#endif // __PERIODOC_ORBIT_PERIODIC_ORBIT_NONLINEAR_OPERATOR_H__