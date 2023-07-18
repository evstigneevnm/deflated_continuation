#ifndef __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_GPU_H__
#define __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_GPU_H__

#include "rossler_operator_ker.h"

namespace nonlinear_operators
{

    // some parameters:
    // a = 0.2 b = 0.2 c = 5.7
    // a = 0.2 b = 0.2 c = 14.0
    // standard bifurcaitons:
    // a=0.1, b=0.1:
    // c = 4, period-1 orbit.
    // c = 6, period-2 orbit.
    // c = 8.5, period-4 orbit.
    // c = 8.7, period-8 orbit.
    // c = 9, sparse chaotic attractor.
    // c = 12, period-3 orbit.
    // c = 12.6, period-6 orbit.
    // c = 13, sparse chaotic attractor.
    // c = 18, filled-in chaotic attractor.
    
template<class VectorOperations>
struct rossler //https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
{

    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

private:
    using ker_t = rossler_operator_ker<T, T_vec, 32>;

    ker_t* ker;

public:
    rossler(VectorOperations* vec_ops_p, unsigned int used_param_number, T a_init, T b_init, T c_init):
    vec_ops_(vec_ops_p),
    used_param_number_(used_param_number), 
    param{a_init, b_init, c_init}, 
    param0{a_init, b_init, c_init}
    {
        vec_ops_->init_vector(x0); vec_ops_->start_use_vector(x0);
        ker = new ker_t();
    }
    ~rossler()
    {
        delete ker;
        vec_ops_->stop_use_vector(x0); vec_ops_->free_vector(x0);
    }

    void F(const T time_p, const T_vec& in_p, const T param_p, T_vec& out_p )const
    {
        param[used_param_number_] = param_p;
        ker->F(in_p, {param[0],param[1],param[2]}, out_p);
    }

    void set_linearization_point(const T_vec& x_p, const T param_p)
    {
        param0[used_param_number_] = param_p;
        vec_ops_->assign(x_p, x0);

    }

    void set_initial(T_vec& x0)const
    {
        ker->set_initial(x0);
    }


    void set_period_point(T_vec& x0)const
    {
        ker->set_period_point(x0);
    }

    void jacobian_u(const T_vec& x_in_p, T_vec& x_out_p)const
    {   
        ker->jacobian_u(x0, {param[0],param[1],param[2]}, x_in_p, x_out_p);
    }

    void norm_bifurcation_diagram(const T_vec& x0, std::vector<T>& norm_vec)const
    {
        T_vec x0_h = vec_ops_->view(x0);
        norm_vec.push_back(x0_h[0]);
        norm_vec.push_back(x0_h[1]);
        norm_vec.push_back(x0_h[2]);
    }
    T check_solution_quality(const T_vec& x)const
    {
        bool finite = vec_ops_->check_is_valid_number(x);
        return finite;
    }

    T get_selected_parameter_value()const
    {
        return param[used_param_number_];
    }

private:
    unsigned int used_param_number_;
    mutable std::array<T, 3> param;
    T_vec x0;
    mutable std::array<T, 3> param0;
    VectorOperations* vec_ops_;


};
}

#endif // __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_H__