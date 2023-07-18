#ifndef __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_H__
#define __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_H__


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

    rossler(VectorOperations* vec_ops_p, unsigned int used_param_number, T a_init, T b_init, T c_init):
    vec_ops_(vec_ops_p),
    used_param_number_(used_param_number), 
    param{a_init, b_init, c_init}, 
    param0{a_init, b_init, c_init}
    {
        vec_ops_->init_vector(x0); vec_ops_->start_use_vector(x0);
    }
    ~rossler()
    {
        vec_ops_->stop_use_vector(x0); vec_ops_->free_vector(x0);
    }

    void F(const T time_p, const T_vec& in_p, const T param_p, T_vec& out_p )const
    {
        param[used_param_number_] = param_p;

        out_p[0] = -in_p[1]-in_p[2];
        out_p[1] = in_p[0]+param[0]*in_p[1];
        out_p[2] = param[1] + in_p[2]*(in_p[0]-param[2]);
    }

    void set_linearization_point(const T_vec& x_p, const T param_p)
    {
        param0[used_param_number_] = param_p;
        vec_ops_->assign(x_p, x0);

    }

    void set_initial(T_vec& x0)const
    {
        x0[0] = 2.2;
        x0[1] = 0.0;
        x0[2] = 0.0;
    }


    void set_period_point(T_vec& x0)const
    {
        x0[0] = 5.28061710527368;
        x0[1] = -7.74063775648652;
        x0[2] = 0.0785779366135108;
    }

    void jacobian_u(const T_vec& x_in_p, T_vec& x_out_p)
    {
        // alpha, b, mu
        // J = [0 -1 -1;1 alpha 0;x(3) 0 x(1)-mu];
        // u = [-v(2)-v(3); v(1)+alpha*v(2); x(3)*v(1)+x(1)*v(3)-mu*v(3)];
        x_out_p[0] = -x_in_p[1]-x_in_p[2];
        x_out_p[1] = x_in_p[0]+param0[0]*x_in_p[1];
        x_out_p[2] = x0[2]*x_in_p[0]+x0[0]*x_in_p[2]-param0[2]*x_in_p[2];

    }

    void norm_bifurcation_diagram(const T_vec& x0, std::vector<T>& norm_vec)const
    {
        norm_vec.push_back(x0[0]);
        norm_vec.push_back(x0[1]);
        norm_vec.push_back(x0[2]);
    }
    T check_solution_quality(const T_vec& x)const
    {
        bool finite = true;
        for(int j = 0;j<3;j++)
        {
            finite &= std::isfinite(x[j]);
        }
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