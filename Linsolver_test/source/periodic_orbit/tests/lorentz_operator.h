#ifndef __NONLINEAR_OPERATORS_LORENTZ_OPERATOR_H__
#define __NONLINEAR_OPERATORS_LORENTZ_OPERATOR_H__

#include <stdexcept>
#include <vector>

namespace nonlinear_operators
{

// some parameters:
// sigma=10.0,     # Similar to Lorenz
// rho=28.0,       # Chaos parameter
// beta=8.0/3.0,   # Lorenz parameter
// epsilon=0.006,  # Very small cubic damping (ensures global stability)
// delta=0.00   # Very small asymmetry (prevents other fixed points)

template <class VectorOperations>
struct lorentz
{
    using T     = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    lorentz(
        VectorOperations *vec_ops_p,
        unsigned int      used_param_number,
        T                 sigma_init,
        T                 rho_init,
        T                 beta_init,
        T                 epsilon_init,
        T                 delta
    )
        : vec_ops_( vec_ops_p ),
          used_param_number_( used_param_number ), param{ sigma_init, rho_init, beta_init, epsilon_init, delta },
          param0{ sigma_init, rho_init, beta_init, epsilon_init, delta }
    {
        vec_ops_->init_vector( x0 );
        vec_ops_->start_use_vector( x0 );
    }
    ~lorentz()
    {
        vec_ops_->stop_use_vector( x0 );
        vec_ops_->free_vector( x0 );
    }

    void F( const T time_p, const T_vec &in_p, const T param_p, T_vec &out_p ) const
    {
        param[used_param_number_] = param_p;

        // # Modified Lorenz equations for global stability
        // dxdt = -self.sigma * x + self.sigma * y - self.epsilon * x**3
        // dydt = self.rho * x - y - x * z + self.delta
        // dzdt = -self.beta * z + x * y - self.delta
        // sigma, rho, beta, epsilon, delta
        // 0      1    2     3        4
        out_p[0] = -param[0] * in_p[0] + param[0] * in_p[1] - param[3] * in_p[0] * in_p[0] * in_p[0];
        out_p[1] = param[1] * in_p[0] - in_p[1] - in_p[0] * in_p[2] + param[4];
        out_p[2] = -param[2] * in_p[2] + in_p[0] * in_p[1] - param[4];
    }

    void set_linearization_point( const T_vec &x_p, const T param_p )
    {
        param0[used_param_number_] = param_p;
        vec_ops_->assign( x_p, x0 );
    }

    void set_initial( T_vec &x0 ) const
    {
        x0[0] = 2.2;
        x0[1] = 30.5;
        x0[2] = 2.5;
    }

    void set_period_point( T_vec &x0 ) const
    {
        x0[0] = -8.308455;
        x0[1] = 21.347423;
        x0[2] = 26.958582;
    }

    void jacobian_u( const T_vec &x_in_p, T_vec &x_out_p ) const
    {
        // sigma, rho, beta, epsilon, delta
        // 0      1    2     3        4
        // [-self.sigma - 3*self.epsilon*x**2, self.sigma, 0],
        // [self.rho - z, -1, -x],
        // [y, x, -self.beta]
        x_out_p[0] = ( -param[0] - 3 * param[3] * x0[0] * x0[0] ) * x_in_p[0] + param[0] * x_in_p[1];
        x_out_p[1] = ( param[1] - x0[2] ) * x_in_p[0] - x_in_p[1] - x0[0] * x_in_p[2];
        x_out_p[2] = x0[1] * x_in_p[0] + x0[0] * x_in_p[1] - param[2] * x_in_p[2];
    }

    void jacobian_alpha( const T_vec &x_in_p, const T param_p, T_vec &x_out_p ) const
    {
        if ( used_param_number_ == 0 )
        {
            x_out_p[0] = 0;
            x_out_p[1] = x_in_p[1];
            x_out_p[2] = 0;
        }
        else if ( used_param_number_ == 1 )
        {
            x_out_p[0] = 0;
            x_out_p[1] = 0;
            x_out_p[2] = 1.0;
        }
        else if ( used_param_number_ == 2 )
        {
            x_out_p[0] = 0;
            x_out_p[1] = 0;
            x_out_p[2] = -x_in_p[2];
        }
        else
        {
            throw std::logic_error( "Incorrect number of parameters provided in the constructor." );
        }
    }

    void jacobian_alpha( T_vec &x_out_p ) const
    {
        jacobian_alpha( x0, param0[used_param_number_], x_out_p );
    }

    void norm_bifurcation_diagram( const T_vec &x0, std::vector<T> &norm_vec ) const
    {
        norm_vec.push_back( x0[0] );
        norm_vec.push_back( x0[1] );
        norm_vec.push_back( x0[2] );
    }
    T check_solution_quality( const T_vec &x ) const
    {
        bool finite = true;
        for ( int j = 0; j < 3; j++ )
        {
            finite &= std::isfinite( x[j] );
        }
        return finite;
    }

    T get_selected_parameter_value() const
    {
        return param[used_param_number_];
    }

private:
    unsigned int             used_param_number_;
    mutable std::array<T, 5> param;
    T_vec                    x0;
    mutable std::array<T, 5> param0;
    VectorOperations        *vec_ops_;
};
} // namespace nonlinear_operators

#endif // __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_H__