#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_DETAIL_PROBLEM_H__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_DETAIL_PROBLEM_H__

#include <utils/device_tag.h>
#include "compute_infinity_cuda.h"

namespace detail
{

template<class T>
struct overscreening_breakdown_problem
{
    overscreening_breakdown_problem(size_t N_p, int parameter_number_p, T sigma_p, T L_p, T delta_p, T gamma_p, T mu_p, T u0_p ):
    N(N_p), parameter_number_(parameter_number_p), L(L_p), delta(delta_p), gamma(gamma_p), mu(mu_p), u0(u0_p), sigma(sigma_p), which(0)
    {}

// T sigma = 1.0; //0
// T L = 1.0;
// T gamma = 1.0; //1
// T delta = 1.0; //2  
// T mu = 1.0;    //3
// T u0 = 1.0;    //4 

    void set_state(T parameter_p)
    {
        switch(parameter_number_)
        {
            case 0:
                sigma = parameter_p;
                break;
            case 1:
                gamma = parameter_p;
                break;
            case 2:
                delta = parameter_p;
                break;
            case 3:
                mu = parameter_p;
                break;
            case 4:
                u0 = parameter_p;
                break;
        }

    }

    __DEVICE_TAG__ inline T g_func(T x)
    {
        if(sigma > 0)
        {
            return 1.0/(sqrt(2.0*M_PI))*exp(-(1.0/2.0)*(x/sigma)*(x/sigma) ); 
        }
        else
        {
            return 1.0/(sqrt(2.0*M_PI));
        }
    }

    __DEVICE_TAG__ inline T right_hand_side(T x, T u)
    {
        T num = sinh(u) - g_func(x)*0.5*mu*exp(u);
        T din = (1 + 2.0*gamma*sinh(u/2.0)*sinh(u/2.0) );
        return num/din;

    }

    __DEVICE_TAG__ inline T right_hand_side_linearization(T x, T u)
    {
        T num = (2.0*(gamma + cosh(u) - gamma*cosh(u)) + (exp(u)*(-1.0 + gamma) - gamma)*mu*g_func(x));
        T din = 2.0*(1.0 - gamma + gamma*cosh(u))*(1.0 - gamma + gamma*cosh(u));
        return num/din;
    }


    __DEVICE_TAG__ inline T right_hand_side_parameter_derivative_sigma(T x, T u)
    {
        // printf("sigma = %e\n", sigma);
        if(sigma > 0)
        {
            return -((exp(u-x*x/(2.0*sigma*sigma))*x*x*mu)/(2.0*sqrt(2.0*M_PI)*sigma*sigma*sigma*(1.0+2.0*gamma*sinh(0.5*u)*sinh(0.5*u))));
        }
        else
        {
            return -0.0;
        }
    }
    __DEVICE_TAG__ inline T right_hand_side_parameter_derivative_gamma(T x, T u)
    {
        return -((2*sinh(u/2)*sinh(u/2)*(-((exp(u-x*x/(2*sigma*sigma))*mu)/(2*sqrt(2*M_PI)))+sinh(u)))/((1+2*gamma*sinh(u/2)*sinh(u/2))*(1+2*gamma*sinh(u/2)*sinh(u/2))));
    } 
    
    __DEVICE_TAG__ inline T right_hand_side_parameter_derivative_delta(T x, T u)
    {
        return 1;
    } 
    __DEVICE_TAG__ inline T right_hand_side_parameter_derivative_mu(T x, T u)
    {
        return -(exp(u-x*x/(2*sigma*sigma))/(2*sqrt(2*M_PI)*(1+2*gamma*sinh(u/2)*sinh(u/2))));
    }  
    __DEVICE_TAG__ inline T right_hand_side_parameter_derivative_u0(T x, T u)
    {
        if(parameter_number_ == 4)
            return 1;
        else
            return 0;
    }               
// T sigma = 1.0; //0
// T L = 1.0;
// T gamma = 1.0; //1
// T delta = 1.0; //2  
// T mu = 1.0;    //3
// T u0 = 1.0;    //4 

    __DEVICE_TAG__ inline T right_hand_side_parameter_derivative(T x, T u)
    {
        switch(parameter_number_)
        {
            case 0:
                return right_hand_side_parameter_derivative_sigma(x, u);
                break;
            case 1:
                return right_hand_side_parameter_derivative_gamma(x, u);
                break;
            case 2:
                return right_hand_side_parameter_derivative_delta(x, u);
                break;      
            case 3:
                return right_hand_side_parameter_derivative_mu(x, u);
                break;                                    
            case 4: // u0
                return 0;
                break;              
        }


    }

    void rotate_initial_function()
    {
        which++;
        which = (which)%3;
    }

    __DEVICE_TAG__ inline T initial_function(T x)
    {
        if (which == 0)
            return u0*exp(-x);
        else if (which == 1)
            return u0*exp(-x*x);
        else
            return (u0/(x*x+1.0));
    }

    __DEVICE_TAG__ inline T point_in_basis(size_t j)
    {
        // will return Chebysev points only for 1<=j<=N-3
        // other points will be incorrect! beacuse we use 3 boundary conditions.
        return M_PI*(2.0*j - 1.0)/(2.0*(N-3) );
    }
    __DEVICE_TAG__ inline T point_in_domain(size_t j)
    {
        auto t_point = point_in_basis(j);
        return from_basis_to_domain(t_point);
    }

    __DEVICE_TAG__ inline T from_basis_to_domain(T t)
    {
        T ss = sin(0.5*t);
        T cs = cos(0.5*t);
        T ss2 = ss*ss;
        T cs2 = cs*cs;
        if(ss2 == 0)
        {
            return compute_infinity<T>();
        }
        T ret = L*cs2/ss2; 
        return ret;
    }
    __DEVICE_TAG__ inline T from_domain_to_basis(T x)
    {
        if(x == 0)
        {
            return M_PI;
        }
        T y = 1/x;
        T ret = 2*atan(sqrt(L*y));
        return ret;
    }
    __DEVICE_TAG__ inline T psi(int n, T t)
    {
        return cos(n*t);
    }
    __DEVICE_TAG__ inline T dpsi(int n, T t)
    {
        return -n*sin(n*t);
    }
    __DEVICE_TAG__ inline T ddpsi(int n, T t)
    {
        return -n*n*cos(n*t);
    }
    __DEVICE_TAG__ inline T dddpsi(int n, T t)
    {
        return n*n*n*sin(n*t);
    }
    __DEVICE_TAG__ inline T ddddpsi(int n, T t)
    {
        return n*n*n*n*cos(n*t);
    }
    __DEVICE_TAG__ T ddpsi_map(int l, T t)
    {
        T sn = sin(t/2.0);
        T tn = tan(t/2.0);
        T sin2 = sn*sn;
        T tan3 = tn*tn*tn;
        return 1.0/(2.0*L*L)*sin2*tan3*((2.0 + cos(t))*dpsi(l, t) + sin(t)*ddpsi(l, t));
    }
    __DEVICE_TAG__ T ddddpsi_map(int l, T t)
    {
        T sn = sin(t/2.0);
        T tn = tan(t/2.0);
        T sin2 = sn*sn;
        T tan7 = tn*tn*tn*tn*tn*tn*tn;
        T common = 1.0/(16.0*L*L*L*L)*sin2*tan7;
        T p1 = 3.0*(32.0 + 29.0*cos(t) + 8.0*cos(2.0*t) + cos(3.0*t))*dpsi(l, t);
        T p2 = (91.0 + 72.0*cos(t) + 11.0*cos(2.0*t))*ddpsi(l, t);
        T p3 = (6.0*(2.0 + cos(t))*dddpsi(l, t) + sin(t)*ddddpsi(l, t));
        return common*(p1+sin(t)*(p2 + 2*sin(t)*p3));
    }
    __DEVICE_TAG__ T dddpsi_map_at_zero(size_t k)
    {
        T m1degk = (k%2==0?1.0:-1.0);//(-1)**k;
        return -(4.0/(15.0*L*L*L))*m1degk*(1.0*k*1.0*k)*(23.0 + 20.0*k*k + 2*k*k*k*k);
    }

    T L, gamma, mu, u0, sigma, delta;
    size_t N;
    char which;
    const T delta_threshold = 0.001;
    int parameter_number_;
};

}

#endif