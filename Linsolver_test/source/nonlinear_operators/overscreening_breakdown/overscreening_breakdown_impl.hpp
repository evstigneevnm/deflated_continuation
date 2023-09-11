#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_IMPL_HPP__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_IMPL_HPP__

#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
// namespace overscreening_breakdown_device

// NB Chabyshev points are defined FROM 1 TO N-2:
//   point_in_basis(k) = pi*(2*j - 1.0)/(2.0*(N-2))

namespace nonlinear_operators
{



template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_stiffness_matrix(size_t N, problem_type& problem, matrix_wrap& A, matrix_wrap& iD)
{
    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        for(size_t k=0;k<N;k++)
        {
            if(j == 0)
            {
                A(j,k) = problem.psi(k, 0);
            }
            else if(j == N-2)
            {
                A(j,k) = problem.psi(k, problem.pi() );
            }
            else if(j == N-1)
            {
                A(j,k) = problem.dddpsi_map_at_zero(k);
            }
            else
            {
                auto t_point = problem.point_in_basis(j);
                if(problem.delta>problem.delta_threshold)
                {
                    A(j,k) = problem.ddpsi_map(k, t_point)/problem.delta/problem.delta - problem.ddddpsi_map(k, t_point);
                }
                else
                {
                    A(j,k) = problem.ddpsi_map(k, t_point) - problem.delta*problem.delta*problem.ddddpsi_map(k, t_point);
                }
            }
            
            if(j == k)
            {
                T aii = ((abs(A(j,k))<1.0)?1.0:A(j,k));
                iD(j,k) = 1.0/aii;
            }
            else
            {
                iD(j,k) = 0.0;
            }
        }
    }
}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_rhs_linearization_matrix(size_t N, problem_type& problem, T_vec& u_solution, matrix_wrap& B)
{

    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        for(size_t k=0;k<N;k++)
        {

            if(j == 0)
            {
                B(j,k) = 0.0;
            }
            else if(j == N-2)
            {
                B(j,k) = 0.0;
            }
            else if(j == N-1)
            {
                B(j,k) = 0.0;
            }
            else
            {
                auto t_point = problem.point_in_basis(j);
                auto x_point = problem.point_in_domain(j);
                auto u_point = u_solution[j];
                B(j,k) = problem.psi(k, t_point)*problem.right_hand_side_linearization(x_point, u_point);
                if(problem.delta>problem.delta_threshold)
                {
                    B(j,k) /= (problem.delta*problem.delta);
                }
            }
        }
    }

}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_mass_matrix(size_t N, problem_type& problem, matrix_wrap& M)
{

    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        for(size_t k=0;k<N;k++)
        {
            if(j == 0)
            {
                M(j,k) = problem.psi(k, 0);
            }
            else if(j == N-2)
            {
                M(j,k) = problem.psi(k, problem.pi() );
            }
            else if(j == N-1)
            {
                M(j,k) = problem.dddpsi_map_at_zero(k);
            }
            else
            {
                auto t_point = problem.point_in_basis(j);
                M(j,k) = problem.psi(k, t_point);
            }
        }
    }
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::calucalte_function_at_basis(size_t N, problem_type& problem, bool mult, T_vec& res)
{
    
    #pragma omp parallel for
    for(size_t j = 0;j<N;j++)
    {
        
        if(j == 0)
        {
            res[j] = 0; //u bc at infinity
        }
        else if(j == N-2)
        {
            res[j] = problem.u0; //u bc at zero
        }
        else if(j == N-1)
        {
            res[j] = 0; //u_{xxx} bc at zero
        }
        else
        {
            auto x_point = problem.point_in_domain(j);
            if(mult)
            {
                res[j] *= problem.initial_function(x_point);
            }
            else
            {
                res[j] = problem.initial_function(x_point);
            }
        }
    }
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::reformulate_bcs(size_t N, problem_type& problem, T_vec& res)
{

    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        if(j == 0)
        {
            res[j] = 0; //u bc at infinity
        }
        else if(j == N-2)
        {
            res[j] = problem.u0; //u bc at zero
        }
        else if(j == N-1)
        {
            res[j] = 0; //u_{xxx} bc at zero
        }     
    }

}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_right_hand_side(size_t N, problem_type& problem, T_vec& u, T_vec& res)
{

    #pragma omp parallel for
    for(size_t j = 0;j<N;j++)
    {
        if(j == 0)
        {
            res[j] = 0; //u bc at infinity
        }
        else if(j == N-2)
        {
            res[j] = -problem.u0; //u bc at zero
        }
        else if(j == N-1)
        {
            res[j] = 0; //u_{xxx} bc at zero
        }
        else
        {
            auto x_point = problem.point_in_domain(j);
            auto u_point = u_solution[j];
            res[j] = -problem.right_hand_side(x_point, u_point); //we assume that the form is Lu-rhs=0, hence '-'
            if(problem.delta>problem.delta_threshold)
            {
                res[j] /= (problem.delta*problem.delta);
            }
        }        
    }

}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_right_hand_parameter_derivative(size_t N, problem_type& problem, T_vec& u, T_vec& res)
{

    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        if(j == 0)
        {
            res[j] = 0; //u bc at infinity
        }
        else if(j == N-2)
        {
            res[j] = -problem.right_hand_side_parameter_derivative_u0(0, 0);
        }
        else if(j == N-1)
        {
            res[j] = 0; //u_{xxx} bc at zero
        }
        else
        {
            auto x_point = problem.point_in_domain(j);
            auto u_point = u_solution[j];
            res[j] = -problem.right_hand_side_parameter_derivative(x_point, u_point); //we assume that the form is Lu-rhs=0, hence '-'
            if(problem.delta>problem.delta_threshold)
            {
                res[j] /= (problem.delta*problem.delta);
            }
        }         
    }

}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::smooth_random_data(size_t N, T_vec& res, int smooth_steps)
{
    const T w = 0.1;
    const T wc = 1.0 - 2.0*w;

    for(int l=0;l<smooth_steps;l++)
    {
        #pragma omp parallel for
        for(size_t j=0;j<N;j++)
        {
            if(j == 0)
            {
                res[j] = 0; //u bc at infinity
            }
            else if(j == N-2)
            {
                res[j] = 1; //u bc at zero
            }
            else if(j == N-1)
            {
                res[j] = 0; //u_{xxx} bc at zero
            }
            else
            {
                res[j] = w*res[j-1] + wc*res[j]+w*res[j+1];
                // if(j<N/3)
                // {
                //     res[j] = 0;
                // }
            }             
        }       
    }
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::fill_points_at_basis(size_t N,  problem_type& problem, T_vec& res)
{
    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        if(j == 0)
        {
            res[j] = 0; //u bc at infinity
        }
        else if(j == N-2)
        {
            auto t_point = problem.pi();
            res[j] = t_point;
        }
        else if(j == N-1)
        {
            res[j] = 0; //u_{xxx} bc at zero
        }
        else
        {
            auto t_point = problem.point_in_basis(j);
            res[j] = t_point;
        }        
    }
}
template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::fill_points_at_domain(size_t N,  problem_type& problem, T_vec& res)
{
    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        if(j == 0)
        {
            res[j] = 0; //u bc at infinity
        }
        else if(j == N-2)
        {
            auto x_point = 0.0;
            res[j] = x_point;
        }
        else if(j == N-1)
        {
            res[j] = 0; //u_{xxx} bc at zero
        }
        else
        {
            auto x_point = problem.point_in_domain(j);
            res[j] = x_point;
        }        
    }

}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::set_shifted_matrix(size_t N, T alpha, matrix_wrap& A, T beta, matrix_wrap& aApbE)
{
    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        for(size_t k=0;k<N;k++)
        {
            T delta = ( (j==k)?1.0:0.0 );

            aApbE(j,k) = alpha*A(j,k) + beta*delta;            
        }
    }

}



template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::set_identity_matrix(size_t N, matrix_wrap& E)
{
    #pragma omp parallel for
    for(size_t j=0;j<N;j++)
    {
        for(size_t k=0;k<N;k++)
        {
            T delta = ( (j==k)?static_cast<T>(1.0):static_cast<T>(0.0) );
            E(j,k) = delta;
        }
    }
}


}


#endif