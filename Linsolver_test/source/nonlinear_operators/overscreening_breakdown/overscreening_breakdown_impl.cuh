#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_IMPL_CUH__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_IMPL_CUH__

#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
// namespace overscreening_breakdown_device

// NB Chabyshev points are defined FROM 1 TO N-2:
//   point_in_basis(k) = pi*(2*j - 1.0)/(2.0*(N-2))

namespace nonlinear_operators
{

template<typename T, typename T_vec, class Problem, typename Matrix>
__global__ void form_stiffness_matrix_kernel(size_t N, T delta, Problem problem, Matrix A, Matrix iD)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;

    if((j>=N)||(k>=N)) return;

    if(j == 0)
    {
        A(j,k) = problem.psi(k, 0);
    }
    else if(j == N-2)
    {
        A(j,k) = problem.psi(k, M_PI);
    }
    else if(j == N-1)
    {
        A(j,k) = problem.dddpsi_map_at_zero(k);
    }
    else
    {
        auto t_point = problem.point_in_basis(j);
        if(delta>problem.delta_threshold)
        {
            A(j,k) = problem.ddpsi_map(k, t_point)/delta/delta - problem.ddddpsi_map(k, t_point);
        }
        else
        {
            A(j,k) = problem.ddpsi_map(k, t_point) - delta*delta*problem.ddddpsi_map(k, t_point);
        }
        
    }
    
    if(j==k)
    {
        T aii = ((abs(A(j,k))<1.0)?1.0:A(j,k));
        iD(j,k) = 1.0/aii;
    }
    else
    {
        iD(j,k) = 0.0;
    }
    

}


template<typename T, typename T_vec, class Problem, typename Matrix>
__global__ void form_rhs_linearization_matrix_kernel(size_t N,  Problem problem, T delta, T_vec u_solution, Matrix B)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;

    if((j>=N)||(k>=N)) return;

    if(j == 0)
    {
        B(j,k) = 0;
    }
    else if(j == N-2)
    {
        B(j,k) = 0;
    }
    else if(j == N-1)
    {
        B(j,k) = 0;
    }
    else
    {
        auto t_point = problem.point_in_basis(j);
        auto x_point = problem.point_in_domain(j);
        auto u_point = u_solution[j];
        B(j,k) = problem.psi(k, t_point)*problem.right_hand_side_linearization(x_point, u_point);
        if(delta>problem.delta_threshold)
        {
            B(j,k) /= (delta*delta);
        }
    }

}

template<typename T, typename T_vec, class Problem, typename Matrix>
__global__ void form_mass_matrix_kernel(size_t N,  Problem problem, Matrix M)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;

    if((j>=N)||(k>=N)) return;

    if(j == 0)
    {
        M(j,k) = problem.psi(k, 0);
    }
    else if(j == N-2)
    {
        M(j,k) = problem.psi(k, M_PI);
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


template<typename T, typename T_vec, class Problem, bool Mult = false>
__global__ void calucalte_function_at_basis_kernel(size_t N,  Problem problem, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;

    if(j>=N) return;

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
        if constexpr (Mult)
        {
            res[j] *= problem.initial_function(x_point);
        }
        else
        {
            res[j] = problem.initial_function(x_point);
        }
    }
}

// note that points in zero boundary conditions are set to 0
template<typename T, typename T_vec, class Problem, bool Basis>
__global__ void fill_points_at_basis_or_domain_kernel(size_t N,  Problem problem, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;

    if(j>=N) return;

    if(j == 0)
    {
        res[j] = 0; //u bc at infinity
    }
    else if(j == N-2)
    {
        
        if constexpr(Basis)
        {
            auto t_point = M_PI;
            res[j] = t_point;
        }
        else
        {
            auto x_point = 0.0;
            res[j] = x_point;
        }

    }
    else if(j == N-1)
    {
        res[j] = 0; //u_{xxx} bc at zero
    }
    else
    {
        if constexpr(Basis)
        {
            auto t_point = problem.point_in_basis(j);
            res[j] = t_point;
        }
        else
        {
            auto x_point = problem.point_in_domain(j);
            res[j] = x_point;
        }
    }
}


template<typename T, typename T_vec, class Problem>
__global__ void reformulate_bcs_kernel(size_t N,  Problem problem, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;

    if(j>=N) return;

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


template<typename T, typename T_vec, class Problem>
__global__ void form_right_hand_side_kernel(size_t N, Problem problem, T delta, T_vec u_solution, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;
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
        if(delta>problem.delta_threshold)
        {
            res[j] /= (delta*delta);
        }
    }
}



template<typename T, typename T_vec, class Problem>
__global__ void form_right_hand_parameter_derivative_kernel(size_t N, Problem problem, T delta, T_vec u_solution, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;
    if(j == 0)
    {
        res[j] = 0; //u bc at infinity
    }
    else if(j == N-2)
    {
        res[j] = 0;//problem.u0; //u bc at zero
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
        if(delta>problem.delta_threshold)
        {
            res[j] /= (delta*delta);
        }
    }    
}


template<typename T, typename T_vec>
__global__ void smooth_random_data_kernel(size_t N, T_vec res)
{
    const T w = 0.1;
    const T wc = 1.0 - 2.0*w;
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;
    if(j == 0)
    {
        res[j] = 0; //u bc at infinity
    }
    else if(j == N-2)
    {
        res[j] = 1;//problem.u0; //u bc at zero
    }
    else if(j == N-1)
    {
        res[j] = 0; //u_{xxx} bc at zero
    }
    else
    {
        res[j] = w*res[j-1] + wc*res[j]+w*res[j+1];
    }    
}



template<typename T, typename T_vec, typename Matrix>
__global__ void set_shifted_matrix_kernel(size_t N, T alpha, Matrix A, T beta, Matrix aApbE)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;

    if((j>=N)||(k>=N)) return;

    T delta = ( (j==k)?1.0:0.0 );

    aApbE(j,k) = alpha*A(j,k) + beta*delta;

}
template<typename T, typename Matrix>
__global__ void set_identity_matrix_kernel(size_t N, Matrix E)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;

    if((j>=N)||(k>=N)) return;

    T delta = ( (j==k)?static_cast<T>(1.0):static_cast<T>(0.0) );
    E(j,k) = delta;

}




template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_stiffness_matrix(size_t N, T delta, problem_s problem, matrix_device_s A, matrix_device_s iD)
{
    form_stiffness_matrix_kernel<T, T_vec, problem_s, matrix_device_s><<<dim_grid, dim_block>>>(N, delta, problem, A, iD);
}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_rhs_linearization_matrix(size_t N, problem_s problem, T delta, T_vec u_solution, matrix_device_s B)
{
    form_rhs_linearization_matrix_kernel<T, T_vec, problem_s, matrix_device_s><<<dim_grid, dim_block>>>(N, problem, delta, u_solution, B);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_mass_matrix(size_t N, problem_s problem, matrix_device_s mass_matrix)
{
    form_mass_matrix_kernel<T, T_vec, problem_s, matrix_device_s><<<dim_grid, dim_block>>>(N, problem, mass_matrix);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::calucalte_function_at_basis(size_t N, problem_s problem, bool mult, T_vec res)
{
    if(mult)
    {
        calucalte_function_at_basis_kernel<T, T_vec, problem_s, true><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
    }
    else
    {
        calucalte_function_at_basis_kernel<T, T_vec, problem_s, false><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
    }        
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::reformulate_bcs(size_t N, problem_s problem, T_vec res)
{
    reformulate_bcs_kernel<T, T_vec, problem_s><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_right_hand_side(size_t N, problem_s problem, T delta, T_vec u, T_vec res)
{
    form_right_hand_side_kernel<T, T_vec, problem_s><<<dim_grid_1d, dim_block_1d>>>(N, problem, delta, u, res);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_right_hand_parameter_derivative(size_t N, problem_s problem, T delta, T_vec u, T_vec res)
{
    form_right_hand_parameter_derivative_kernel<T, T_vec, problem_s><<<dim_grid_1d, dim_block_1d>>>(N, problem, delta, u, res);
}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::smooth_random_data(size_t N, T_vec res, int smooth_steps)
{
    for(int j=0;j<smooth_steps;j++)
    {
        smooth_random_data_kernel<T, T_vec><<<dim_grid_1d, dim_block_1d>>>(N, res);
    }
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::fill_points_at_basis(size_t N,  problem_s problem, T_vec res)
{
    fill_points_at_basis_or_domain_kernel<T, T_vec, problem_s, true><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
}
template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::fill_points_at_domain(size_t N,  problem_s problem, T_vec res)
{
    fill_points_at_basis_or_domain_kernel<T, T_vec, problem_s, false><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::set_shifted_matrix(size_t N, T a, matrix_device_s A, T b, matrix_device_s aApbE)
{
    set_shifted_matrix_kernel<T, T_vec, matrix_device_s><<<dim_grid, dim_block>>>(N, a, A, b, aApbE);

}



template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::set_identity_matrix(size_t N, matrix_device_s E)
{
    set_identity_matrix_kernel<T, matrix_device_s><<<dim_grid, dim_block>>>(N, E);

}


}


#endif