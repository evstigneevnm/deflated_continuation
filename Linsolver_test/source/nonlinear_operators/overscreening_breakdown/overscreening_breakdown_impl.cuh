#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_IMPL_CUH__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_IMPL_CUH__


#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_ker.h>
#include <utils/cuda_support.h>


// namespace overscreening_breakdown_device

// NB Chabyshev points are defined FROM 1 TO N-2:
//   point_in_basis(k) = pi*(2*j - 1.0)/(2.0*(N-2))

namespace nonlinear_operators
{

const unsigned int BLOCKSIZE_x = 32;
const unsigned int BLOCKSIZE_y = 32;
const unsigned int BLOCKSIZE = BLOCKSIZE_x*BLOCKSIZE_y;


namespace detail
{
template<class Ord, class T, class Vec>
T sum_reduce(Ord N, Vec ptr)
{
    T res = thrust::reduce(
        ::thrust::device_pointer_cast(ptr), 
        ::thrust::device_pointer_cast(ptr) + N, 
        T(0)
        );

    return res;
}

template<class Ord, class Keys, class Vals>
Ord reduce_by_key(Ord rows_n, const Keys keys_in, const Vals vals_in, Keys keys_out, Vals vals_out)
{
    
    auto new_end = thrust::reduce_by_key(
        thrust::host, //thrust::device probably contains a bug for cuda 10.2!!!
        ::thrust::device_pointer_cast(keys_in ),
        ::thrust::device_pointer_cast(keys_in ) + rows_n, 
        ::thrust::device_pointer_cast(vals_in ),
        ::thrust::device_pointer_cast(keys_out ),
        ::thrust::device_pointer_cast(vals_out )
        ); 

    auto end_keys = new_end.first - ::thrust::device_pointer_cast(keys_out );
    auto end_vals = new_end.second - ::thrust::device_pointer_cast(vals_out );
    return end_keys;
}


// for j in (0,N-1)
template<class T, class Problem>
__global__ void set_quadrature_points_kernel(size_t N, Problem problem, T* t, T* x, T* w, T* t_ab, T* x_ab)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=(N-2)) return;

    auto t_min = problem.point_in_basis(j);
    auto t_max = problem.point_in_basis(j+1);
    if(j == 0)
    {
        t_min = 1e-10;
    }
    else if(j==N-3)
    {
        t_max = problem.pi();
    }
    
    
    auto x_max = problem.from_basis_to_domain(t_max);
    auto x_min = problem.from_basis_to_domain(t_min);
    
    
    auto xsi = problem.quadrature_points(t_max, t_min);
    auto wi = problem.quadrature_wights();

    #pragma unroll
    for(int jj = 0;jj<5;++jj)
    {
        auto t_p = xsi[jj];
        t[5*j+jj] = t_p;
        x[5*j+jj] = problem.from_basis_to_domain(t_p);
    }
    #pragma unroll
    for(int jj = 0;jj<5;++jj)
    {
        w[5*j+jj] = wi[jj];
        t_ab[5*j+jj] = 0.5*(t_max-t_min);
        x_ab[5*j+jj] = 0.5*(x_max-x_min);
    } 

}

// for j in (0,N-1): quadrature points
// for k in (0,N): harmonic numbers
// keys size is 5*(N-1)*N
// indexing is keys[i+5*j+5*(N-1)*k]
template<class T, class T_vec, class Problem>
__global__ void set_quadrature_values_kernel(size_t N, Problem problem, T* t, T_vec u, size_t* keys, T* values)
{ 
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;
    if((j>=(N-2))||(k>=N)) return; 
    // printf("blockDim.y = %i, blockIdx.y = %i, threadIdx.y = %i\n", blockDim.y, blockIdx.y, threadIdx.y);
    

    #pragma unroll
    for(int jj=0;jj<5;++jj)
    {
        keys[k+5*(N)*j+jj*N] = 5*j+jj;
    }

    #pragma unroll
    for(int jj=0;jj<5;++jj)
    {
        values[k+5*(N)*j+jj*N] = problem.psi(k, t[ 5*j+jj ])*u[k];
    }


}

template<class T, class Problem>
__global__ void set_rhs_quadrature_values_kernel(size_t N, Problem problem, T* t, T* x, T* u, T* rhs, T* dx_to_dt)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;    
    auto t_p = t[j];
    auto u_p = u[j];
    auto x_p = x[j];
    rhs[j] = problem.right_hand_side(x_p, u_p);
    dx_to_dt[j] = problem.from_basis_to_domain_differential(t_p);
}



template<class T>
__global__ void quadrature_poinwise_kernel(size_t N, T* u, T* w, T* dt, T* t_ab, T* x_ab, T* res_basis, T* res_domain)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;    
    auto w_p = w[j];
    auto u_p = u[j];
    auto dt_p = dt[j];
    auto t_ab_p = t_ab[j];
    auto x_ab_p = x_ab[j];
    res_basis[j] = w_p*u_p*dt_p*t_ab_p;
    res_domain[j] = w_p*u_p*x_ab_p;


}


template<class T>
__global__ void pointwise_integrals_points(size_t N, T* x, T* t, T* x_p, T* t_p)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=(N-2)) return;    

    x_p[j] = 0;
    t_p[j] = 0;
    #pragma unroll
    for(int jj=0;jj<5;jj++)
    {
        x_p[j] += x[5*j+jj]*0.2;
        t_p[j] += t[5*j+jj]*0.2;
    }

}

template<class T>
__global__ void pointwise_integrals_values_kernels(size_t N, T* u, T* u_p)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=(N-2)) return;    

    u_p[j] = 0;
    #pragma unroll
    for(int jj=0;jj<5;jj++)
    {
        u_p[j] += u[5*j+jj];
    }

}


template<class T, class T_vec, class Problem>
void obtain_quadrature_points_wights_values(size_t N, Problem& problem, const T_vec& u_coeffs)
{
    dim3 dim_grid_2d, dim_block_2d;
    dim_grid_2d = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx=floor( (N-2)/( BLOCKSIZE_x ))+1;
    unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
    dim_block_2d = dim3(kx, ky);    
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kl=floor( (N-2)/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kl);    

    size_t size_points = 5*(N-2);
    size_t size_points_times_harmonics = 5*(N-2)*N;


    dim3 dim_grid_1d_e, dim_block_1d_e;
    dim_block_1d_e = dim3(BLOCKSIZE);
    unsigned int kl_e=floor( (size_points)/( BLOCKSIZE ))+1;
    dim_grid_1d_e = dim3(kl_e);


    T* basis_points = device_allocate<T>( size_points );
    T* domain_points = device_allocate<T>( size_points );
    T* wight_points = device_allocate<T>( size_points );
    T* basis_ab = device_allocate<T>( size_points );
    T* domain_ab = device_allocate<T>( size_points );
    T* differential_points = device_allocate<T>( size_points );
    size_t* keys = device_allocate<size_t>( size_points_times_harmonics );
    T* values = device_allocate<T>( size_points_times_harmonics );
    size_t* keys_out = device_allocate<size_t>( size_points_times_harmonics );
    T* values_out = device_allocate<T>( size_points_times_harmonics );
    T* rhs_out = device_allocate<T>( size_points );
    T* poinwise_integral_solution_basis = device_allocate<T>(size_points);
    T* poinwise_integral_solution_domain = device_allocate<T>(size_points);
    T* poinwise_integral_rhs_basis = device_allocate<T>(size_points);
    T* poinwise_integral_rhs_domain = device_allocate<T>(size_points);
    T* segment_integral_solution_basis = device_allocate<T>(N);
    T* segment_integral_solution_domain = device_allocate<T>(N);
    T* segment_integral_rhs_basis = device_allocate<T>(N);
    T* segment_integral_rhs_domain = device_allocate<T>(N);
    T* basis_segments = device_allocate<T>(N);
    T* domain_segments = device_allocate<T>(N);

    set_quadrature_points_kernel<T, Problem><<<dim_grid_1d, dim_block_1d>>>(N, problem, basis_points, domain_points, wight_points, basis_ab, domain_ab);

    set_quadrature_values_kernel<T, T_vec, Problem><<<dim_grid_2d, dim_block_2d>>>(N, problem, basis_points, u_coeffs, keys, values);

    auto out_size = reduce_by_key<size_t, size_t*, T*>(size_points_times_harmonics, keys, values, keys_out, values_out);

    set_rhs_quadrature_values_kernel<T, Problem><<<dim_grid_1d_e, dim_block_1d_e>>>(size_points, problem, basis_points, domain_points, values_out, rhs_out, differential_points);

    quadrature_poinwise_kernel<T><<<dim_grid_1d_e, dim_block_1d_e>>>(size_points, values_out, wight_points, differential_points, basis_ab, domain_ab, poinwise_integral_solution_basis, poinwise_integral_solution_domain);

    quadrature_poinwise_kernel<T><<<dim_grid_1d_e, dim_block_1d_e>>>(size_points, rhs_out, wight_points, differential_points, basis_ab, domain_ab, poinwise_integral_rhs_basis, poinwise_integral_rhs_domain);

    pointwise_integrals_points<T><<<dim_grid_1d, dim_block_1d>>>(N, basis_points, domain_points, basis_segments, domain_segments);

    pointwise_integrals_values_kernels<T><<<dim_grid_1d, dim_block_1d>>>(N, poinwise_integral_solution_basis, segment_integral_solution_basis);
    pointwise_integrals_values_kernels<T><<<dim_grid_1d, dim_block_1d>>>(N, poinwise_integral_solution_domain, segment_integral_solution_domain);
    pointwise_integrals_values_kernels<T><<<dim_grid_1d, dim_block_1d>>>(N, poinwise_integral_rhs_basis, segment_integral_rhs_basis);
    pointwise_integrals_values_kernels<T><<<dim_grid_1d, dim_block_1d>>>(N, poinwise_integral_rhs_domain, segment_integral_rhs_domain);        

    // auto out_size = size_points;
    // std::cout << "out_size = " << out_size << std::endl;

    std::vector<T> th(size_points, 0.0);
    std::vector<T> xh(size_points, 0.0);
    std::vector<T> iu_basis(size_points, 0.0);
    std::vector<T> iu_domain(size_points, 0.0);
    std::vector<T> ir_basis(size_points, 0.0);
    std::vector<T> ir_domain(size_points, 0.0);

    device_2_host_cpy<T>(th.data(), basis_points, size_points);
    device_2_host_cpy<T>(xh.data(), domain_points, size_points);
    device_2_host_cpy<T>(iu_basis.data(), poinwise_integral_solution_basis, size_points);
    device_2_host_cpy<T>(iu_domain.data(), poinwise_integral_solution_domain, size_points);
    device_2_host_cpy<T>(ir_basis.data(), poinwise_integral_rhs_basis, size_points);
    device_2_host_cpy<T>(ir_domain.data(), poinwise_integral_rhs_domain, size_points);

    // for(size_t jj=0;jj<size_points;jj++)
    // {
    
    //     std::cout << th[jj] << " " << iu_basis[jj] << " " << ir_basis[jj] << " " << xh[jj] << " " << iu_domain[jj] << " " << ir_domain[jj] << std::endl;
    // }

    std::vector<T> t_points(N-2, 0.0);
    std::vector<T> x_points(N-2, 0.0);
    std::vector<T> u_basis_points(N-2, 0.0);
    std::vector<T> u_domain_points(N-2, 0.0);
    std::vector<T> r_basis_points(N-2, 0.0);
    std::vector<T> r_domain_points(N-2, 0.0);        
    device_2_host_cpy<T>(t_points.data(), basis_segments, N-2);
    device_2_host_cpy<T>(x_points.data(), domain_segments, N-2);    
    device_2_host_cpy<T>(u_basis_points.data(), segment_integral_solution_basis, N-2);
    device_2_host_cpy<T>(u_domain_points.data(), segment_integral_solution_domain, N-2); 
    device_2_host_cpy<T>(r_basis_points.data(), segment_integral_rhs_basis, N-2);
    device_2_host_cpy<T>(r_domain_points.data(), segment_integral_rhs_domain, N-2);     
    for(size_t jj=0;jj<N-2;jj++)
    {
        std::cout << t_points[jj] << " " << u_basis_points[jj] << " " << r_basis_points[jj] << " " << x_points[jj] << " " << u_domain_points[jj] << " " << r_domain_points[jj] << std::endl;
    }
    
    cudaFree(segment_integral_rhs_domain);
    cudaFree(segment_integral_rhs_basis);
    cudaFree(segment_integral_solution_domain);
    cudaFree(segment_integral_solution_basis);

    cudaFree(poinwise_integral_rhs_domain);
    cudaFree(poinwise_integral_rhs_basis);
    cudaFree(poinwise_integral_solution_domain);
    cudaFree(poinwise_integral_solution_basis);
    cudaFree(differential_points);
    cudaFree(rhs_out);
    cudaFree(keys_out);
    cudaFree(values_out);
    cudaFree(values);
    cudaFree(keys);
    cudaFree(wight_points);
    cudaFree(basis_ab);
    cudaFree(domain_ab);
    cudaFree(domain_points);
    cudaFree(basis_points);

}



template<class T, class Problem>
__global__ void set_points_kernel(size_t N, T max_x, Problem problem, T* x, T* t)
{
    auto dx = max_x/static_cast<T>(N);

    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=(N)) return;
    T x_p = dx*j;
    t[j] = problem.from_domain_to_basis(x_p);
    x[j] = x_p;
}

template<class T, class T_vec, class Problem>
__global__ void set_values_harmonics_kernel(size_t N, size_t N_points, Problem problem, T* t, T_vec u, size_t* keys, T* values)
{ 
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;
    if((j>=(N_points))||(k>=N)) return; 
    // printf("blockDim.y = %i, blockIdx.y = %i, threadIdx.y = %i\n", blockDim.y, blockIdx.y, threadIdx.y);
    
    keys[k+(N)*j] = j;
    values[k+(N)*j] = problem.psi(k, t[j])*u[k];

}

template<class T, class Problem>
__global__ void set_rhs_values_kernels(size_t N_points, Problem problem, T* x, T* u, T* rhs)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=(N_points)) return;    

    auto u_p = u[j];
    auto x_p = x[j];
    rhs[j] = problem.right_hand_side(x_p, u_p);


}


template<class T, class Problem>
__global__ void set_values_upper_triangluar_kernel(size_t N, Problem problem, T* u, size_t* keys, T* values)
{ 
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    size_t k=blockDim.y * blockIdx.y + threadIdx.y;
    if((j>=(N))||(k>=N)) return; 
    // printf("blockDim.y = %i, blockIdx.y = %i, threadIdx.y = %i\n", blockDim.y, blockIdx.y, threadIdx.y);
    
    values[k+(N)*j] = 0;
    size_t idx = k+j*N;
    keys[idx] = j;
    T val = (j>=k)?u[j-k]:0.0;
    values[idx] = val;

}


template<class T, class T_vec, class Problem>
void obtain_equali_distributed_points_wights_values(size_t N, size_t N_points, const T max_x, Problem& problem, const T_vec& u_coeffs)
{

    dim3 dim_grid_2d, dim_block_2d;
    dim_grid_2d = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx=floor( (N_points)/( BLOCKSIZE_x ))+1;
    unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
    dim_block_2d = dim3(kx, ky);    
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kl=floor( (N_points)/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kl);  
    
    size_t size_points_times_harmonics = N_points*N;

    T* domain_points = device_allocate<T>( N_points );
    T* basis_points = device_allocate<T>( N_points );
    size_t* keys = device_allocate<size_t>( size_points_times_harmonics );
    T* values = device_allocate<T>( size_points_times_harmonics );
    size_t* keys_out = device_allocate<size_t>( size_points_times_harmonics );
    T* values_out = device_allocate<T>( size_points_times_harmonics );
    T* rhs_out = device_allocate<T>( N_points );

    set_points_kernel<T, Problem><<<dim_grid_1d, dim_block_1d>>>(N_points, max_x, problem, domain_points, basis_points);
    set_values_harmonics_kernel<T, T_vec, Problem><<<dim_grid_2d, dim_block_2d>>>(N, N_points, problem, basis_points, u_coeffs, keys, values);

    auto out_size = reduce_by_key<size_t, size_t*, T*>(size_points_times_harmonics, keys, values, keys_out, values_out);

    set_rhs_values_kernels<T, Problem><<<dim_grid_1d, dim_block_1d>>>(N_points, problem, domain_points, values_out, rhs_out);

    std::vector<T> x_(N_points, 0.0);
    std::vector<T> t_(N_points, 0.0);    
    std::vector<T> u_(N_points, 0.0);
    std::vector<T> rhs_(N_points, 0.0);

    device_2_host_cpy<T>(x_.data(), domain_points, N_points);
    device_2_host_cpy<T>(t_.data(), basis_points, N_points);
    device_2_host_cpy<T>(u_.data(), values_out, N_points);
    device_2_host_cpy<T>(rhs_.data(), rhs_out, N_points);

    std::stringstream ss;
    ss << "equali_distributed_points_L" << problem.L << "_g" << problem.gamma << "_mu" << problem.mu << "_uo" << problem.u0 << "_s" << problem.sigma << "_d" << problem.delta << ".dat";
    std::fstream f(ss.str(), std::ofstream::out);
    for(size_t jj=0;jj<N_points;jj++)
    {
        f << t_[jj] << " " << x_[jj] << " " << u_[jj] << " " << rhs_[jj] << std::endl;
    }    
    f.close();
    cudaFree(values);
    cudaFree(keys);
    cudaFree(keys_out);
    cudaFree(values_out);
    keys = device_allocate<size_t>( N_points*N_points );
    values = device_allocate<T>( N_points*N_points );
    keys_out = device_allocate<size_t>( N_points*N_points );
    values_out = device_allocate<T>( N_points*N_points );

    
    dim3 dim_grid_2d_NN, dim_block_2d_NN;
    dim_grid_2d_NN = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx_NN=floor( N_points/( BLOCKSIZE_x ))+1;
    unsigned int ky_NN=floor(N_points/( BLOCKSIZE_y ))+1;
    dim_block_2d_NN = dim3(kx_NN, ky_NN);    

    set_values_upper_triangluar_kernel<T, Problem><<<dim_grid_2d_NN, dim_block_2d_NN>>>(N_points, problem, rhs_out, keys, values);
    

    // std::vector<T> values_h(N_points*N_points);
    // std::vector<size_t> keys_h(N_points*N_points);
    // device_2_host_cpy<T>(values_h.data(), values, N_points*N_points);
    // device_2_host_cpy<size_t>(keys_h.data(), keys, N_points*N_points);

    // for(size_t jj=0;jj<N_points*N_points;jj++)
    // {
    //     std::cout << keys_h[jj] << " " << values_h[jj] << std::endl;
    // }


    auto out_size_1 = reduce_by_key<size_t, size_t*, T*>(N_points*N_points, keys, values, keys_out, values_out);

    T whole_integral = detail::sum_reduce<size_t, T, T*>(N_points, rhs_out);
    // std::cout << "out_size_1 = " << out_size_1 << std::endl;

    device_2_host_cpy<T>(rhs_.data(), values_out, N_points);

    std::stringstream ss1;
    ss1 << "equali_distributed_integral_L" << problem.L << "_g" << problem.gamma << "_mu" << problem.mu << "_uo" << problem.u0 << "_s" << problem.sigma << "_d" << problem.delta << ".dat";
    std::fstream f1(ss1.str(), std::ofstream::out);
    for(size_t jj=0;jj<N_points;jj++)
    {
        f1 << t_[jj] << " " << x_[jj] << " " << rhs_[jj]/whole_integral << std::endl;
    }  
    f1.close();
    cudaFree(values);
    cudaFree(keys);
    cudaFree(keys_out);
    cudaFree(values_out);   

    cudaFree(rhs_out);
    cudaFree(basis_points);
    cudaFree(domain_points);

}


}



template<typename T, typename T_vec, class Problem, typename Matrix>
__global__ void form_stiffness_matrix_kernel(size_t N, Problem problem, Matrix A, Matrix iD)
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
        if(problem.delta>problem.delta_threshold)
        {
            A(j,k) = problem.ddpsi_map(k, t_point)/problem.delta/problem.delta - problem.ddddpsi_map(k, t_point);
        }
        else
        {
            A(j,k) = problem.ddpsi_map(k, t_point) - problem.delta*problem.delta*problem.ddddpsi_map(k, t_point);
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
__global__ void form_rhs_linearization_matrix_kernel(size_t N,  Problem problem, T_vec u_solution, Matrix B)
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
        if(problem.delta>problem.delta_threshold)
        {
            B(j,k) /= (problem.delta*problem.delta);
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
__global__ void form_right_hand_side_kernel(size_t N, Problem problem, T_vec u_solution, T_vec res)
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
        if(problem.delta>problem.delta_threshold)
        {
            res[j] /= (problem.delta*problem.delta);
        }
    }
}


//returns -rhs(x,u) without boundary conditoins
template<typename T, typename T_vec, class Problem>
__global__ void get_right_hand_side_kernel(size_t N, Problem problem, T_vec u_solution, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;
    auto x_point = problem.point_in_domain(j);
    auto u_point = u_solution[j];
    res[j] = problem.right_hand_side(x_point, u_point); //we assume that the form is Lu-rhs=0, hence '-'
    // if(problem.delta>problem.delta_threshold)
    // {
    res[j] /= (problem.delta*problem.delta);
    // }
}


template<typename T, typename T_vec, class Problem>
__global__ void form_right_hand_parameter_derivative_kernel(size_t N, Problem problem, T_vec u_solution, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;
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

template<class T, class T_vec, class Problem, bool PointInDomain>
__global__ void get_solution_value_at_point_kernel(size_t N, T x, Problem problem, T_vec u_coeffs, T_vec res)
{
    size_t j=blockDim.x * blockIdx.x + threadIdx.x;
    if(j>=N) return;

    T t;
    if constexpr(PointInDomain)
    {
        t = problem.from_domain_to_basis(x);
    }
    else
    {
        t = x;
    }

    res[j] = problem.psi(j, t)*u_coeffs[j];

}



template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_stiffness_matrix(size_t N, problem_type& problem, matrix_wrap& A, matrix_wrap& iD)
{
        
    dim3 dim_grid, dim_block;
    dim_block = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx=floor(N/( BLOCKSIZE_x ))+1;
    unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
    dim_grid = dim3(kx, ky);


    form_stiffness_matrix_kernel<T, T_vec, problem_type, matrix_wrap><<<dim_grid, dim_block>>>(N, problem, A, iD);
}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_rhs_linearization_matrix(size_t N, problem_type& problem, T_vec& u_solution, matrix_wrap& B)
{
    dim3 dim_grid, dim_block;
    dim_block = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx=floor(N/( BLOCKSIZE_x ))+1;
    unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
    dim_grid = dim3(kx, ky);

    form_rhs_linearization_matrix_kernel<T, T_vec, problem_type, matrix_wrap><<<dim_grid, dim_block>>>(N, problem, u_solution, B);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_mass_matrix(size_t N, problem_type& problem, matrix_wrap& mass_matrix)
{
    dim3 dim_grid, dim_block;
    dim_block = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx=floor(N/( BLOCKSIZE_x ))+1;
    unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
    dim_grid = dim3(kx, ky);    
    form_mass_matrix_kernel<T, T_vec, problem_type, matrix_wrap><<<dim_grid, dim_block>>>(N, problem, mass_matrix);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::calucalte_function_at_basis(size_t N, problem_type& problem, bool mult, T_vec& res)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);

    if(mult)
    {
        calucalte_function_at_basis_kernel<T, T_vec, problem_type, true><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
    }
    else
    {
        calucalte_function_at_basis_kernel<T, T_vec, problem_type, false><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
    }        
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::reformulate_bcs(size_t N, problem_type& problem, T_vec& res)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);

    reformulate_bcs_kernel<T, T_vec, problem_type><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_right_hand_side(size_t N, problem_type& problem, T_vec& u, T_vec& res)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);
    form_right_hand_side_kernel<T, T_vec, problem_type><<<dim_grid_1d, dim_block_1d>>>(N, problem, u, res);
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::get_right_hand_side(size_t N, problem_type& problem, const T_vec& u, T_vec& res)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);
    get_right_hand_side_kernel<T, T_vec, problem_type><<<dim_grid_1d, dim_block_1d>>>(N, problem, u, res);
}



template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::form_right_hand_parameter_derivative(size_t N, problem_type& problem, T_vec& u, T_vec& res)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);

    form_right_hand_parameter_derivative_kernel<T, T_vec, problem_type><<<dim_grid_1d, dim_block_1d>>>(N, problem, u, res);
}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::smooth_random_data(size_t N, T_vec& res, int smooth_steps)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);

    for(int j=0;j<smooth_steps;j++)
    {
        smooth_random_data_kernel<T, T_vec><<<dim_grid_1d, dim_block_1d>>>(N, res);
    }
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::fill_points_at_basis(size_t N,  problem_type& problem, T_vec& res)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);

    fill_points_at_basis_or_domain_kernel<T, T_vec, problem_type, true><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
}
template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::fill_points_at_domain(size_t N,  problem_type& problem, T_vec& res)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);

    fill_points_at_basis_or_domain_kernel<T, T_vec, problem_type, false><<<dim_grid_1d, dim_block_1d>>>(N, problem, res);
}

template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::set_shifted_matrix(size_t N, T a, matrix_wrap& A, T b, matrix_wrap& aApbE)
{
    dim3 dim_grid, dim_block;
    dim_block = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx=floor(N/( BLOCKSIZE_x ))+1;
    unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
    dim_grid = dim3(kx, ky);

    set_shifted_matrix_kernel<T, T_vec, matrix_wrap><<<dim_grid, dim_block>>>(N, a, A, b, aApbE);

}



template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::set_identity_matrix(size_t N, matrix_wrap& E)
{
    dim3 dim_grid, dim_block;
    dim_block = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
    unsigned int kx=floor(N/( BLOCKSIZE_x ))+1;
    unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
    dim_grid = dim3(kx, ky);

    set_identity_matrix_kernel<T, matrix_wrap><<<dim_grid, dim_block>>>(N, E);

}




template<class VecOps, class MatOps>
typename VecOps::scalar_type overscreening_breakdown_ker<VecOps, MatOps>::get_solution_value_at_point(size_t N,  T x, problem_type& problem, const T_vec& u_coeffs, T_vec& temp_storage, bool point_in_domain)
{
    dim3 dim_grid_1d, dim_block_1d;
    dim_block_1d = dim3(BLOCKSIZE);
    unsigned int kx=floor(N/( BLOCKSIZE ))+1;
    dim_grid_1d = dim3(kx);

    if(point_in_domain)
    {
        get_solution_value_at_point_kernel<T, T_vec, problem_type, true><<<dim_grid_1d, dim_block_1d>>>(N, x, problem, u_coeffs, temp_storage);
    }
    else
    {
        get_solution_value_at_point_kernel<T, T_vec, problem_type, false><<<dim_grid_1d, dim_block_1d>>>(N, x, problem, u_coeffs, temp_storage);
    }

    return nonlinear_operators::detail::sum_reduce<size_t, T, T_vec>(N, temp_storage);

}




template<class VecOps, class MatOps>
typename VecOps::scalar_type overscreening_breakdown_ker<VecOps, MatOps>::integrate_solution(size_t N, problem_type& problem, const T_vec& u_coeffs)
{

    // detail::obtain_quadrature_points_wights_values<T, T_vec, problem_type>(N, problem, u_coeffs);

    detail::obtain_equali_distributed_points_wights_values<T, T_vec, problem_type>(N, 1000, 50.0, problem, u_coeffs);



    return 0;
}


template<class VecOps, class MatOps>
void overscreening_breakdown_ker<VecOps, MatOps>::get_exact_solution(size_t N, problem_type& problem, T_vec& res)
{
    //to be implemented
}



}


#endif