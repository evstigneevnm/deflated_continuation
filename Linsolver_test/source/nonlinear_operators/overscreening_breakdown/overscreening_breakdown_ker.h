#ifndef __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_H__
#define __NONLINEAR_PROBLEM_OVERSCREENING_BREAKDOWN_KER_H__

#include <external_libraries/cusolver_wrap.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <utils/device_tag.h>

namespace nonlinear_operators
{

    /*
        0x7f800000 = infinity

        0xff800000 = -infinity

        0x7ff0000000000000 = infinity

        0xfff0000000000000 = -infinity
     */
    // infinit spec
template <class T>
__DEVICE_TAG__ inline T cuda_infinity()
{}

template <>
__DEVICE_TAG__ inline float cuda_infinity()
{
    return 0x7f800000;
}
template <>
__DEVICE_TAG__ inline double cuda_infinity()
{
    return 0x7ff0000000000000;
}

const unsigned int BLOCKSIZE_x = 32;
const unsigned int BLOCKSIZE_y = 32;


template<class VectorOperations, class MatrixOperations>
class overscreening_breakdown_ker
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using vec_file_ops_t = gpu_file_operations<VectorOperations>;
    using mat_ops_t = MatrixOperations;
    using mat_file_ops_t = gpu_matrix_file_operations<mat_ops_t>;
    using T_mat = typename MatrixOperations::matrix_type;


    //struct under external managment!
    struct matrix_device_s
    {
        matrix_device_s(size_t rows_p, size_t cols_p):
        rows(rows_p), cols(cols_p), data(nullptr)
        {}

        __DEVICE_TAG__ inline T& operator()(size_t j, size_t k)
        {
            // return data[cols*j+k];
            return data[rows*k+j];
        }
        size_t rows;
        size_t cols;
        T_mat data;        
    };


    struct problem_s
    {
        problem_s(size_t N_p, T L_p, T gamma_p, T mu_p, T u0_p ):
        N(N_p), L(L_p), gamma(gamma_p), mu(mu_p), u0(u0_p), sigma(0.0), which(0)
        {}

        void set_state(T sigma_p)
        {
            sigma = sigma_p;
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

        __DEVICE_TAG__ inline T right_hand_side_linearization(T x, T u)
        {
            T num = (2.0*(gamma + cosh(u) - gamma*cosh(u)) + (exp(u)*(-1.0 + gamma) - gamma)*mu*g_func(x));
            T din = 2.0*(1.0 - gamma + gamma*cosh(u))*(1.0 - gamma + gamma*cosh(u));
            return num/din;
        }

        __DEVICE_TAG__ inline T right_hand_side(T x, T u)
        {
            T num = sinh(u) - g_func(x)*0.5*mu*exp(u);
            T din = (1 + 2.0*gamma*sinh(u/2.0)*sinh(u/2.0) );
            return num/din;

        }
        __DEVICE_TAG__ inline T right_hand_side_parameter_derivative(T x, T u)
        {
            if(sigma > 0)
            {
                return -((exp(u-x*x/(2.0*sigma*sigma))*x*x*mu)/(2.0*sqrt(2.0*M_PI)*sigma*sigma*sigma*(1.0+2.0*gamma*sinh(0.5*u)*sinh(0.5*u))));
            }
            else
            {
                return -0.0;
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
                return cuda_infinity<T>();
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

        T L, gamma, mu, u0, sigma;
        size_t N;
        char which;
        const T delta_threshold = 1.0;
    };


    
    template<class Params>
    overscreening_breakdown_ker(VectorOperations* vec_ops, MatrixOperations* mat_ops, const Params& params):
    vec_ops_(vec_ops), mat_ops_(mat_ops),
    cusolver_(new cusolver_wrap(true) ),
    apply_inverse_diag_(true)
    {
        cusolver_->set_cublas(vec_ops_->get_cublas_ref() );
        N = vec_ops_->get_vector_size();
        delta = params.delta;
        mat_file_ops_ = new mat_file_ops_t(mat_ops_);
        vec_file_ops_ = new vec_file_ops_t(vec_ops_);

        linear_operator = new matrix_device_s( N, N );
        mass_matrix = new matrix_device_s( N, N );
        stiffness_matrix = new matrix_device_s( N, N );
        shifted_linear_operator = new matrix_device_s( N, N );
        eye = new matrix_device_s(N, N);
        iD = new matrix_device_s(N, N);
        iDA = new matrix_device_s(N, N);

        problem = new problem_s(N, params.L, params.gamma, params.mu, params.u0);
        
        mat_ops_->init_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        mat_ops_->start_use_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        
        vec_ops_->init_vectors(u_solution, u_randomized, iDb);
        vec_ops_->start_use_vectors(u_solution, u_randomized, iDb);

        dim_block = dim3(BLOCKSIZE_x, BLOCKSIZE_y);
        unsigned int kx=floor(N/( BLOCKSIZE_x ))+1;
        unsigned int ky=floor(N/( BLOCKSIZE_y ))+1;
        dim_grid = dim3(kx, ky);

        dim_block_1d = dim3(BLOCKSIZE_x);
        dim_grid_1d = dim3(kx);

        form_mass_matrix(N, *problem, *mass_matrix);
        form_stiffness_matrix(N, delta, *problem, *stiffness_matrix, *iD);
        form_rhs_linearization_matrix(N, *problem, delta, u_solution, *linear_operator);
        set_identity_matrix(N, *eye);
        

        // cudaDeviceSynchronize();
        // mat_file_ops_->write_matrix("M.dat", mass_matrix->data);        
        // mat_file_ops_->write_matrix("S.dat", stiffness_matrix->data);
        // mat_file_ops_->write_matrix("N.dat", linear_operator->data); 
        // mat_file_ops_->write_matrix("E.dat", eye->data);   

    }
    ~overscreening_breakdown_ker()
    {
        mat_ops_->stop_use_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        mat_ops_->free_matrices(linear_operator->data, mass_matrix->data, stiffness_matrix->data, shifted_linear_operator->data, eye->data, iD->data, iDA->data);
        vec_ops_->stop_use_vectors(u_solution, u_randomized, iDb);
        vec_ops_->free_vectors(u_solution, u_randomized, iDb); 
        delete iDA;
        delete iD;
        delete eye;
        delete shifted_linear_operator;
        delete stiffness_matrix;
        delete mass_matrix;
        delete linear_operator;
        delete problem;
        delete mat_file_ops_;
        delete vec_file_ops_;
        delete cusolver_;
    }

    void get_solution_at_basis_points(const T_vec u, T_vec u_solution_p)
    {
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution_p);
    }

//  jacobian w.r.t. the expansion coefficients.
    void form_jacobian_operator(const T_vec u, const T sigma)
    {
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution);
        reformulate_bcs(N, *problem, u_solution);
        problem->set_state(sigma);
        form_rhs_linearization_matrix(N, *problem, sigma, u_solution, *linear_operator);
        // mat_file_ops_->write_matrix("N.dat", linear_operator->data); 
        mat_ops_->geam('N', N, N, 1.0, stiffness_matrix->data, -1.0, linear_operator->data, linear_operator->data);
        // mat_file_ops_->write_matrix("J.dat", linear_operator->data);
        // vec_file_ops_->write_vector("exp1.dat", u_solution);
    }
    T_mat& get_jacobi_matrix()
    {
        return linear_operator->data;
    }
    void apply_jacobi_matrix(const T_vec& x, T_vec& f) const
    {
        mat_ops_->gemv('N', linear_operator->data, 1.0, x, 0.0, f);
    }


// returns F(u, sigma), where F is the whole problem.
    void form_operator(const T_vec u, const T sigma_p, T_vec res)
    {
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution);
        reformulate_bcs(N, *problem, u_solution);
        problem->set_state(sigma_p);  
        form_right_hand_side(N, *problem, sigma_p, u_solution, res);
        mat_ops_->gemv('N', stiffness_matrix->data, 1.0, u, 1.0, res);
        // vec_file_ops_->write_vector("Fu.dat", res);

    }

//  derivative of the RHS w.r.t. sigma:
//  -((E^(u-x^2/(2 \[Sigma]^2)) x^2 \[Mu])/(2 Sqrt[2 \[Pi]] \[Sigma]^3 (1+2 \[Gamma] Sinh[u/2]^2)))
//  -((exp(u-x^2/(2*sigma^2))*x^2*mu)/(2*sqrt(2*M_PI)*sigma^3*(1+2*gamma*sinh(u/2)^2)))
//  -((exp(u-x*x/(2.0*sigma*sigma))*x*x*mu)/(2.0*sqrt(2.0*M_PI)*sigma*sigma*sigma*(1.0+2.0*gamma*sinh(0.5*u)*sinh(0.5*u))))
    void form_operator_parameter_derivative(const T_vec u, const T sigma_p, T_vec res)
    {
        
        mat_ops_->gemv('N', mass_matrix->data, 1.0, u, 0.0, u_solution);
        reformulate_bcs(N, *problem, u_solution);
        problem->set_state(sigma_p);
        form_right_hand_parameter_derivative(N, *problem, sigma_p, u_solution, res);
        // vec_file_ops_->write_vector("Fu_sigma.dat", res);
    }

    void calucalte_function_at_basis(T_vec coeffs, bool mult = false)
    {
        calucalte_function_at_basis(N, *problem, mult, coeffs);
        // vec_file_ops_->write_vector("exp.dat", coeffs);
    }
    void expend_function(const T_vec u_at_points, T_vec coeffs)
    {
        cusolver_->gesv(N, mass_matrix->data, u_at_points, coeffs); // mass matrix is not changed.
        // mat_file_ops_->write_matrix("M1.dat", mass_matrix->data); 
        // vec_file_ops_->write_vector("exp_coeffs.dat", coeffs);
    }
    void expend_function(T_vec coeffs)
    {
        cusolver_->gesv(N, mass_matrix->data, coeffs); // mass matrix is changed.
        form_mass_matrix(N, *problem, *mass_matrix);   //regenerate matrix
        // mat_file_ops_->write_matrix("M1.dat", mass_matrix->data); 
        // vec_file_ops_->write_vector("exp_coeffs.dat", coeffs);
    }

//  apply inverse diags in these functions
    void solve_linear_system(const T_mat& matrix, const T_vec& b, T_vec& x)
    {
        if(apply_inverse_diag_)
        {
            // iDA, iDb
            mat_ops_->mat2column_mult_mat(iD->data, matrix, N, 1.0, 0.0, iDA->data);
            mat_ops_->gemv('N', iD->data, 1.0, b, 0.0, iDb);
            cusolver_->gesv(N, iDA->data, iDb, x);
        }
        else
        {    
            cusolver_->gesv(N, matrix, b, x);
        }
    }
    void solve_linear_system(const T alpha, const T_mat& matrix, const T beta, const T_vec& b, T_vec& x)
    {         
        mat_ops_->geam('N', N, N, alpha, matrix, beta, eye->data, shifted_linear_operator->data);
        // mat_file_ops_->write_matrix("A.dat", matrix);
        // mat_file_ops_->write_matrix("aApbE.dat", shifted_linear_operator->data);
        cusolver_->gesv(N, shifted_linear_operator->data, b, x);
    }

    void solve_linear_system(T_mat& matrix, T_vec& b2x)
    {
        if(apply_inverse_diag_)
        {
            // iDA, iDb
            mat_ops_->mat2column_mult_mat(iD->data, matrix, N, 1.0, 0.0, iDA->data);
            mat_ops_->gemv('N', iD->data, 1.0, b2x, 0.0, iDb);
            cusolver_->gesv(N, iDA->data, iDb);
        }
        else
        {  
            cusolver_->gesv(N, matrix, b2x);
        }
    }
    void solve_linear_system(const T alpha, T_mat& matrix, const T beta, T_vec& b2x)
    {
        
        mat_ops_->geam('N', N, N, alpha, matrix, beta, eye->data, shifted_linear_operator->data);
        cusolver_->gesv(N, shifted_linear_operator->data, b2x);
    }



    void set_random_smoothed_data(T_vec coeffs, int steps)
    {
        vec_ops_->assign_random(coeffs, 0.5, 1.5);
        smooth_random_data(N, coeffs, steps);
        calucalte_function_at_basis(coeffs, true);
        expend_function(coeffs);
        problem->rotate_initial_function(); //switch to another predefined function
    }


    void fill_points_at_basis(T_vec res)
    {
        fill_points_at_basis(N, *problem, res);
    }
    void fill_points_at_domain(T_vec res)
    {
        fill_points_at_domain(N, *problem, res);
    }    
     

private:
    problem_s* problem;
    matrix_device_s* stiffness_matrix;
    matrix_device_s* linear_operator;
    matrix_device_s* mass_matrix;
    matrix_device_s* shifted_linear_operator;
    matrix_device_s* eye;
    matrix_device_s* iD;
    matrix_device_s* iDA;

    bool apply_inverse_diag_;
    T_vec u_solution;
    T_vec u_randomized;
    T_vec iDb;
    VectorOperations* vec_ops_;
    cusolver_wrap* cusolver_;
    mat_ops_t* mat_ops_;
    vec_file_ops_t* vec_file_ops_;
    mat_file_ops_t* mat_file_ops_;
    dim3 dim_grid, dim_block;
    dim3 dim_grid_1d, dim_block_1d;
    T delta;
    size_t N;

    void form_stiffness_matrix(size_t N, T delta, problem_s problem, matrix_device_s A, matrix_device_s iD);
    void form_rhs_linearization_matrix(size_t N, problem_s problem, T delta, T_vec u_solution, matrix_device_s B);
    void form_mass_matrix(size_t N, problem_s problem, matrix_device_s mass_matrix);
    void form_jacobian_matrix(size_t N, T delta, problem_s problem, matrix_device_s A);
    void calucalte_function_at_basis(size_t N, problem_s problem, bool mult, T_vec res);
    void reformulate_bcs(size_t N, problem_s problem, T_vec res);
    void form_right_hand_parameter_derivative(size_t N, problem_s problem, T delta, T_vec u, T_vec res);
    void form_right_hand_side(size_t N, problem_s problem, T delta, T_vec u, T_vec res);
    void smooth_random_data(size_t N, T_vec res, int smooth_steps);
    void fill_points_at_basis(size_t N,  problem_s problem, T_vec res);
    void fill_points_at_domain(size_t N,  problem_s problem, T_vec res);
    void set_shifted_matrix(size_t N, T a, matrix_device_s A, T b, matrix_device_s aApbE);
    void set_identity_matrix(size_t N, matrix_device_s E);
};

}

#endif