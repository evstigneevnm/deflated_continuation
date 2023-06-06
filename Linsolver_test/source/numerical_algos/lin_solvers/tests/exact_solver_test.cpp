#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <utils/log.h>
#include <utils/init_cuda.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/cusolver_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>


template<class VectorOperations, class MatrixOperations>
struct linear_operator
{
    using vector_operations_type = VectorOperations;
    using matrix_operations_type = MatrixOperations;
    using scalar_type = typename vector_operations_type::scalar_type;
    using vector_type = typename vector_operations_type::vector_type;
    using matrix_type = typename matrix_operations_type::matrix_type;

    linear_operator(VectorOperations* vec_ops_p, MatrixOperations* mat_ops_p, matrix_type& A_p): 
    vec_ops_(vec_ops_p),
    mat_ops_(mat_ops_p),
    A_(A_p),
    is_constant_matrix_(true)
    {}


    void apply(const vector_type& x, vector_type& f)const
    {
        mat_ops_->gemv('N', A_, 1.0, x, 0.0, f);
    }
    matrix_type& get_matrix_ref()const
    {
        return A_;
    }
    void set_constant_matrix(bool is_constant_matrix_p)const
    {
        is_constant_matrix_ = is_constant_matrix_p;
    }
    bool is_constant_matrix()const
    {
        return is_constant_matrix_;
    }

private:
    mutable matrix_type A_;
    vector_operations_type* vec_ops_;
    matrix_operations_type* mat_ops_;
    mutable bool is_constant_matrix_;
};


template<class VectorOperations, class LinearOperator>
struct prec_operator
{
    using vector_operations_type = VectorOperations;
    using scalar_type = typename LinearOperator::scalar_type;
    using vector_type = typename LinearOperator::vector_type;
    using matrix_type = typename LinearOperator::matrix_type;

    prec_operator(VectorOperations* vec_ops_p, cusolver_wrap* cusolver_p):
    vec_ops_(vec_ops_p),
    cusolver_(cusolver_p)
    {
        vec_ops_->init_vector(b_);
        vec_ops_->start_use_vector(b_);
        N_ = vec_ops_->get_vector_size();
    }

    ~prec_operator()
    {
        vec_ops_->stop_use_vector(b_);
        vec_ops_->free_vector(b_);
    }

    void set_operator(const LinearOperator *op_p)
    {
        op_ = op_p;
    }

    void apply(vector_type& x)const
    {
        if( op_->is_constant_matrix() )
        {
            vec_ops_->assign(x, b_);
            cusolver_->gesv(N_, op_->get_matrix_ref(), b_, x);
        }
        else
        {
            cusolver_->gesv(N_, op_->get_matrix_ref(), x);
        }
    }

private:
    vector_type b_;
    const LinearOperator *op_;
    vector_operations_type* vec_ops_;
    cusolver_wrap* cusolver_;
    size_t N_;

};


int main(int argc, char **args)
{

    if(argc != 2)
    {
        std::cout << "USAGE: " << std::string(args[0]) << " <use_small_alpha_optimization>"  << std::endl;
        return 0;        
    }

    using real = SCALAR_TYPE;

    using vec_ops_t = gpu_vector_operations<real>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using vec_files_ops_t = gpu_file_operations<vec_ops_t>;
    using mat_files_ops_t = gpu_matrix_file_operations<mat_ops_t>;

    using lin_op_t = linear_operator<vec_ops_t, mat_ops_t>;
    using prec_t = prec_operator<vec_ops_t, lin_op_t>;

    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t, log_t>;
    
    using linsolver_t = numerical_algos::lin_solvers::exact_wrapper<lin_op_t, prec_t, vec_ops_t, monitor_t, log_t>;
    using sm_linsolver_t = numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<lin_op_t, prec_t, vec_ops_t, monitor_t, log_t, numerical_algos::lin_solvers::exact_wrapper >;

    bool use_small_alpha = static_cast<bool>(std::stoi(args[1]));
    real rel_tol = 1.0e-10;
    size_t max_iters = 2;

    utils::init_cuda(-1);
    cublas_wrap cublas(true);
    cusolver_wrap cusolver(&cublas);

    size_t sz = file_operations::read_matrix_size("./dat_files/A.dat");
    T_mat A;
    T_vec x, x0, b, c, d, r;
    vec_ops_t vec_ops( sz, &cublas );
    mat_ops_t mat_ops( sz, sz, vec_ops.get_cublas_ref() );
    vec_files_ops_t vec_files_ops(&vec_ops);
    mat_files_ops_t mat_files_ops(&mat_ops);

    mat_ops.init_matrix(A); 
    mat_ops.start_use_matrix(A);
    vec_ops.init_vectors(x0, x, c, d, b, r);
    vec_ops.start_use_vectors(x0, x, c, d, b, r);

    real alpha=1.0e-9;
    real beta=1.9;
    real v=0.0;

    mat_files_ops.read_matrix("./dat_files/A.dat", A);
        
    vec_files_ops.read_vector("./dat_files/x0.dat", x0);
    vec_files_ops.read_vector("./dat_files/b.dat", b);
    vec_files_ops.read_vector("./dat_files/c.dat", c);
    vec_files_ops.read_vector("./dat_files/d.dat", d);

    std::cout << "matrix size = " << sz << std::endl;

    log_t log;

    lin_op_t Ax(&vec_ops, &mat_ops, A);
    prec_t prec(&vec_ops, &cusolver);

    
    vec_ops.assign_scalar(0.0, x);
    
    
    Ax.set_constant_matrix(true);

    linsolver_t linsolver(&vec_ops);
    linsolver.set_preconditioner(&prec);
    linsolver.solve(Ax, b, x);
    vec_files_ops.write_vector("./dat_files/x.dat", x);
    vec_ops.assign(b, r); //b->r
    mat_ops.gemv('N', A, static_cast<real>(1.0), x, static_cast<real>(-1.0), r);
    log.info_f("residual norm = %e", vec_ops.norm(r) );
    vec_ops.assign_mul(static_cast<real>(1.0), x, static_cast<real>(-1.0), x0, r);
    log.info_f("solution difference norm = %e", vec_ops.norm(r) );


    sm_linsolver_t sm_linsolver(&prec, &vec_ops, &log);
    monitor_t* mon;
    monitor_t* mon_original;
    mon = &sm_linsolver.get_linsolver_handle()->monitor();
    mon_original = &sm_linsolver.get_linsolver_handle_original()->monitor();

    mon->init(rel_tol, real(0), max_iters);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    mon_original->init(rel_tol, real(0), max_iters);
    mon_original->set_save_convergence_history(true);
    mon_original->set_divide_out_norms_by_rel_base(true);

    bool res_flag;
    int iters_performed;

    sm_linsolver.is_small_alpha(use_small_alpha);

    std::cout << "\n ========= \ntesting: rank1_update(A)u = b; v=f(u,b) \n";
    res_flag = sm_linsolver.solve(Ax, c, d, alpha, b, beta, x, v);
    iters_performed = mon->iters_performed();

    if (res_flag) 
    {
        log.info("lin_solver returned success result");
    }
    else
    {
        log.error("lin_solver returned fail result");
    }
    vec_files_ops.write_vector("./dat_files/x_rank1_exact.dat", x);
    std::cout << "v = " << v << std::endl;

    // vec_ops.add_mul(1.0, x0, -1.0, x);
    // std::cout << "||x-x0|| = " << vec_ops.norm(x);

    //test (beta A - 1/alpha d c^T) u = b;
    std::cout << "\n ========= \ntesting: (beta A - 1/alpha d c^T) u = b \n";
    res_flag = sm_linsolver.solve(beta, Ax, alpha, c, d, b, x);
    
    iters_performed = mon->iters_performed();

    if (res_flag)
        log.info("lin_solver returned success result");
    else
        log.error("lin_solver returned fail result");    
    
    // vec_ops.add_mul(1.0, x0, -1.0, x);
    // std::cout << "||x-x0|| = " << vec_ops.norm(x);

    vec_files_ops.write_vector("./dat_files/x_sm_exact.dat", x);

//     std::cout << "\n ========= \ntesting: A u = b \n";
//     res_flag = SM->solve(Ax, b, x);
//     iters_performed = mon->iters_performed();
//     if (res_flag)
//         log.info("lin_solver returned success result");
//     else
//         log.error("lin_solver returned fail result");    
  

//     file_operations::write_vector<real>("./dat_files/x_orig.dat", sz, x);
            

    
    mat_ops.stop_use_matrix(A);
    mat_ops.free_matrix(A); 
    vec_ops.stop_use_vectors(x0, x, c, d, b, r);
    vec_ops.free_vectors(x0, x, c, d, b, r);

    return 0;
}