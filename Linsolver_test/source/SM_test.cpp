#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
//#include <numerical_algos/lin_solvers/bicgstab.h>
#include <file_operations.h>
#include <cpu_vector_operations.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>


using namespace numerical_algos::lin_solvers;
using namespace numerical_algos::sherman_morrison_linear_system;

typedef SCALAR_TYPE   real;
typedef real*         vector_t;

typedef cpu_vector_operations<real> cpu_vector_operations_real;

struct system_operator: public cpu_vector_operations_real
{
    int sz;
    vector_t A;
    system_operator(int sz_, vector_t& A_): 
    sz(sz_), 
    A(A_), 
    cpu_vector_operations(sz_)
    {
    }

    void apply(const vector_t& x, vector_t& f)const
    {
        matrix_vector_prod(x, f);
        //assign(x, f);
    }
private:
    void matrix_vector_prod(const vector_t &vec, vector_t &result)const
    {

        for (int i=0; i<sz; i++)
        {
            real *mat_row = &A[I2(i, 0, sz)];
            result[i] = scalar_prod(mat_row, vec);
        }
    }

};


struct prec_operator:public cpu_vector_operations_real
{
    int sz;
    vector_t iP;
    vector_t some_vec;
    const system_operator *op;

    

    prec_operator(int sz_, vector_t& iP_): 
    sz(sz_), 
    iP(iP_), 
    cpu_vector_operations(sz_)
    {
        init_vector(some_vec); start_use_vector(some_vec);
    }
    ~prec_operator()
    {
        stop_use_vector(some_vec); free_vector(some_vec);
    }

    void set_operator(const system_operator *op_)
    {
        op = op_;
    }

    void apply(vector_t& x)const
    {
       matrix_vector_prod(x, some_vec);
       assign(some_vec, x);
    }

private:

    void matrix_vector_prod(vector_t &vec, real *result)const
    {
        
        for (int i=0; i<sz; i++)
        {
            real *mat_row = &iP[I2(i,0,sz)];
            result[i] = scalar_prod(mat_row, vec);
        }
    }

};

typedef utils::log_std log_t;
typedef default_monitor<cpu_vector_operations_real,log_t> monitor_t;
//typedef bicgstabl<system_operator,prec_operator,cpu_vector_operations_real,monitor_t,log_t> lin_solver_bicgstabl_t;
// Sherman Morrison class
typedef sherman_morrison_linear_system_solve<system_operator,prec_operator,cpu_vector_operations_real,monitor_t,log_t,bicgstabl> sherman_morrison_linear_system_solve_t;

int main(int argc, char **args)
{
    if (argc != 6) {
        std::cout << "USAGE: " << std::string(args[0]) << " <maximum iterations> <relative tolerance> <use preconditioned residual> <residual recalculation frequency> <basis size>"  << std::endl;
        return 0;
    }

    int max_iters = atoi(args[1]);
    real rel_tol = atof(args[2]);
    bool use_precond_resid = atoi(args[3]);
    int resid_recalc_freq = atoi(args[4]);
    int basis_sz = atoi(args[5]);

    int sz = file_operations::read_matrix_size("./dat_files/A.dat");
    vector_t A, iP, x, x0, b, c, d;
    cpu_vector_operations_real vec_ops(sz);


    A=(vector_t) malloc(sz*sz*sizeof(real));
    iP=(vector_t) malloc(sz*sz*sizeof(real));
    x0=(vector_t) malloc(sz*sizeof(real));
    x=(vector_t) malloc(sz*sizeof(real));
    c=(vector_t) malloc(sz*sizeof(real));
    d=(vector_t) malloc(sz*sizeof(real));
    b=(vector_t) malloc(sz*sizeof(real));

    real alpha=1.0/1.0;
    real beta=1.9;
    real v=0.0;

    file_operations::read_matrix<real>("./dat_files/A.dat",  sz, sz, A);
    file_operations::read_matrix<real>("./dat_files/iP.dat",  sz, sz, iP);
    //file_operations::read_vector<real>("./dat_files/x0.dat",  sz, x0);
    file_operations::read_vector<real>("./dat_files/b.dat",  sz, b);
    file_operations::read_vector<real>("./dat_files/c.dat",  sz, c);
    file_operations::read_vector<real>("./dat_files/d.dat",  sz, d);


    std::cout << sz << std::endl;
    log_t log;
    system_operator Ax(sz, A);
    prec_operator prec(sz, iP);
    
    vec_ops.assign_scalar(1.0, x);
    monitor_t *mon;
    monitor_t *mon_original;
    
    //lin_solver_bicgstabl_t lin_solver_bicgstabl(&vec_ops, &log);
    //lin_solver_bicgstabl.set_preconditioner(&prec);
    //mon = &lin_solver_bicgstabl.monitor();

    sherman_morrison_linear_system_solve_t *SM = new sherman_morrison_linear_system_solve_t(&prec, &vec_ops, &log);   
    mon = &SM->get_linsolver_handle()->monitor();
    mon_original = &SM->get_linsolver_handle_original()->monitor();

    mon->init(rel_tol, real(0), max_iters);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    mon_original->init(rel_tol, real(0), max_iters);
    mon_original->set_save_convergence_history(true);
    mon_original->set_divide_out_norms_by_rel_base(true);


    bool res_flag;
    int iters_performed;

    // lin_solver_bicgstabl.set_use_precond_resid(use_precond_resid);
    // lin_solver_bicgstabl.set_resid_recalc_freq(resid_recalc_freq);
    // lin_solver_bicgstabl.set_basis_size(basis_sz);
    // bool res_flag = lin_solver_bicgstabl.solve(Ax, b, x);

    SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle()->set_basis_size(basis_sz);
    SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
    SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
    SM->get_linsolver_handle_original()->set_basis_size(basis_sz);


    std::cout << "\n ========= \ntesting: rank1_update(A)u = b; v=f(u,b) \n";
    res_flag = SM->solve(Ax, c, d, alpha, b, beta, x, v);
    iters_performed = mon->iters_performed();

    if (res_flag) 
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");

    file_operations::write_vector<real>("./dat_files/x_r1.dat", sz, x);
    std::cout << " v = " << v << std::endl;

    //test (beta A - 1/alpha d c^T) u = b;
    std::cout << "\n ========= \ntesting: (beta A - 1/alpha d c^T) u = b \n";
    res_flag = SM->solve(beta, Ax, alpha, c, d, b, x);
    
    iters_performed = mon->iters_performed();

    if (res_flag)
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");    
  

    file_operations::write_vector<real>("./dat_files/x_sm.dat", sz, x);

    std::cout << "\n ========= \ntesting: A u = b \n";
    res_flag = SM->solve(Ax, b, x);
    iters_performed = mon->iters_performed();
    if (res_flag)
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");    
  

    file_operations::write_vector<real>("./dat_files/x_orig.dat", sz, x);
            

    free(A);
    free(iP);
    free(x0);
    free(b);
    free(c);
    free(d);
    free(x);
    return 0;
}