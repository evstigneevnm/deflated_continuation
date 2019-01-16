
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <file_operations.h>
#include <cpu_vector_operations.h>
#include <sherman_morrison_linear_system_solve.h>


using namespace numerical_algos::lin_solvers;
using namespace numerical_algos::sherman_morrison_linear_system;

typedef SCALAR_TYPE   real;
typedef real*         vector_t;

typedef cpu_vector_operations<real> cpu_vector_operations_real;

struct system_operator: public cpu_vector_operations_real
{
    int sz;
    vector_t A;
    system_operator(int sz_, vector_t A_) : sz(sz_), A(A_), cpu_vector_operations(sz_)
    {
    }
    void apply(const vector_t& x, vector_t& f)const
    {
        matrix_vector_prod(A, x, f);
         //assign(x, f);
        //file_operations::write_vector<real>("x_test.dat", sz, x);
        //file_operations::write_vector<real>("f_test.dat", sz, f);
    }
private:
    void matrix_vector_prod(const vector_t &mat, const vector_t &vec, vector_t &result)const
    {

        for (int i=0; i<sz; i++)
        {
            real *mat_row = &mat[I2(i,0,sz)];
            result[i] = scalar_prod(mat_row, vec);
        }
    }

};


struct prec_operator:public cpu_vector_operations_real
{
    int sz;
    vector_t iP;
    real* some_vec;
    const system_operator *op;

    

    prec_operator(int sz_, vector_t iP_): sz(sz_), iP(iP_), cpu_vector_operations(sz_)
    {
        
        some_vec=(real*)malloc(sizeof(real)*sz);
        if(some_vec==NULL)
            throw std::runtime_error("prec_operator: error while allocating vector memory");
    }
    ~prec_operator()
    {
        free(some_vec);
    }

    void set_operator(const system_operator *op_)
    {
        op=op_;
    }

    void apply(vector_t& x)const
    {
       matrix_vector_prod(iP, x, some_vec);
       assign(some_vec, x);
    }

private:

    void matrix_vector_prod(const vector_t &mat, vector_t &vec, real *result)const
    {
        
        for (int i=0; i<sz; i++)
        {
            result[i] = scalar_prod(&mat[I2(i,0,sz)], vec);
        }
    }

};

typedef utils::log_std                                                                  log_t;
typedef default_monitor<cpu_vector_operations_real,log_t>                                    monitor_t;
typedef bicgstabl<system_operator,prec_operator,cpu_vector_operations_real,monitor_t,log_t>  lin_solver_bicgstabl_t;
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

    int sz = file_operations::read_matrix_size("A.dat");
    vector_t A, iP, x, x0, b, c, d;

    A=(vector_t) malloc(sz*sz*sizeof(real));
    iP=(vector_t) malloc(sz*sz*sizeof(real));

    x0=(vector_t) malloc(sz*sizeof(real));
    x=(vector_t) malloc(sz*sizeof(real));
    c=(vector_t) malloc(sz*sizeof(real));
    d=(vector_t) malloc(sz*sizeof(real));

    b=(vector_t) malloc(sz*sizeof(real));
    real alpha=1.0/100.0;
    real beta=1.0;
    real v=0.0;

    file_operations::read_matrix<real>("A.dat",  sz, sz, A);
    file_operations::read_matrix<real>("iP.dat",  sz, sz, iP);
    file_operations::read_vector<real>("x0.dat",  sz, x0);
    file_operations::read_vector<real>("b.dat",  sz, b);
    file_operations::read_vector<real>("c.dat",  sz, c);
    file_operations::read_vector<real>("d.dat",  sz, d);


    std::cout << sz << std::endl;
    log_t log;
    cpu_vector_operations_real vec_ops(sz);
    system_operator Ax(sz, A);
    prec_operator prec(sz, iP);
    
    vec_ops.assign_scalar(0.0, x);


    sherman_morrison_linear_system_solve_t SM(&prec, &vec_ops, &log);
    
    //lin_solver_bicgstabl_t lin_solver_bicgstabl(&vec_ops, &log);
    monitor_t *mon;
    //mon = &lin_solver_bicgstabl.monitor();
    mon = &SM.get_linsolver_handle()->monitor();

    mon->init(rel_tol, real(0.f), max_iters);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    //lin_solver_bicgstabl.set_preconditioner(&prec);
    //SM.get_linsolver_handle()->set_preconditioner(&prec);



    //lin_solver_bicgstabl.set_use_precond_resid(use_precond_resid);
    //lin_solver_bicgstabl.set_resid_recalc_freq(resid_recalc_freq);
    //lin_solver_bicgstabl.set_basis_size(basis_sz);
    SM.get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
    SM.get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
    SM.get_linsolver_handle()->set_basis_size(basis_sz);

    //bool res_flag = lin_solver_bicgstabl.solve(Ax, b, x);
    //bool res_flag = SM.solve(Ax, b, x);
    bool res_flag = SM.solve(Ax, c, d, alpha, b, beta, x, v);
    int iters_performed = mon->iters_performed();

    if (res_flag) 
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");


    file_operations::write_vector<real>("x.dat", sz, x);
    std::cout << "v=" << v << std::endl;

    free(A);
    free(iP);
    free(x0);
    free(b);
    free(c);
    free(d);
    free(x);
    return 0;
}