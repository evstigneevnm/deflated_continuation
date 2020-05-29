// #include <cstdlib>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/gmres.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
// #include <numerical_algos/lin_solvers/bicgstab.h>
#include <common/file_operations.h>
#include <common/cpu_vector_operations.h>



using namespace numerical_algos::lin_solvers;

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
            real *mat_row = &mat[I2(i,0,sz)];
            result[i] = scalar_prod(mat_row, vec);
        }
    }

};

typedef utils::log_std log_t;

typedef default_monitor<
    cpu_vector_operations_real,log_t
        > monitor_t;

typedef bicgstabl<
// typedef gmres<
    system_operator,
    prec_operator,
    cpu_vector_operations_real,
    monitor_t,
    log_t>  lin_solver_c_t;

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
    vector_t A, iP, x, x0, b;

    A=(vector_t) malloc(sz*sz*sizeof(real));
    iP=(vector_t) malloc(sz*sz*sizeof(real));
    x0=(vector_t) malloc(sz*sizeof(real));
    x=(vector_t) malloc(sz*sizeof(real));
    b=(vector_t) malloc(sz*sizeof(real));

    file_operations::read_matrix<real>("./dat_files/A.dat",  sz, sz, A);
    file_operations::read_matrix<real>("./dat_files/iP.dat",  sz, sz, iP);
    //file_operations::read_vector<real>("./dat_files/x0.dat",  sz, x0);
    file_operations::read_vector<real>("./dat_files/b.dat",  sz, b);


    std::cout << "vector size is " << sz << std::endl;
    log_t log;
    log.set_verbosity(3);

    cpu_vector_operations_real vec_ops(sz);
    system_operator Ax(sz, A);
    prec_operator prec(sz, iP);
    
    vec_ops.assign_scalar(real(0.0), x);
    vec_ops.assign_scalar(real(0.0), x0);

    lin_solver_c_t lin_solver_c(&vec_ops, &log);
    monitor_t *mon;
    mon = &lin_solver_c.monitor();
   
    mon->init(rel_tol, real(0.0), max_iters);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    lin_solver_c.set_preconditioner(&prec);
    lin_solver_c.set_use_precond_resid(use_precond_resid);
    lin_solver_c.set_resid_recalc_freq(resid_recalc_freq);
    // lin_solver_c.set_restarts(basis_sz);
    lin_solver_c.set_basis_size(basis_sz);
    
    bool res_flag = lin_solver_c.solve(Ax, b, x);
    if (res_flag) 
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");



    //pseudo newton method to solve the problem iteratively with residual update

    std::cout << "testing pseudo-newton iterative method" << std::endl;
    std::cin.get();

    vector_t r, f, dx;
    r=(vector_t) malloc(sz*sizeof(real));
    f=(vector_t) malloc(sz*sizeof(real));
    dx=(vector_t) malloc(sz*sizeof(real));

    bool use_newton = true;
    int iter = 0;
    if(use_newton)
    {
        vec_ops.assign_scalar(real(0.0), x);
        vec_ops.assign_scalar(real(0.0), dx);
        real resid_norm = real(1.0);

        mon->init(1.0e-1, real(0.0), max_iters);
        real resid_base = vec_ops.norm(b);

        while(resid_norm>rel_tol*resid_base)
        {
            vec_ops.assign_scalar(real(0.0), dx);
            Ax.apply(x,r);
            //calc: z := mul_x*x + mul_y*y
            vec_ops.assign_mul(1.0, b, -1.0, r, f);
            bool res_flag = lin_solver_c.solve(Ax, f, dx);
            if (res_flag) 
                log.info("lin_solver returned success result");
            else
                log.info("lin_solver returned fail result");        
            
            vec_ops.add_mul(1.0, dx, x);
            resid_norm = vec_ops.norm(f);
            std::cout << "residual norm = " << resid_norm << std::endl;
            std::cin.get();
            iter++;
        }
    }

    Ax.apply(x, r);
    vec_ops.add_mul(-1.0, b, r);
    std::cout << "final true residual = " << vec_ops.norm(r) << " for " << iter << " outer iterations." << std::endl;

    file_operations::write_vector<real>("./dat_files/x.dat", sz, x);

    free(dx);
    free(r);
    free(f);
    free(A);
    free(iP);
    free(x0);
    free(b);
    free(x);
    return 0;
}