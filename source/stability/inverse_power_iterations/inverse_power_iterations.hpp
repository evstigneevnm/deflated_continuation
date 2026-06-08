#ifndef __STABILITY_INVERSE_POWER_ITERATIONS_HPP__
#define __STABILITY_INVERSE_POWER_ITERATIONS_HPP__


#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <numerical_algos/arnolid_process/arnoldi_process.h>
#include <external_libraries/lapack_wrap.h>
#include <stability/inverse_power_iterations/system_operator_stability.h>
#include <stability/inverse_power_iterations/arnoldi_power_iterations.h>

namespace stability
{

template<class VectorOperations, class MatrixOperations,  class NonlinearOperations, class LinearOperator,  class LinearSolver, class Log, class Newton>
class inverse_power_iterations
{

public:
    typedef typename VectorOperations::scalar_type T;
    typedef typename VectorOperations::vector_type T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;

// remove all spesific eigensolver shit out of here.
// we only use eigensolver as an abstract method that returns the set of eigenvalues
// The eigensolver class is passed as the template parameter.
// It should contain the methods:
// void set_linear_operator_stable_eigenvalues_halfplane(int)
// std::vector<std::complex<T>> execute()
// 
private:
    typedef lapack_wrap<T> lapack_t;
    typedef system_operator_stability<VectorOperations, NonlinearOperations, LinearOperator, LinearSolver, Log> sys_op_t;
    typedef numerical_algos::eigen_solvers::arnoldi_process<VectorOperations, MatrixOperations, sys_op_t, Log> arnoldi_proc_t;
    typedef arnoldi_power_iterations<VectorOperations, MatrixOperations, arnoldi_proc_t, lapack_t, Log> arnoldi_pow_t;

    typedef typename arnoldi_pow_t::eigs_t eigs_t;

public:
    inverse_power_iterations(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, Log* log_, NonlinearOperations* nonlin_op_, LinearOperator* lin_op_, LinearSolver* lin_slv_, Newton* newton_):
    vec_ops_l(vec_ops_l_),
    mat_ops_l(mat_ops_l_),
    vec_ops_s(vec_ops_s_),
    mat_ops_s(mat_ops_s_),    
    log(log_),
    nonlin_op(nonlin_op_),
    lin_op(lin_op_),
    lin_slv(lin_slv_),
    newton(newton_)
    {
        small_rows = mat_ops_s->get_rows();
        small_cols = mat_ops_s->get_cols();
        
        lapack = new lapack_t(small_rows);
        sys_op = new sys_op_t(vec_ops_l, nonlin_op, lin_op, lin_slv, log);
        arnoldi_proc = new arnoldi_proc_t(vec_ops_l, vec_ops_s, mat_ops_l, mat_ops_s, sys_op, log);
        arnoldi_pow = new arnoldi_pow_t(vec_ops_l, mat_ops_l, vec_ops_s, mat_ops_s, log, arnoldi_proc, lapack);

        vec_ops_l->init_vector(x_p1); vec_ops_l->start_use_vector(x_p1);
        vec_ops_l->init_vector(x_p2); vec_ops_l->start_use_vector(x_p2);

    }
    ~inverse_power_iterations()
    {
        vec_ops_l->stop_use_vector(x_p1); vec_ops_l->free_vector(x_p1);
        vec_ops_l->stop_use_vector(x_p2); vec_ops_l->free_vector(x_p2);        
        delete lapack;
        delete sys_op;
        delete arnoldi_proc;
        delete arnoldi_pow;

    }

    void set_linear_operator_stable_eigenvalues_halfplane(const T sign_)
    {
        arnoldi_pow->set_linear_operator_stable_eigenvalues_halfplane(sign_);
    }   
    
    void set_target_eigs(std::string& which_)
    {
        which = which_;
    }

    eigs_t execute()
    {

        return 0;
    }





private:
    //passed:
    Log* log;
    VectorOperations* vec_ops_l;
    MatrixOperations* mat_ops_l;
    VectorOperations* vec_ops_s;
    MatrixOperations* mat_ops_s;
    NonlinearOperations* nonlin_op;
    LinearOperator* lin_op; 
    LinearSolver* lin_slv;
    Newton* newton;
//  created_locally:
    lapack_t* lapack = nullptr;
    sys_op_t* sys_op = nullptr;
    arnoldi_proc_t* arnoldi_proc = nullptr;
    arnoldi_pow_t* arnoldi_pow = nullptr;

    size_t small_rows;
    size_t small_cols;

    T_vec x_p1;
    T_vec x_p2;

    std::string which = "LM";

};

}

#endif // __STABILITY_STABILITY_HPP__