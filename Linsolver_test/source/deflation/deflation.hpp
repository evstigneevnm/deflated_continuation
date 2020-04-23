#ifndef __DEFLATION_HPP__
#define __DEFLATION_HPP__


/**
*	 main mart of the deflation-continuation process.
*
*    Deflation class that utilizes single step deflation to find a solution.
*    If 'find_solution' returns false, then no more solutions are avaiable for the provided parameter value
*/

#include <deflation/system_operator_deflation.h>
#include <deflation/convergence_strategy.h>
#include <deflation/deflation_operator.h>


namespace deflation
{

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperations, class LinearOperator, class LinearSolver, class SolutionStorage>
class deflation
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;


    typedef SolutionStorage sol_storage_def_t;

    typedef newton_method_extended::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_def_t;
    
    typedef system_operator_deflation<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        LinearSolver,
        sol_storage_def_t> system_operator_def_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        VectorOperations, 
        NonlinearOperations,
        system_operator_def_t, 
        convergence_newton_def_t, 
        T /* point solution class here instead of real!*/ 
        > newton_def_t;


    typedef deflation_operator<
        VectorOperations,
        newton_def_t,
        NonlinearOperations,
        sol_storage_def_t,
        Log
        >deflation_operator_t;

public:
	deflation(VectorOperations*& vec_ops_, VectorFileOperations*& file_ops_, Log*& log_, NonlinearOperations*& nonlin_op_, LinearOperator*& lin_op_, LinearSolver*& SM_, SolutionStorage*& solution_storage_):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    solution_storage(solution_storage_)
	{
        conv_newton_def = new convergence_newton_def_t(vec_ops, log);
        system_operator_def = new system_operator_def_t(vec_ops, lin_op_, SM_, solution_storage);
        newton_def = new newton_def_t(vec_ops, system_operator_def, conv_newton_def);
        deflation_op = new deflation_operator_t(vec_ops, log, newton_def);
	}
	~deflation()
	{
        delete deflation_op;
        delete newton_def;
        delete system_operator_def;
        delete conv_newton_def;

	}
	
    void set_newton(T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true)
    {
        conv_newton_def->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_,  verbose_);

    }
    void set_max_retries(unsigned int max_retries_)
    {
        deflation_op->set_max_retries(max_retries_);
    }


    bool find_solution(const T& lambda_l)
    {
        bool result = deflation_op->find_solution(lambda_l, nonlin_op);
        return result;
    }
    void get_solution_ref(T_vec& sol_ref_)
    {
        deflation_op->get_solution_ref(sol_ref_);
    }


private:
    //passed:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperations* nonlin_op;
    SolutionStorage* solution_storage;

    //created localy:
    convergence_newton_def_t* conv_newton_def;
    system_operator_def_t* system_operator_def;
    newton_def_t* newton_def;
    deflation_operator_t* deflation_op;

};



}
#endif // __DEFLATION_HPP__