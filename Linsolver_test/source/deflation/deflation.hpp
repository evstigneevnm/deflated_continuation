#ifndef __DEFLATION_HPP__
#define __DEFLATION_HPP__


/**
*	 main mart of the deflation-continuation process.
*
*    Deflation class that utilizes single step deflation to find a solution.
*    Accepts knots class and performs deflation on the current value of the knots class.
*	 If no more knots are avaliabe, then the deflation returns false
*/

#include <deflation/system_operator_deflation.h>
#include <deflation/convergence_strategy.h>
#include <deflation/deflation_operator.h>


namespace deflation
{
template<class VectorOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, class Knots, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator>
class deflation
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef Monitor monitor_t;




	deflation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, NonlinearOperations* nonlin_op_, Knots* knots_):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    knots(knots_)
	{


	}
	~deflation()
	{

	}
	
private:
    //passed:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperations* nonlin_op;
    Knots* knots;



};



}
#endif // __DEFLATION_HPP__