#ifndef __DEFLATION_CONTINUATION_HPP__
#define __DEFLATION_CONTINUATION_HPP__

/**
*	The main class of the whole Deflation-Continuaiton Process (DCP). 
*
*	It uses nonlinear operator and other set options to configure the whole project.
*	After vector and file operations, the nonlinear operator, log and monitor are configured,
*   this class is initialized and configured to perform the whole DCP.
*/

template<class VectorOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator>
class deflation_continuation
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef Monitor monitor_t;


	deflation_continuation()
	{

	}
	~deflation_continuation()
	{

	}



	
};




#endif // __DEFLATION_CONTINUATION_HPP__