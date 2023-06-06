#ifndef __SYSTEM_OPERATOR_TIME_GLOBALIZATION_H__
#define __SYSTEM_OPERATOR_TIME_GLOBALIZATION_H__

/**
*
*   class that inherits from  system operator to redefine the solve method for time globalization method
*
*
*/

#include "system_operator.h"

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSolver>
class system_operator_time_globalization: public system_operator<VectorOperations, NonlinearOperator, LinearOperator, LinearSolver> 
{
public:

    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    
    system_operator_time_globalization(VectorOperations* vec_ops_, LinearOperator* lin_op_, LinearSolver* lin_solver_):
    system_operator<VectorOperations, NonlinearOperator, LinearOperator, LinearSolver>(vec_ops_, lin_op_, lin_solver_)
    {

    }
    ~system_operator_time_globalization()
    {

    }

    
    
};


}
#endif // __SYSTEM_OPERATOR_TIME_GLOBALIZATION_H__