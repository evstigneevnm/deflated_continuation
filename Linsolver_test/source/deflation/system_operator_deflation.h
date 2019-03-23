#ifndef __SYSTEM_OPERATOR_DEFLATION_H__
#define __SYSTEM_OPERATOR_DEFLATION_H__

template<class vector_operations, class nonliner_operator, class linear_operator, class preconditioner, class linear_solver, class sherman_morrison_linear_system_solve>
class system_operator_deflation
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;
    

    system_operator_deflation()
    {
        
    }
    ~system_operator_deflation()
    {

    }
    
};



#endif