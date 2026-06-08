#ifndef __LINEAR_OPERATOR_TG_HOMOTOPY_H__
#define __LINEAR_OPERATOR_TG_HOMOTOPY_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*
*/


namespace nonlinear_operators
{


template<class VectorOperations, class NonlinearOperator> 
class linear_operator_TG
{
public:    
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    linear_operator_TG(NonlinearOperator*& nonlin_op_): 
    nonlin_op(nonlin_op_)
    {

    }
    ~linear_operator_TG()
    {

    }

    void apply(const T_vec& x, T_vec& f)const
    {
        nonlin_op->jacobian_u_homotopy(x, f);
    }

private:
    NonlinearOperator* nonlin_op;

};

}

#endif