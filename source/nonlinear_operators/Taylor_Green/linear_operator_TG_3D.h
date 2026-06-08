#ifndef __LINEAR_OPERATOR_TG_H__
#define __LINEAR_OPERATOR_TG_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*
*/


namespace nonlinear_operators
{


template<class vector_operations, class nonlinear_operator> 
class linear_operator_TG
{
public:    
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    linear_operator_TG(nonlinear_operator*& nonlin_op_): 
    nonlin_op(nonlin_op_)
    {

    }
    ~linear_operator_TG()
    {

    }

    void apply(const T_vec& x, T_vec& f)const
    {
        nonlin_op->jacobian_u(x, f);
    }

private:
    nonlinear_operator* nonlin_op;

};

}

#endif