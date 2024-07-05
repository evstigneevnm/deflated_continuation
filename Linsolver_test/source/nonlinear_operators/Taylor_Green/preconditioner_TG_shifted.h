#ifndef __PRECONDITIONER_TG_SHIFTED_H__
#define __PRECONDITIONER_TG_SHIFTED_H__


/**
*   Helper preconditioner class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*   Also requires linear operator while using iterative Krylov solver
*/

#include <utility>

namespace nonlinear_operators
{

template<class vector_operations, class nonlinear_operator, class linear_operator> 
class preconditioner_TG_shifted
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    preconditioner_TG_shifted(nonlinear_operator* nonlin_op_):
    nonlin_op(nonlin_op_)
    {

    }

    ~preconditioner_TG_shifted()
    {

    }
    //sets a shifted linear operator!
    void set_operator(const linear_operator *op_)
    {
        lin_op = (linear_operator*)op_;
    }

    void apply(T_vec& x)const
    {
        auto ab = lin_op->get_a_b();
        nonlin_op->preconditioner_jacobian_temporal_u(x, ab.first, ab.second);
    }

private:
    nonlinear_operator* nonlin_op;
    linear_operator* lin_op;


    
};


}


#endif