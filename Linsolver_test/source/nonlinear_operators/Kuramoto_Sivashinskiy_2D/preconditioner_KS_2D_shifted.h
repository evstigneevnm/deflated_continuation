#ifndef __PRECONDITIONER_KS_2D_SHIFTED_H__
#define __PRECONDITIONER_KS_2D_SHIFTED_H__


/**
*   Helper preconditioner class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*   Also requires linear operator while using iterative Krylov solver
*/

#include <utility>

namespace nonlinear_operators
{

template<class vector_operations, class nonlinear_operator, class linear_operator> 
class preconditioner_KS_2D_shifted
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    preconditioner_KS_2D_shifted(nonlinear_operator*& nonlin_op_):
    nonlin_op(nonlin_op_)
    {

    }

    ~preconditioner_KS_2D_shifted()
    {

    }
    //sets a shifted linear operator!
    void set_operator(const linear_operator *op_)const
    {
        lin_op = (linear_operator*)op_;
    }

    void apply(T_vec& x)const
    {
        auto ab = lin_op->get_a_b();
        nonlin_op->preconditioner_jacobian_temporal_u(x, ab.a, ab.b);
    }

private:
    nonlinear_operator* nonlin_op;
    mutable const linear_operator* lin_op;


    
};


}


#endif