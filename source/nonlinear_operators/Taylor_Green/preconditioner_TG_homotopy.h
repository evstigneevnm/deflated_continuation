#ifndef __PRECONDITIONER_TG_HOMOTOPY_H__
#define __PRECONDITIONER_TG_HOMOTOPY_H__


/**
*   Helper preconditioner class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*   Also requires linear operator while using iterative Krylov solver
*/


namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator> 
class preconditioner_TG_homotopy
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    preconditioner_TG_homotopy(NonlinearOperator*& nonlin_op_):
    nonlin_op(nonlin_op_)
    {

    }

    ~preconditioner_TG_homotopy()
    {

    }
    
    void set_operator(const LinearOperator *op_)const
    {
        lin_op = (LinearOperator*)op_;
    }

    void apply(T_vec& x)const
    {
        nonlin_op->preconditioner_jacobian_u_homotopy(x);
    }

private:
    NonlinearOperator* nonlin_op;
    mutable const LinearOperator* lin_op;

    
};


}


#endif