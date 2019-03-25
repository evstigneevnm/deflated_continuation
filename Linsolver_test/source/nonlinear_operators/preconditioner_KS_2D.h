#ifndef __PRECONDITIONER_KS_2D_H__
#define __PRECONDITIONER_KS_2D_H__

namespace nonlinear_operators
{

template<class vector_operations, class nonlinear_operator, class linear_operator> 
class preconditioner_KS_2D
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    preconditioner_KS_2D(nonlinear_operator*& nonlin_op_):
    nonlin_op(nonlin_op_)
    {

    }

    ~preconditioner_KS_2D()
    {

    }
    
    void set_operator(const linear_operator *op_)
    {
        lin_op = (linear_operator*)op_;
    }

    void apply(T_vec& x)const
    {
        nonlin_op->preconditioner_jacobian_u(x);
    }

private:
    nonlinear_operator* nonlin_op;
    linear_operator* lin_op;

    
};


}


#endif