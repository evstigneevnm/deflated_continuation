#ifndef __PRECONDITIONER_KS_2D_H__
#define __PRECONDITIONER_KS_2D_H__

template<class vector_operations, class nonlinear_operator> 
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

    void apply(vector_t& x)const
    {
        // we must apply some general framework for the preconditioner?!
        

    }

private:
    nonlinear_operator* nonlin_op;

    
};

#endif