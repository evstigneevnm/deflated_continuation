#ifndef __PRECONDITIONER_TG_STIFF_H__
#define __PRECONDITIONER_TG_STIFF_H__


/**
*   Helper preconditioner class for iterative linear solver
*   It is calling a method defined in nonLinearOperator
*   Also requires linear operator while using iterative Krylov solver
*/

#include <utility>

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator> 
class preconditioner_TG_stiff
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    preconditioner_TG_stiff(VectorOperations* vec_ops, NonlinearOperator* nonlin_op):
    vec_ops_(vec_ops), nonlin_op_(nonlin_op)
    {

    }

    ~preconditioner_TG_stiff()
    {

    }
    //sets a shifted linear operator!
    void set_operator(const LinearOperator *op_)
    {
        lin_op = (LinearOperator*)op_;
    }

    void apply(T_vec& x)const
    {
        auto ab = lin_op->get_a_b();
        // std::cout << "ab = " << ab.first << ", " << ab.second << std::endl;
        // std::cout << "norm Lx = " << vec_ops_->norm_l2(x) << std::endl;
        nonlin_op_->preconditioner_jacobian_stiff_u(x, ab.first, ab.second);
        // std::cout << "norm iPLx = " << vec_ops_->norm_l2(x) << std::endl;

    }

private:
    NonlinearOperator* nonlin_op_;
    LinearOperator* lin_op;
    VectorOperations* vec_ops_;


    
};


}


#endif