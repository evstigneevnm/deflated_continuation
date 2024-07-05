#ifndef __LINEAR_OPERATOR_TG_STIFF_H__
#define __LINEAR_OPERATOR_TG_STIFF_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in NonlinearOperator
*
*/
#include <utility>

namespace nonlinear_operators
{


template<class VectorOperations, class NonlinearOperator> 
class linear_operator_TG_stiff
{
public:    
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    linear_operator_TG_stiff(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_): 
    vec_ops(vec_ops_),
    nonlin_op(nonlin_op_)
    {

    }
    ~linear_operator_TG_stiff()
    {

    }

    // a+b*A
    void set_aE_plus_bA(const std::pair<T,T>& ab_pair)
    {
        ab = ab_pair;
    }

    void apply(const T_vec& x, T_vec& f)const
    {
        nonlin_op->jacobian_stiff_u(x, f);
        //calc: y := mul_x*x + mul_y*y
        vec_ops->add_mul(ab.first, x, ab.second, f);
    }

    std::pair<T, T> get_a_b()
    {
        return ab;
    }

private:
    VectorOperations* vec_ops;
    NonlinearOperator* nonlin_op;
    std::pair<T, T> ab;

};

}

#endif