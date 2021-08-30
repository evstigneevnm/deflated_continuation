#ifndef __LINEAR_OPERATOR_K_3D_SHIFTED_H__
#define __LINEAR_OPERATOR_K_3D_SHIFTED_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in NonlinearOperator
*
*/
#include <utility>

namespace NonlinearOperators
{


template<class VectorOperations, class NonlinearOperator> 
class linear_operator_K_3D_shifted
{
public:    
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    linear_operator_K_3D_shifted(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_): 
    vec_ops(vec_ops_),
    nonlin_op(nonlin_op_)
    {

    }
    ~linear_operator_K_3D_shifted()
    {

    }

    // a+b*A
    void set_bA_plus_a(const T b_, const T a_)
    {
        a = a_;
        b = b_;
    }

    void apply(const T_vec& x, T_vec& f)const
    {
        nonlin_op->jacobian_u(x, f);
        //calc: y := mul_x*x + mul_y*y
        vec_ops->add_mul(a, x, b, f);

    }

    std::pair<T,T> get_a_b()
    {
        return {a, b};
    }

private:
    VectorOperations* vec_ops;
    NonlinearOperator* nonlin_op;
    T a = T(0.0);
    T b = T(1.0);

};

}

#endif