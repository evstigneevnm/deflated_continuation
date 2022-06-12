#ifndef __LINEAR_OPERATOR_ABC_FLOW_SHIFTED_H__
#define __LINEAR_OPERATOR_ABC_FLOW_SHIFTED_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in NonlinearOperator
*
*/
#include <utility>

namespace nonlinear_operators
{


template<class VectorOperations, class NonlinearOperator> 
class linear_operator_abc_flow_shifted
{
public:    
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    struct ab_t
    {
        T a = 0.0;
        T b = 1.0;
    };

    linear_operator_abc_flow_shifted(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_): 
    vec_ops(vec_ops_),
    nonlin_op(nonlin_op_)
    {

    }
    ~linear_operator_abc_flow_shifted()
    {

    }

    // a+b*A
    void set_bA_plus_a(const T b_, const T a_)
    {
        ab.a = a_;
        ab.b = b_;
    }

    void apply(const T_vec& x, T_vec& f)const
    {
        nonlin_op->jacobian_u(x, f);
        //calc: y := mul_x*x + mul_y*y
        vec_ops->add_mul(ab.a, x, ab.b, f);

    }

    ab_t get_a_b()
    {
        return ab;
    }

private:
    VectorOperations* vec_ops;
    NonlinearOperator* nonlin_op;
    ab_t ab;

};

}

#endif