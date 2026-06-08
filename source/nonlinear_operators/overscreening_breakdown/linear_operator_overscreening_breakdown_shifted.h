#ifndef __LINEAR_OPERATOR_OVERSCREENING_BREAKDOWN_SHIFTED_H__
#define __LINEAR_OPERATOR_OVERSCREENING_BREAKDOWN_SHIFTED_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*
*/


namespace nonlinear_operators
{


template<class VectorOperations, class MatrixOperations, class NonlinearOperator> 
class linear_operator_overscreening_breakdown_shifted
{
public:    
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using T_mat = typename MatrixOperations::matrix_type;

    linear_operator_overscreening_breakdown_shifted(const VectorOperations* vec_ops_p, const NonlinearOperator* nonlin_op): 
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op),
    is_constant_matrix_(true)
    {

    }
    ~linear_operator_overscreening_breakdown_shifted()
    {

    }

    struct ab_t
    {
        T a = 0.0;
        T b = 1.0;
    };

    // a+b*A
    void set_bA_plus_a(const T b_, const T a_)
    {
        ab.a = a_;
        ab.b = b_;
    }
    ab_t get_a_b() const
    {
        return ab;
    }

    T_mat& get_matrix_ref()const
    {
        return nonlin_op_->jacobian_u();
    }
    void set_constant_matrix(bool is_constant_matrix_p)const
    {
        is_constant_matrix_ = is_constant_matrix_p;
    }
    bool is_constant_matrix()const
    {
        return is_constant_matrix_;
    }


    void apply(const T_vec& x, T_vec& f)const
    {
        nonlin_op_->jacobian_u(x, f);
        vec_ops_->add_mul(ab.a, x, ab.b, f);
    }

private:
    const VectorOperations* vec_ops_;
    const NonlinearOperator* nonlin_op_;
    bool is_constant_matrix_;
    ab_t ab;

};

}

#endif