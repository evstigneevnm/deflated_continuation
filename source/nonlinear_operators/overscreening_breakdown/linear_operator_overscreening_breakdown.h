#ifndef __LINEAR_OPERATOR_OVERSCREENING_BREAKDOWN_H__
#define __LINEAR_OPERATOR_OVERSCREENING_BREAKDOWN_H__

/**
*   Helper class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*
*/


namespace nonlinear_operators
{


template<class VectorOperations, class MatrixOperations, class NonlinearOperator> 
class linear_operator_overscreening_breakdown
{
public:    
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using T_mat = typename MatrixOperations::matrix_type;

    linear_operator_overscreening_breakdown(const VectorOperations* vec_ops_p, const NonlinearOperator* nonlin_op): 
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op),
    is_constant_matrix_(true)
    {

    }
    linear_operator_overscreening_breakdown(const NonlinearOperator* nonlin_op): 
    nonlin_op_(nonlin_op),
    is_constant_matrix_(true)
    {
        vec_ops_ = nonlin_op_->get_vec_ops_ref();
    }


    ~linear_operator_overscreening_breakdown()
    {

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

    void apply(const T_vec& x, T_vec& f)const //required because is used in extended linear soler
    {
        nonlin_op_->jacobian_u(x, f);
    }

private:
    VectorOperations* vec_ops_;
    const NonlinearOperator* nonlin_op_;
    bool is_constant_matrix_;

};

}

#endif