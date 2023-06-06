#ifndef __PRECONDITIONER_OVERSCREENING_BREAKDOWN_SHIFTED_H__
#define __PRECONDITIONER_OVERSCREENING_BREAKDOWN_SHIFTED_H__


/**
*   Helper preconditioner class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*   Also requires linear operator while using iterative Krylov solver
*/


namespace nonlinear_operators
{

template<class VectorOperations, class MatrixOperations, class NonlinearOperator, class LinearOperator> 
class preconditioner_overscreening_breakdown_shifted
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using T_mat = typename MatrixOperations::matrix_type;


    preconditioner_overscreening_breakdown_shifted(const VectorOperations* vec_ops_p, const NonlinearOperator* nonlin_op_p):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p)
    {
        vec_ops_->init_vector(b_);
        vec_ops_->start_use_vector(b_);
    }

    ~preconditioner_overscreening_breakdown_shifted()
    {
        vec_ops_->stop_use_vector(b_);
        vec_ops_->free_vector(b_);
    }
    
    void set_operator(const LinearOperator* op_) const
    {
        lin_op_ = op_;
    }

    void apply(T_vec& x) const
    {
        auto ab = lin_op_->get_a_b();
        if( lin_op_->is_constant_matrix() )
        {
            vec_ops_->assign(x, b_);
            nonlin_op_->solve_linear_system(ab.b, lin_op_->get_matrix_ref(), ab.a, b_, x );
        }
        else
        {
            nonlin_op_->solve_linear_system(ab.b, lin_op_->get_matrix_ref(), ab.a, x);
        }
    }

private:
    const NonlinearOperator* nonlin_op_;
    mutable const LinearOperator* lin_op_;
    const VectorOperations* vec_ops_;
    T_vec b_;

    
};


}


#endif