#ifndef __PRECONDITIONER_OVERSCREENING_BREAKDOWN_H__
#define __PRECONDITIONER_OVERSCREENING_BREAKDOWN_H__


/**
*   Helper preconditioner class for iterative linear solver
*   It is calling a method defined in nonlinear_operator
*   Also requires linear operator while using iterative Krylov solver
*/


namespace nonlinear_operators
{

template<class VectorOperations, class MatrixOperations, class NonlinearOperator, class LinearOperator> 
class preconditioner_overscreening_breakdown
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using T_mat = typename MatrixOperations::matrix_type;


    preconditioner_overscreening_breakdown(const VectorOperations* vec_ops_p, const NonlinearOperator* nonlin_op_p):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p)
    {
        vec_ops_->init_vector(b_);
        vec_ops_->start_use_vector(b_);
    }

    preconditioner_overscreening_breakdown(const NonlinearOperator* nonlin_op_p):
    nonlin_op_(nonlin_op_p)
    {
        vec_ops_ = nonlin_op_p->get_vec_ops_ref();
        vec_ops_->init_vector(b_);
        vec_ops_->start_use_vector(b_);
    }    

    ~preconditioner_overscreening_breakdown()
    {
        vec_ops_->stop_use_vector(b_);
        vec_ops_->free_vector(b_);
    }
    
    void set_operator(const LinearOperator* op_)const
    {
        lin_op_ = op_;
        // std::cout << "set operator!" << std::endl;
    }

    void apply(T_vec& x)const
    {
        if( lin_op_->is_constant_matrix() )
        {
            vec_ops_->assign(x, b_);
            nonlin_op_->solve_linear_system(lin_op_->get_matrix_ref(), b_, x );
        }
        else
        {
            nonlin_op_->solve_linear_system(lin_op_->get_matrix_ref(), x);
        }
    }

private:
    const NonlinearOperator* nonlin_op_;
    mutable const LinearOperator* lin_op_;
    const VectorOperations* vec_ops_;
    mutable T_vec b_;

    
};


}


#endif