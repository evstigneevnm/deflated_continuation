#ifndef __TEST_LIN_SOLVER_test_matrix_liner_operator_H__
#define __TEST_LIN_SOLVER_test_matrix_liner_operator_H__


template<class MatOps>
struct system_operator
{
    
    using vector_t = typename MatOps::vector_type;
    using matrix_t = typename MatOps::matrix_type;
    int sz;
    matrix_t A;
    mutable const MatOps* mat_ops_;
    system_operator(int sz_, const MatOps* mat_ops, matrix_t& A_): 
    sz(sz_), 
    A(A_), 
    mat_ops_(mat_ops)
    { }

    void apply(const vector_t& x, vector_t& f) const
    {
        mat_ops_->gemv('N', A, 1.0, x, 0, f);
    }

};

template<class VecOps, class MatOps>
struct prec_operator
{
    int sz;
    using vector_t = typename MatOps::vector_type;
    using matrix_t = typename MatOps::matrix_type;
    matrix_t iP;
    const VecOps* vec_ops_;
    const MatOps* mat_ops_;  
    mutable const system_operator<MatOps>* op;

    

    prec_operator(int sz_, const VecOps* vec_ops, const MatOps* mat_ops, matrix_t& iP_): 
    sz(sz_), 
    iP(iP_), 
    vec_ops_(vec_ops),
    mat_ops_(mat_ops)
    {
        
    }
    ~prec_operator()
    {
        
    }

    void set_operator(const system_operator<MatOps> *op_)const
    {
        std::cout << "prec_operator: operator is set." << std::endl;
        op = op_;
    }

    void apply(vector_t& x)const
    {
        
        vector_t some_vec;
        vec_ops_->init_vector(some_vec); vec_ops_->start_use_vector(some_vec);
        mat_ops_->gemv('N', iP, 1.0, x, 0, some_vec);
        vec_ops_->assign(some_vec, x);
        vec_ops_->stop_use_vector(some_vec); vec_ops_->free_vector(some_vec);
    }

};

#endif