#ifndef __PRECONDITIONER_ELLIPTIC__
#define __PRECONDITIONER_ELLIPTIC__


/**
*   Test class for iterative linear solver
*   Implements preconditioner to the  linear operator of the elliptic equation of the type:
*
*   -u_{xx} = f(x), x\in[0;1), u - periodic, f(x) is in the perator domain
*
*   -u_{j-1}/h^2 +2 u_{j}/h^2 - u_{j+1}/h^2 = f(x_j) 
*    
* 
*   A U = U^{n}
* 
*   preconditioner for the residual vecotr R={r_j}:
* 
*   u_{j}^{n+1} = (r_{j} + u_{j+1}/h^2^{n} + u_{j-1}/h^2 ^{n+1})/(2/h^2)
*
*/

#include <memory>

namespace tests
{

template<class VectorOperations, class LinearOperator, class Log> 
class preconditioner_elliptic
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    preconditioner_elliptic(std::shared_ptr<VectorOperations> vec_ops, int prec_iter=1):
    vec_ops_(vec_ops), prec_iter_(prec_iter)
    {
        vec_ops_->init_vector(y_);
        vec_ops_->start_use_vector(y_);
    }

    ~preconditioner_elliptic()
    {
        vec_ops_->stop_use_vector(y_);
        vec_ops_->free_vector(y_);
    }
    
    void set_operator(const LinearOperator* op_) 
    {
        N = op_->get_size();
        h_ = op_->get_h();
        diag_coeff_ = (2/h_/h_);
        side_coeff_ = (1/h_/h_);
    }

    void apply(T_vec& x)const
    {
        vec_ops_->assign(x, y_);
        vec_ops_->assign_scalar(0, x);
        for(int aa=0;aa<prec_iter_;aa++)
        {
            for(std::size_t j=0; j<N;j++)
            {
                if(j>0&&j<N-1)
                    x[j] = (y_[j]+side_coeff_*x[j-1]+side_coeff_*x[j+1])/diag_coeff_;
                if(j==0)
                    x[0] = (y_[0]+side_coeff_*x[1]+side_coeff_*x[N-1])/diag_coeff_;
                if(j==N-1)
                    x[N-1] = (y_[N-1]+side_coeff_*x[N-2]+side_coeff_*x[0])/diag_coeff_;            
            }
            for(std::size_t j=N-1; j-->0;)
            {
                if(j>0&&j<N-1)
                    x[j] = (y_[j]+side_coeff_*x[j-1]+side_coeff_*x[j+1])/diag_coeff_;
                if(j==0)
                    x[0] = (y_[0]+side_coeff_*x[1]+side_coeff_*x[N-1])/diag_coeff_;
                if(j==N-1)
                    x[N-1] = (y_[N-1]+side_coeff_*x[N-2]+side_coeff_*x[0])/diag_coeff_;              
            }

        }
    }

private:
    // const LinearOperator* lin_op;
    T diag_coeff_;
    T side_coeff_;
    std::size_t N;
    T h_;
    T tau_;
    std::shared_ptr<VectorOperations> vec_ops_;
    mutable T_vec y_;
    int prec_iter_;

};


}


#endif