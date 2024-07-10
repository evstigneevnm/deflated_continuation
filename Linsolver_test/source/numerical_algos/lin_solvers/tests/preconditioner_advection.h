#ifndef __PRECONDITIONER_ADVECTION_H__
#define __PRECONDITIONER_ADVECTION_H__


/**
*   Test class for iterative linear solver
*   Implements preconditioner to the  linear operator of the advection equation of the type:
*
*   u_t+a u_x = 0, a>0, x\in[0;1), u - periodic
*
*   u_j^{n+1} - u_j^{n} + a tau/dx( u_j^{n+1} - u_{j-1}^{n+1} ) = 0 
*    
*   u_j^{n+1} + a tau/dx (u_j^{n+1} - u_{j-1}^{n+1} ) = u_j^{n}
* 
*   A U^{n+1} = U^{n}
* 
*/

#include <memory>

namespace tests
{

template<class VectorOperations, class LinearOperator, class Log> 
class preconditioner_advection
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    preconditioner_advection(std::shared_ptr<VectorOperations> vec_ops, int prec_iter=1):
    vec_ops_(vec_ops), prec_iter_(prec_iter)
    {
        vec_ops_->init_vector(y_);
        vec_ops_->start_use_vector(y_);        
    }

    ~preconditioner_advection()
    {
        vec_ops_->stop_use_vector(y_);
        vec_ops_->free_vector(y_);        
    }
    
    void set_operator(const LinearOperator* op) const
    {
        N = op->get_size();
        diag_coeff_ = op->diag_coefficient();
        side_coeff_ = op->side_coefficient();
    }

    void apply(T_vec& x)const
    {
        vec_ops_->assign(x, y_);
        vec_ops_->assign_scalar(0, x);        
        for(int aa=0;aa<prec_iter_;aa++)
        {
            for(std::size_t j=0; j<N; j++)
            {
                std::size_t jm = ((j==0)?N-1:j-1);
                x[j] = ( y_[j]-side_coeff_*x[jm] )/diag_coeff_;
            }
            for(std::size_t j=N-1; j-->0;)
            {
                std::size_t jm = ((j==0)?N-1:j-1);
                x[j] = ( y_[j]-side_coeff_*x[jm] )/diag_coeff_;
            }
        }
    }

private:
    std::shared_ptr<VectorOperations> vec_ops_;
    mutable T_vec y_;
    int prec_iter_;
    mutable T diag_coeff_;
    mutable T side_coeff_;
    mutable std::size_t N;
    
};


}


#endif