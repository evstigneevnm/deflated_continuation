#ifndef __PRECONDITIONER_DIFFUSION__
#define __PRECONDITIONER_DIFFUSION__


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
class preconditioner_diffusion
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    preconditioner_diffusion(std::shared_ptr<VectorOperations> vec_ops, int prec_iter=1):
    vec_ops_(vec_ops), prec_iter_(prec_iter)
    {
        vec_ops_->init_vector(y_);
        vec_ops_->start_use_vector(y_);
    }

    ~preconditioner_diffusion()
    {
        vec_ops_->stop_use_vector(y_);
        vec_ops_->free_vector(y_);
    }
    
    void set_operator(const LinearOperator* op_) 
    {
        N = op_->get_size();
        h_ = op_->get_h();
        tau_ = op_->get_tau();
        diag_coeff_ = (1+2*tau_/h_/h_);
        side_coeff_ = (tau_/h_/h_);
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
                    x[j] = (y_[j]+side_coeff_*x[j+1])/diag_coeff_;
                if(j==N-1)
                    x[j] = (y_[j]+side_coeff_*x[j-1])/diag_coeff_;            
            }
            for(std::size_t j=N-1; j-->0;)
            {
                if(j>0&&j<N-1)
                    x[j] = (y_[j]+side_coeff_*x[j-1]+side_coeff_*x[j+1])/diag_coeff_;
                if(j==0)
                    x[j] = (y_[j]+side_coeff_*x[j+1])/diag_coeff_;
                if(j==N-1)
                    x[j] = (y_[j]+side_coeff_*x[j-1])/diag_coeff_;            
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