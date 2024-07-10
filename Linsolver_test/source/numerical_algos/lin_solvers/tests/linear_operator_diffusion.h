#ifndef __TEST_LINEAR_OPERATOR_DIFFUSION_H__
#define __TEST_LINEAR_OPERATOR_DIFFUSION_H__

/**
*   Test class for iterative linear solver
*   Implements linear operator of the diffusion equation of the type:
*
*   u_t=u_xx, x\in[0;1), u=0 on boundaries
*
*   u_j^{n+1} - u_j^{n} + tau/dx^2( -u_{j+1}^{n+1} +2u_j^{n+1} - u_{j-1}^{n+1} ) = 0 
*    
*   (1+ 2tau/dx^2 ) u_j^{n+1} - tau/dx^2 u_{j+1}^{n+1} - tau/dx^2 u_{j-1}^{n+1} = u_j^{n}
* 
*   A U^{n+1} = U^{n}
* 
*/


namespace tests
{


template<class VectorOperations, class Log> 
class linear_operator_diffusion
{
public:    
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using Ord = typename VectorOperations::ordinal_type;
private:
    using T = scalar_type;
    using T_vec = vector_type;
    const VectorOperations& vec_ops_;
    Ord N;
    T tau_;
    T h_;
    T cfl_;

public:

    linear_operator_diffusion(const VectorOperations& vec_ops, T tau):
    vec_ops_(vec_ops), tau_(tau)
    {
        N = vec_ops_.size();
        h_ = 1.0/static_cast<T>(N);
    }
    ~linear_operator_diffusion()
    {}

    T diag_coefficient() const
    {
        return (1+2*tau_/h_/h_);
    }
    Ord get_size() const
    {
        return N;
    }
    T get_h() const
    {
        return h_;
    }
    T get_tau() const
    {
        return tau_;
    }

    void apply(const T_vec& x, T_vec& f)const
    {
        //action of the linear operator on the input vector
        //(1+ 2tau/dx^2 ) u_j^{n+1} - tau/dx^2 u_{j+1}^{n+1} - tau/dx^2 u_{j-1}^{n+1}
        for(Ord j=0; j<N; j++)
        {
            if((j>0)&&(j<N-1))
                f[j] = (1+2*tau_/h_/h_)*x[j] - (tau_/h_/h_)*x[j-1] - (tau_/h_/h_)*x[j+1];
            else if(j==0)
                f[j] = (1+2*tau_/h_/h_)*x[j] - (tau_/h_/h_)*x[j+1];
            else if(j==N-1)
                f[j] = (1+2*tau_/h_/h_)*x[j] - (tau_/h_/h_)*x[j-1];
        }

    }

private:

};

}

#endif