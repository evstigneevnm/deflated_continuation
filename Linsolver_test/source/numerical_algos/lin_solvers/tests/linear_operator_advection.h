#ifndef __TEST_LINEAR_OPERATOR_ADVECTION_H__
#define __TEST_LINEAR_OPERATOR_ADVECTION_H__

/**
*   Test class for iterative linear solver
*   Implements linear operator of the advection equation of the type:
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


namespace tests
{


template<class VectorOperations, class Log> 
class linear_operator_advection
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
    T a_;
    T tau_;
    T h_;
    T cfl_;

public:

    linear_operator_advection(const VectorOperations& vec_ops, T a, T tau):
    vec_ops_(vec_ops), a_(a), tau_(tau)
    {
        N = vec_ops_.size();
        h_ = 1.0/static_cast<T>(N);
        cfl_ = a_*tau_/h_;
    }
    ~linear_operator_advection()
    {}

    T diag_coefficient() const
    {
        return (1.0+cfl_);
    }
    T side_coefficient() const
    {
        return (-cfl_);   
    }
    Ord get_size() const
    {
        return N;
    }    

    void apply(const T_vec& x, T_vec& f)const
    {
        //action of the linear operator on the input vector
        for(Ord j=0; j<N; j++)
        {
            Ord jm = ((j==0)?N-1:j-1);

            f[j] = (1.0+cfl_)*x[j]-(cfl_)*x[jm];
        }
    }

private:

};

}

#endif