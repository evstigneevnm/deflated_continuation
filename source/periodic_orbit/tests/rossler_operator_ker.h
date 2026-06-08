#ifndef __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_KER_H__
#define __NONLINEAR_OPERATORS_ROSSLER_OPERATOR_KER_H__



#include <tuple>
//for low level operations
#include <utils/cuda_support.h>

namespace nonlinear_operators
{

template<class T, class T_vec, unsigned int BlockSize = 32>
struct rossler_operator_ker
{
    rossler_operator_ker()
    {
        calculate_cuda_grid();
    }
    ~rossler_operator_ker()
    {    }

    void set_initial(T_vec& x)const;
    void set_period_point(T_vec& x0)const;
    void jacobian_u(const T_vec& x0, const std::tuple<T,T,T>& params, const T_vec& x_in_p, T_vec& x_out_p)const;
    void F(const T_vec& in_p, const std::tuple<T,T,T>& params, T_vec& out_p )const;


private:
    dim3 dimBlock;
    dim3 dimGrid;

    void calculate_cuda_grid()
    {
        dim3 s_dimBlock(BlockSize);
        dimBlock = s_dimBlock;
        unsigned int blocks_x=(3+BlockSize)/BlockSize;
        dim3 s_dimGrid(blocks_x);
        dimGrid=s_dimGrid;
    }    

    
};

}

#endif