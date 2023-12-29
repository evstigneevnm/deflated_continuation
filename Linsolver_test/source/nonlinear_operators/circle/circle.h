#ifndef __CIRCLE_TEST_ND__
#define __CIRCLE_TEST_ND__


/**
*    Problem class for:
*    (1)    f(x,lambda) := x*x+lambda*lambda-R^2 = 0
*   
*    1-dim sphere for x \in R and lambda \in R
*    testing problem for the debugging of the continuation algorythm
*    The vector operation class with size = 1 must be used in templates for this to work
*
*
*
*/

#include <nonlinear_operators/circle/circle_ker.h>
#include <vector>

namespace nonlinear_operators
{


template<class VectorOperations_R, unsigned int BLOCK_SIZE_x=64>
class circle
{
public:
    
    typedef VectorOperations_R vector_operations_real;
    typedef typename VectorOperations_R::scalar_type  T;
    typedef typename VectorOperations_R::vector_type  T_vec;

    circle(T R_, size_t Nx_, vector_operations_real *vec_ops_R_): 
    R(R_), 
    vec_ops_R(vec_ops_R_),
    Nx(Nx_)
    {
        common_constructor_operation();
        calculate_cuda_grid();
    }

    circle(T R_, size_t Nx_, dim3 dimGrid_, dim3 dimBlock_, vector_operations_real *vec_ops_R_): 
    R(R_), 
    dimGrid(dimGrid_), dimBlock(dimBlock_), 
    vec_ops_R(vec_ops_R_),
    Nx(Nx_)
    {
        common_constructor_operation();

    }

    ~circle()
    {

        vec_ops_R->stop_use_vector(u_0); vec_ops_R->free_vector(u_0);
        free(xp_host);
    }

    //nonlinear operator:
    //   F(u,alpha)=v
    void F(const T_vec& u, const T alpha, T_vec& v)
    {
        function<T>(dimGrid, dimBlock, Nx, R, (const T*&)u, alpha, v);
    }

    //sets (u_0, alpha_0) for jacobian linearization
    //stores alpha_0, u_0, u_ext_0, u_x_ext_0, u_y_ext_0
    //NOTE: u_ext_0, u_x_ext_0 and u_y_ext_0 MUST NOT BE CHANGED!!!
    void set_linearization_point(const T_vec& u_0_, const T alpha_0_)
    {
        vec_ops_R->assign(u_0_,u_0);
        alpha_0 = alpha_0_;

    }

    //variational jacobian for 2D KS equations J=dF/du
    //returns vector dv as Jdu->dv, where J(u_0,alpha_0) linearized at (u_0, alpha_0) by set_linearization_point
    void jacobian_u(const T_vec& du, T_vec& dv)
    {
        jacobian_x<T>(dimGrid, dimBlock, Nx, R, (const T*&) u_0, alpha_0, (const T*&) du, dv);
    }


    //variational jacobian for 2D KS equations J=dF/dalpha
    void jacobian_alpha(T_vec& dv)
    {
        jacobian_lambda<T>(dimGrid, dimBlock, Nx, R, (const T*&) u_0, alpha_0, dv);
    }   
    void jacobian_alpha(const T_vec& x0, const T& alpha, T_vec& dv)
    {
        jacobian_lambda<T>(dimGrid, dimBlock, Nx, R, (const T*&) x0, alpha, dv);
    }


    void preconditioner_jacobian_u(T_vec& dr)
    {
        //void function cos there's no need for a preconditioner.
    }

    void set_cuda_grid(dim3 dimGrid_, dim3 dimBlock_)
    {
        dimGrid=dimGrid_;
        dimBlock=dimBlock_;
    }

    void get_cuda_grid(dim3 &dimGrid_, dim3 &dimBlock_)
    {  
        dimGrid_=dimGrid;
        dimBlock_=dimBlock;

    }
        
    void physical_solution(T_vec& u_in, T_vec& u_out)
    {
        //void funciton that should return a physical solution
    }
    void project(T_vec& u_)
    {
        //void fuction to project to invariant subspace
    }
    void exact_solution(const T& param, T_vec& u_out)
    {
        //void function for exact solution.
    }
    T check_solution_quality(const T_vec& u_in)
    {
        //void function that returns some norm of solution diverge from invariant subspace
        return 0;
    }

    void norm_bifurcation_diagram(const T_vec& u_in, std::vector<T>& res) const
    {
        device_2_host_cpy(xp_host, u_in, Nx);
        T val = xp_host[0];   
        res.clear();
        res.reserve(2);
        res.push_back(val);
        res.push_back(std::abs(val));
    }

    void randomize_vector(T_vec& u_out)
    {
        vec_ops_R->assign_random(u_out);
    }
    

private:
    T R;
    dim3 dimGrid;
    dim3 dimBlock;
    vector_operations_real *vec_ops_R;
    size_t Nx;
    T* xp_host; //for bifurcation diagram plotting

    T_vec u_0=nullptr; // linearization point solution
    T alpha_0=0.0;   // linearization point parameter



    void common_constructor_operation()
    {  
        vec_ops_R->init_vector(u_0); vec_ops_R->start_use_vector(u_0); 
        xp_host = (T*)malloc(Nx*sizeof(T));
    }


    void calculate_cuda_grid()
    {
        dim3 s_dimBlock( BLOCK_SIZE_x );
        dimBlock=s_dimBlock;
        unsigned int blocks_x=floor(Nx/( BLOCK_SIZE_x ))+1;
        dim3 s_dimGrid(blocks_x);
        dimGrid=s_dimGrid;
    }

    


};

}

#endif
