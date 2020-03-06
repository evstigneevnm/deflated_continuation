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

namespace nonlinear_operators
{


template<class VectorOperations_R, unsigned int BLOCK_SIZE_x=128>
class circle
{
public:
    
    typedef VectorOperations_R vector_operations_real;
    typedef typename VectorOperations_R::scalar_type  T;
    typedef typename VectorOperations_R::vector_type  T_vec;

    circle(T R_, vector_operations_real *vec_ops_R_): 
    R(R_), 
    vec_ops_R(vec_ops_R_)
    {
        common_constructor_operation();
        calculate_cuda_grid();
    }

    circle(T R_, dim3 dimGrid_, dim3 dimBlock_, vector_operations_real *vec_ops_R_): 
    R(R_), 
    dimGrid(dimGrid_), dimBlock(dimBlock_), 
    vec_ops_R(vec_ops_R_)
    {
        common_constructor_operation();

    }

    ~circle()
    {

        vec_ops_R->stop_use_vector(u_0); vec_ops_R->free_vector(u_0);

    }

    //nonlinear operator:
    //   F(u,alpha)=v
    void F(const T_vec& u, const T alpha, T_vec& v)
    {
        



        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u, TC(1.0,0.0), (const TC_vec&)gradient_x, u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u, TC(1.0,0.0), (const TC_vec&)gradient_y, u_y_hat);


        //z=x*y;
        vec_ops_R->mul_pointwise(1.0, u_x_ext, 1.0, u_ext, w1_ext);
        vec_ops_R->mul_pointwise(1.0, u_y_ext, 1.0, u_ext, w2_ext);
        fft(w1_ext, u_x_hat);
        fft(w2_ext, u_y_hat);

        // b_val*biharmonic*u->v
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(b_val), (const TC_vec&)biharmonic, v);
       
        // alpha*Laplace*u->b_hat
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(alpha), (const TC_vec&)Laplace, b_hat);
 
        // alpha*a_val*(uu_x)+alpha*a_val*(uu_y)+b_hat->b_hat
        vec_ops_C->add_mul(TC(alpha*a_val), (const TC_vec&) u_x_hat, TC(alpha*a_val), (const TC_vec&) u_y_hat, TC(1.0), b_hat);
       
        // b_hat+v->v
        vec_ops_C->add_mul(TC(1.0), (const TC_vec&)b_hat, v);
       

        
    }

    //sets (u_0, alpha_0) for jacobian linearization
    //stores alpha_0, u_0, u_ext_0, u_x_ext_0, u_y_ext_0
    //NOTE: u_ext_0, u_x_ext_0 and u_y_ext_0 MUST NOT BE CHANGED!!!
    void set_linearization_point(const TC_vec& u_0_, const T alpha_0_)
    {
        vec_ops_C->assign(u_0_,u_0);
        alpha_0=alpha_0_;
        ifft((TC_vec&)u_0,u_ext_0);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u_0, TC(1.0,0.0), (const TC_vec&)gradient_x, u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u_0, TC(1.0,0.0), (const TC_vec&)gradient_y, u_y_hat);
        ifft(u_x_hat,u_x_ext_0);
        ifft(u_y_hat,u_y_ext_0);
    }

    void set_linearization_point(const T_vec_im& u_0_, const T alpha_0_)
    {
        R2C(u_0_, u_helper_in);
        set_linearization_point(u_helper_in, alpha_0_);
    }

    //variational jacobian for 2D KS equations J=dF/du
    //returns vector dv as Jdu->dv, where J(u_0,alpha_0) linearized at (u_0, alpha_0) by set_linearization_point
    void jacobian_u(const TC_vec& du, TC_vec& dv)
    {
 

    }


    //variational jacobian for 2D KS equations J=dF/dalpha
    void jacobian_alpha(TC_vec& dv)
    {
        vec_ops_R->mul_pointwise(1.0, u_x_ext_0, 1.0, u_ext_0, w1_ext_0);
        vec_ops_R->mul_pointwise(1.0, u_y_ext_0, 1.0, u_ext_0, w2_ext_0);
        //du_x0*u_0->u_x_hat0
        fft(w1_ext_0, u_x_hat0);
        //du_y0*u_0->u_y_hat0
        fft(w2_ext_0, u_y_hat0);
        // d^2du->b_hat
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u_0, TC(1.0), (const TC_vec&)Laplace, dv);
        //a_val*alpha*(u_x_hat0+u_y_hat0)->u_y_hat0
        vec_ops_C->add_mul(TC(1.0), u_x_hat0, TC(1.0), u_y_hat0);
        // a_val*u_y_hat0+dv->dv
        vec_ops_C->add_mul(TC(a_val), u_y_hat0, TC(1.0,0.0), dv);
    }    
    void jacobian_alpha(const TC_vec& x0, const T& alpha, TC_vec& dv)
    {
           

    }


    void preconditioner_jacobian_u(TC_vec& dr)
    {
        //void function cos there's no need for a preconditioner.

    }

    void set_cuda_grid(dim3 dimGrid_, dim3 dimGrid_F_, dim3 dimBlock_)
    {
        dimGrid=dimGrid_;
        dimGrid_F=dimGrid_F_;
        dimBlock=dimBlock_;
    }

    void get_cuda_grid(dim3 &dimGrid_, dim3 &dimGrid_F_, dim3 &dimBlock_)
    {  
        dimGrid_=dimGrid;
        dimGrid_F_=dimGrid_F;
        dimBlock_=dimBlock;

    }
    

    
    void physical_solution(TC_vec& u_in, T_vec& u_out)
    {

    }

    void fourier_solution(T_vec& u_in, TC_vec& u_out)
    {

    }


    void randomize_vector(T_vec_im& u_out)
    {
        vec_ops_R->assign_random(du_x_ext);
    }
    

private:
    T a_val,b_val;
    dim3 dimGrid;
    dim3 dimBlock;

    vector_operations_real *vec_ops_R;
    
 
    T_vec u_0=nullptr; // linearization point solution
    T alpha_0=0.0;   // linearization point parameter


    void common_constructor_operation()
    {  
        vec_ops_R->init_vector(u_0); vec_ops_R->start_use_vector(u_0); 

    }


    void calculate_cuda_grid()
    {
        dim3 s_dimBlock( BLOCK_SIZE_x );
        dimBlock=s_dimBlock;
        unsigned int blocks_x=floor(Nx/( BLOCK_SIZE_x ))+1;
        dim3 s_dimGrid(blocks_x);
        dimGrid=s_dimGrid;
    }


    void set_gradient_coefficients()
    {
        gradient_Fourier<TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y);
    }

    void set_Laplace_coefficients()
    {
        Laplace_Fourier<TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, Laplace);
    }
   
    void set_biharmonic_coefficients()
    {
        biharmonic_Fourier<TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, biharmonic);
    }


    


};

}

#endif
