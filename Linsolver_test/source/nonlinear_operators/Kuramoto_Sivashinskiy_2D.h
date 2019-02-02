#ifndef __KURAMOTO_SIVASHINSKIY_2D__
#define __KURAMOTO_SIVASHINSKIY_2D__

#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D_ker.h>


template<class FFT_type, class VectorOperations_R, class VectorOperations_C, unsigned int BLOCK_SIZE_x=64, unsigned int BLOCK_SIZE_y=16>
class Kuramoto_Sivashinskiy_2D
{
public:
    
    typedef VectorOperations_R vector_operations_real;
    typedef VectorOperations_C vector_operations_complex;
    typedef typename VectorOperations_R::scalar_type  T;
    typedef typename VectorOperations_R::vector_type  T_vec;
    typedef typename VectorOperations_C::scalar_type  TC;
    typedef typename VectorOperations_C::vector_type  TC_vec;
    


    Kuramoto_Sivashinskiy_2D(T a_val_, T b_val_, size_t Nx_, size_t Ny_, vector_operations_real *vec_ops_R_, vector_operations_complex *vec_ops_C_, FFT_type *FFT_): Nx(Nx_), Ny(Ny_), vec_ops_R(vec_ops_R_), vec_ops_C(vec_ops_C_), FFT(FFT_), a_val(a_val_), b_val(b_val_)
    {
        common_constructor_operation();
        calculate_cuda_grid();
        init_all_derivatives();
    }

    Kuramoto_Sivashinskiy_2D(T a_val_, T b_val_, size_t Nx_, size_t Ny_, dim3 dimGrid_, dim3 dimGrid_F_, dim3 dimBlock_, vector_operations_real *vec_ops_R_, vector_operations_complex *vec_ops_C_, FFT_type &FFT_): Nx(Nx_), Ny(Ny_), dimGrid(dimGrid_), dimGrid_F(dimGrid_F_), dimBlock(dimBlock_), vec_ops_R(vec_ops_R_), vec_ops_C(vec_ops_C_), FFT(FFT_), a_val(a_val_), b_val(b_val_)
    {

        common_constructor_operation();
        init_all_derivatives();
    }

    ~Kuramoto_Sivashinskiy_2D()
    {
        vec_ops_C->stop_use_vector(gradient_x); vec_ops_C->free_vector(gradient_x);
        vec_ops_C->stop_use_vector(gradient_y); vec_ops_C->free_vector(gradient_y);
        vec_ops_C->stop_use_vector(Laplace); vec_ops_C->free_vector(Laplace);
        vec_ops_C->stop_use_vector(biharmonic); vec_ops_C->free_vector(biharmonic);
        vec_ops_C->stop_use_vector(u_x_hat); vec_ops_C->free_vector(u_x_hat);
        vec_ops_C->stop_use_vector(u_y_hat); vec_ops_C->free_vector(u_y_hat);        
        vec_ops_C->stop_use_vector(b_hat); vec_ops_C->free_vector(b_hat);                

        vec_ops_R->stop_use_vector(w1_ext); vec_ops_R->free_vector(w1_ext);
        vec_ops_R->stop_use_vector(w2_ext); vec_ops_R->free_vector(w2_ext);
        vec_ops_R->stop_use_vector(u_ext); vec_ops_R->free_vector(u_ext);
        vec_ops_R->stop_use_vector(u_x_ext); vec_ops_R->free_vector(u_x_ext);
        vec_ops_R->stop_use_vector(u_y_ext); vec_ops_R->free_vector(u_y_ext);

    }

    //nonlinear Kuramoto Sivashinskiy 2D operator:
    //   F(u,alpha)=v
    void F(const TC*& u, const T alpha, TC*& v)
    {
        
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u, TC(1.0,0.0), (const TC_vec&)gradient_x, u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u, TC(1.0,0.0), (const TC_vec&)gradient_y, u_y_hat);
        ifft(u_x_hat,u_x_ext);
        ifft(u_y_hat,u_y_ext);
        ifft((TC_vec&)u,u_ext);

         //z=x*y;
        vec_ops_R->mul_pointwise(1.0, u_x_ext, 1.0, u_ext, w1_ext);
        vec_ops_R->mul_pointwise(1.0, u_y_ext, 1.0, u_ext, w2_ext);
        fft(w1_ext,u_x_hat);
        fft(w2_ext,u_y_hat);
        // b_val*biharmonic*u->v
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(b_val), (const TC_vec&)biharmonic, v);
        // alpha*Laplace*u->b_hat
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(alpha), (const TC_vec&)Laplace, b_hat);
        // alpha*a_val*(uu_x)+alpha*a_val*(uu_y)+b_hat->b_hat
        vec_ops_C->add_mul(TC(alpha*a_val), (const TC_vec&) u_x_hat, TC(alpha*a_val), (const TC_vec&) u_y_hat, TC(1.0), b_hat);
        // b_hat+v->v
        vec_ops_C->add_mul(TC(1.0), (const TC_vec&)b_hat, v);
        
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


private:
    T a_val,b_val;
    dim3 dimGrid;
    dim3 dimBlock;
    dim3 dimGrid_F;
    vector_operations_real *vec_ops_R;
    vector_operations_complex *vec_ops_C;


    size_t Nx, Ny, My;
    FFT_type *FFT;
    TC_vec gradient_x=NULL;
    TC_vec gradient_y=NULL;
    TC_vec Laplace=NULL;
    TC_vec biharmonic=NULL;
    TC_vec u_x_hat=NULL;
    TC_vec u_y_hat=NULL;
    TC_vec b_hat=NULL;

    T_vec w1_ext=NULL;
    T_vec w2_ext=NULL;
    T_vec u_ext=NULL;
    T_vec u_x_ext=NULL;
    T_vec u_y_ext=NULL;



    void common_constructor_operation()
    {   
        My=FFT->get_reduced_size();
        vec_ops_C->init_vector(gradient_x); vec_ops_C->start_use_vector(gradient_x); 
        vec_ops_C->init_vector(gradient_y); vec_ops_C->start_use_vector(gradient_y); 
        vec_ops_C->init_vector(Laplace); vec_ops_C->start_use_vector(Laplace); 
        vec_ops_C->init_vector(biharmonic); vec_ops_C->start_use_vector(biharmonic); 
        vec_ops_C->init_vector(u_x_hat); vec_ops_C->start_use_vector(u_x_hat); 
        vec_ops_C->init_vector(u_y_hat); vec_ops_C->start_use_vector(u_y_hat); 
        vec_ops_C->init_vector(b_hat); vec_ops_C->start_use_vector(b_hat); 

        vec_ops_R->init_vector(w1_ext); vec_ops_R->start_use_vector(w1_ext); 
        vec_ops_R->init_vector(w2_ext); vec_ops_R->start_use_vector(w2_ext); 
        vec_ops_R->init_vector(u_ext); vec_ops_R->start_use_vector(u_ext); 
        vec_ops_R->init_vector(u_x_ext); vec_ops_R->start_use_vector(u_x_ext); 
        vec_ops_R->init_vector(u_y_ext); vec_ops_R->start_use_vector(u_y_ext); 

    }

    void init_all_derivatives()
    {
        set_gradient_coefficients();
        set_Laplace_coefficients();
        set_biharmonic_coefficients();        
    }

    void calculate_cuda_grid()
    {
        dim3 s_dimBlock( BLOCK_SIZE_x, BLOCK_SIZE_y );
        dimBlock=s_dimBlock;
        unsigned int blocks_x=floor(Nx/( BLOCK_SIZE_x ))+1;
        unsigned int blocks_y=floor(Ny/( BLOCK_SIZE_y ))+1;
        unsigned int blocks_y_F=floor(My/( BLOCK_SIZE_y ))+1;
        dim3 s_dimGrid( blocks_x, blocks_y);
        dimGrid=s_dimGrid;
        dim3 s_dimGrid_F( blocks_x, blocks_y_F);
        dimGrid_F=s_dimGrid_F;

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

    void ifft(TC_vec& u_hat_, T_vec& u_)
    {
        FFT->ifft(u_hat_, u_);
        T scale = T(1.0)/(Nx*Ny);
        vec_ops_R->scale(scale, u_);
    }
    void fft(T_vec& u_, TC_vec& u_hat_)
    {
        FFT->fft(u_, u_hat_);
    }

};






#endif