#ifndef __KURAMOTO_SIVASHINSKIY_2D__
#define __KURAMOTO_SIVASHINSKIY_2D__

#include <external_libraries/cufft_wrap.h>
#include <thrust/complex.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D_ker.h>
//DEBUG
#include <iostream>
#include "file_operations.h"
#include <limits>
//ENDS

template<typename VectorOperations_R, typename VectorOperations_C, unsigned int BLOCK_SIZE_x=64, unsigned int BLOCK_SIZE_y=16>
class Kuramoto_Sivashinskiy_2D
{
public:
    
    typedef VectorOperations_R vector_operations_real;
    typedef VectorOperations_C vector_operations_complex;
    typedef typename VectorOperations_R::scalar_type  T;
    typedef typename VectorOperations_R::vector_type  T_vec;
    typedef typename VectorOperations_C::scalar_type  TC;
    typedef typename VectorOperations_C::vector_type  TC_vec;
    


    Kuramoto_Sivashinskiy_2D(size_t Nx_, size_t Ny_, vector_operations_real *vec_ops_R_, vector_operations_complex *vec_ops_C_, cufft_wrap_R2C<T> *CUFFT_): Nx(Nx_), Ny(Ny_), vec_ops_R(vec_ops_R_), vec_ops_C(vec_ops_C_), CUFFT(CUFFT_)
    {
        common_constructor_operation();
        calculate_cuda_grid();
        init_all_derivatives();
    }

    Kuramoto_Sivashinskiy_2D(size_t Nx_, size_t Ny_, dim3 dimGrid_, dim3 dimGrid_F_, dim3 dimBlock_, vector_operations_real *vec_ops_R_, vector_operations_complex *vec_ops_C_, cufft_wrap_R2C<T> &CUFFT_): Nx(Nx_), Ny(Ny_), dimGrid(dimGrid_), dimGrid_F(dimGrid_F_), dimBlock(dimBlock_), vec_ops_R(vec_ops_R_), vec_ops_C(vec_ops_C_), CUFFT(CUFFT_)
    {

        common_constructor_operation();
        init_all_derivatives();
    }

    ~Kuramoto_Sivashinskiy_2D()
    {
        if(gradient_x!=NULL)
            cudaFree(gradient_x);
        if(gradient_y!=NULL)
            cudaFree(gradient_y);
        if(Laplace!=NULL)
            cudaFree(Laplace);
        if(biharmonic!=NULL)
            cudaFree(biharmonic);
        if(u_x_hat!=NULL)
            cudaFree(u_x_hat);
        if(u_y_hat!=NULL)
            cudaFree(u_y_hat);
    }

    //nonlinear Kuramoto Sivashinskiy 2D operator:
    //   F(u,alpha)=v
    void F(const TC*& u, const T alpha, TC*& v)
    {
        
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(1.0), (const TC_vec&)gradient_x, (TC*&)u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(1.0), (const TC_vec&)gradient_y, (TC*&)u_y_hat);
        

    }



//DEBUG
    void tests()
    {
        
        TC_vec U_hat_d = device_allocate<TC>(Nx*My);
        TC_vec U_hat1_d = device_allocate<TC>(Nx*My);
        TC_vec U_hat2_d = device_allocate<TC>(Nx*My);
        T_vec U = (T_vec) malloc(sizeof(T)*Nx*Ny);
        T_vec U1 = (T_vec) malloc(sizeof(T)*Nx*Ny);
        for (int i = 0; i < Nx*Ny; ++i)
        {
            U[i]=T(i)/sqrt(Nx*Ny);
        }
        T_vec U_d = device_allocate<T>(Nx*Ny);
        T_vec U1_d = device_allocate<T>(Nx*Ny);
        host_2_device_cpy<T>(U_d, U, Nx*Ny);        

        CUFFT->fft(U_d, U_hat_d);
        CUFFT->ifft(U_hat_d, U1_d);


        device_2_host_cpy<T>(U1, U1_d, Nx*Ny);
        T residual = 0.0;
        for (int i = 0; i < Nx*Ny; ++i)
        {
            T diff = U1[i]/Nx/Ny-U[i];
            residual+=sqrt(diff*diff);
        }
        

        printf("||U||_2=%le\n",(double)vec_ops_R->normalize(U_d));
        printf("||U||_2=%le\n",(double)vec_ops_R->norm(U_d));
        printf("(U,U1)=%le\n",(double)vec_ops_R->scalar_prod(U1_d,U_d));

        
        CUFFT->fft(U_d, U_hat1_d);
        CUFFT->fft(U_d, U_hat2_d);

        vec_ops_C->assign_mul( (TC)1.0, U_hat_d, (TC)-5.0, U_hat1_d, U_hat2_d);
        
        std::cout << "(complex)-5=" << (TC)-5.0 << std::endl;
        printf("sqrt((U_hat,U_hat))=%le\n",(double)sqrt(vec_ops_C->scalar_prod(U_hat2_d,U_hat2_d).real()) );
        printf("||U_hat||_2=%le\n",(double)vec_ops_C->normalize(U_hat2_d));
        printf("||U_hat||_2=%le\n",(double)vec_ops_C->norm(U_hat2_d));
        std::cout << "||U_hat||_2=" << sqrt(vec_ops_C->scalar_prod(U_hat2_d,U_hat2_d)) << std::endl;

        cudaFree(U_d);
        cudaFree(U1_d);
        cudaFree(U_hat_d);
        cudaFree(U_hat1_d);
        cudaFree(U_hat2_d);
        free(U);
        free(U1);
        printf("resid = %le\n",(double)residual);


        T_vec Laplace_h = (T_vec) malloc(sizeof(T)*Nx*My);
        T_vec biharmonic_h = (T_vec) malloc(sizeof(T)*Nx*My);
        device_2_host_cpy<T>(Laplace_h, Laplace, Nx*My);
        device_2_host_cpy<T>(biharmonic_h, biharmonic, Nx*My);
        
        printf("is valid? %i. ture=%i, false=%i.\n",vec_ops_R->check_is_valid_number(Laplace), true, false);
        file_operations::write_matrix<T>("laplace.dat",  Nx, My, Laplace_h);
        file_operations::write_matrix<T>("biharm.dat",  Nx, My, biharmonic_h);
        
        //Laplace_h[Nx*My/2]=std::numeric_limits<T>::quiet_NaN();
        Laplace_h[Nx*My-1]=std::numeric_limits<T>::infinity();
        host_2_device_cpy<T>(Laplace, Laplace_h, Nx*My);   
        printf("is valid? %i. ture=%i, false=%i.\n",vec_ops_R->check_is_valid_number(Laplace), true, false);

        free(Laplace_h);
        free(biharmonic_h);



    }
//ENDS

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
    dim3 dimGrid;
    dim3 dimBlock;
    dim3 dimGrid_F;
    vector_operations_real *vec_ops_R;
    vector_operations_complex *vec_ops_C;


    size_t Nx, Ny, My;
    cufft_wrap_R2C<T> *CUFFT;
    TC_vec gradient_x=NULL;
    TC_vec gradient_y=NULL;
    T_vec Laplace=NULL;
    T_vec biharmonic=NULL;
    TC_vec u_x_hat=NULL;
    TC_vec u_y_hat=NULL;




    void common_constructor_operation()
    {   
        My=CUFFT->get_reduced_size();
        gradient_x = device_allocate<TC>(Nx*My);
        gradient_y = device_allocate<TC>(Nx*My);
        Laplace = device_allocate<T>(Nx*My);
        biharmonic = device_allocate<T>(Nx*My);
        u_x_hat = device_allocate<TC>(Nx*My);
        u_y_hat = device_allocate<TC>(Nx*My);        
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
        gradient_Fourier<T, TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y);
    }

    void set_Laplace_coefficients()
    {
        Laplace_Fourier<T, TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, Laplace);
    }
   
    void set_biharmonic_coefficients()
    {
        biharmonic_Fourier<T, TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, biharmonic);
    }

};






#endif