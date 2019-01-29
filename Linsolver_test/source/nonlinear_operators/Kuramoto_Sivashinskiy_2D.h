#ifndef __KURAMOTO_SIVASHINSKIY_2D__
#define __KURAMOTO_SIVASHINSKIY_2D__

#include <external_libraries/cufft_wrap.h>
#include <thrust/complex.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D_ker.h>
//for debug!
#include "file_operations.h"
#include <limits>

template<typename VectorOperations, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
class Kuramoto_Sivashinskiy_2D
{
public:
    
    typedef VectorOperations vector_operations_type;
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef thrust::complex<T> thrust_complex_type;    


    Kuramoto_Sivashinskiy_2D(size_t Nx_, size_t Ny_, vector_operations_type *vec_ops_, cufft_wrap_R2C<T> *CUFFT_): Nx(Nx_), Ny(Ny_), vec_ops(vec_ops_), CUFFT(CUFFT_)
    {
        common_constructor_operation();
        calculate_cuda_grid();
        init_all_derivatives();
    }

    Kuramoto_Sivashinskiy_2D(size_t Nx_, size_t Ny_, dim3 dimGrid_, dim3 dimGrid_F_, dim3 dimBlock_, vector_operations_type *vec_ops_, cufft_wrap_R2C<T> &CUFFT_): Nx(Nx_), Ny(Ny_), dimGrid(dimGrid_), dimGrid_F(dimGrid_F_), dimBlock(dimBlock_), vec_ops(vec_ops_), CUFFT(CUFFT_)
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
    }


    void construct_derivatives()
    {

    }

    void tests()
    {
        
        thrust_complex_type *U_hat_d = device_allocate<thrust_complex_type>(Nx*My);
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
        cudaFree(U_d);
        cudaFree(U1_d);
        cudaFree(U_hat_d);
        free(U);
        free(U1);
        printf("resid = %le\n",(double)residual);


        T_vec Laplace_h = (T_vec) malloc(sizeof(T)*Nx*My);
        T_vec biharmonic_h = (T_vec) malloc(sizeof(T)*Nx*My);
        device_2_host_cpy<T>(Laplace_h, Laplace, Nx*My);
        device_2_host_cpy<T>(biharmonic_h, biharmonic, Nx*My);
        
        printf("is valid? %i. ture=%i, false=%i.\n",vec_ops->check_is_valid_number(Laplace), true, false);
        file_operations::write_matrix<T>("laplace.dat",  Nx, My, Laplace_h);
        file_operations::write_matrix<T>("biharm.dat",  Nx, My, biharmonic_h);
        
        //Laplace_h[Nx*My/2]=std::numeric_limits<T>::quiet_NaN();
        Laplace_h[Nx*My-1]=std::numeric_limits<T>::infinity();
        host_2_device_cpy<T>(Laplace, Laplace_h, Nx*My);   
        printf("is valid? %i. ture=%i, false=%i.\n",vec_ops->check_is_valid_number(Laplace), true, false);

        free(Laplace_h);
        free(biharmonic_h);



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
    dim3 dimGrid;
    dim3 dimBlock;
    dim3 dimGrid_F;
    vector_operations_type *vec_ops;


    size_t Nx, Ny, My;
    cufft_wrap_R2C<T> *CUFFT;
    thrust_complex_type *gradient_x=NULL;
    thrust_complex_type *gradient_y=NULL;
    T *Laplace=NULL;
    T *biharmonic=NULL;




    void common_constructor_operation()
    {   
        My=CUFFT->get_reduced_size();
        gradient_x = device_allocate<thrust_complex_type>(Nx*My);
        gradient_y = device_allocate<thrust_complex_type>(Nx*My);
        Laplace = device_allocate<T>(Nx*My);
        biharmonic = device_allocate<T>(Nx*My);
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
        gradient_Fourier<T, thrust_complex_type>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y);
    }

    void set_Laplace_coefficients()
    {
        Laplace_Fourier<T, thrust_complex_type>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, Laplace);
    }
   
    void set_biharmonic_coefficients()
    {
        biharmonic_Fourier<T, thrust_complex_type>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, biharmonic);
    }

};






#endif