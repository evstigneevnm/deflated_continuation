#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D_ker.h>


template<typename T_C>
__global__ void gradient_Fourier_kernel(int Nx, int My, T_C *gradient_x, T_C *gradient_y)
{

    unsigned int j=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k=blockDim.y * blockIdx.y + threadIdx.y;
    
    if((j>=Nx)||(k>=My)) return;

    int m=j;
    if(j>=Nx/2)
        m=j-Nx;
    
    int n=k;
    
    gradient_x[I2(j,k,Nx)]=T_C(0,m);
    gradient_y[I2(j,k,Nx)]=T_C(0,n);

}


template<typename T_C>
__global__ void Laplace_Fourier_kernel(int Nx, int My, T_C *gradient_x, T_C *gradient_y, T_C *Laplace)
{
    //result of grad_x^2 + grad_y^2 is pure real
    //we use T_C(*.real,0.0) to get rid of rounding errors for float type

    unsigned int j=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k=blockDim.y * blockIdx.y + threadIdx.y;
    
    if((j>=Nx)||(k>=My)) return;

    T_C x2 = gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)];
    T_C y2 = gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)];

    Laplace[I2(j,k,Nx)] = T_C(x2.real() + y2.real(),0.0);

}

template<typename T_C>
__global__ void biharmonic_Fourier_kernel(int Nx, int My, T_C *gradient_x, T_C *gradient_y, T_C *biharmonic)
{

    //result of grad_x^4 + grad_y^4 + 2*grad_x^2*grad_y^2 is pure real
    //we use T_C(*.real,0.0) to get rid of rounding errors for float type

    unsigned int j=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k=blockDim.y * blockIdx.y + threadIdx.y;
    
    if((j>=Nx)||(k>=My)) return;

    T_C x4 = gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)];
    T_C y4 = gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)];
    T_C x2y2 = gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)];
    
    biharmonic[I2(j,k,Nx)] = T_C(x4.real()+2.0*x2y2.real()+y4.real(),0.0);

}





template<typename T_C>
void gradient_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y)
{
    gradient_Fourier_kernel<T_C><<<dimGrid, dimBlock>>>(Nx,My,gradient_x,gradient_y);
}



template<typename T_C>
void Laplace_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y, T_C*& Laplce)
{
    Laplace_Fourier_kernel<T_C><<<dimGrid, dimBlock>>>(Nx,My,gradient_x, gradient_y, Laplce);
}



template<typename T_C>
void biharmonic_Fourier(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y, T_C*& biharmonic)
{
    biharmonic_Fourier_kernel<T_C><<<dimGrid, dimBlock>>>(Nx,My,gradient_x, gradient_y, biharmonic);
}

//explicit instantiation
template void gradient_Fourier<thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y);
template void gradient_Fourier<thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y);
template void Laplace_Fourier<thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y, thrust::complex<float>*& Laplce);
template void Laplace_Fourier<thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y, thrust::complex<double>*& Laplce);
template void biharmonic_Fourier<thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y, thrust::complex<float>*& biharmonic);
template void biharmonic_Fourier<thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y, thrust::complex<double>*& biharmonic);