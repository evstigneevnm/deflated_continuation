#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D_ker.h>


template<typename T_C>
__global__ void gradient_Fourier_kernel(int Nx, int My, T_C *gradient_x, T_C *gradient_y)
{

    int j=blockDim.x * blockIdx.x + threadIdx.x;
    int k=blockDim.y * blockIdx.y + threadIdx.y;
    
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

    int j=blockDim.x * blockIdx.x + threadIdx.x;
    int k=blockDim.y * blockIdx.y + threadIdx.y;
    
    if((j>=Nx)||(k>=My)) return;

    T_C x2 = gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)];
    T_C y2 = gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)];

    Laplace[I2(j,k,Nx)] = T_C(x2.real() + y2.real(), 0.0);

}

template<typename T_C>
__global__ void biharmonic_Fourier_kernel(int Nx, int My, T_C *gradient_x, T_C *gradient_y, T_C *biharmonic)
{

    //result of grad_x^4 + grad_y^4 + 2*grad_x^2*grad_y^2 is pure real
    //we use T_C(*.real,0.0) to get rid of rounding errors for float type

    int j=blockDim.x * blockIdx.x + threadIdx.x;
    int k=blockDim.y * blockIdx.y + threadIdx.y;
    
    if((j>=Nx)||(k>=My)) return;

    T_C x4 = gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)];
    T_C y4 = gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)];
    T_C x2y2 = gradient_x[I2(j,k,Nx)]*gradient_x[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)]*gradient_y[I2(j,k,Nx)];
    
    biharmonic[I2(j,k,Nx)] = T_C(x4.real()+2.0*x2y2.real()+y4.real(), 0.0);

}


template<typename TC, typename T_vec_im, typename TC_vec>
__global__ void C2R_kernel(size_t N, TC_vec arrayC, T_vec_im arrayR_im)
{


    unsigned int j=blockDim.x * blockIdx.x + threadIdx.x;
    
    if(j>=N) return;

    arrayR_im[j] = arrayC[j+1].imag();

}


template<typename TC, typename T_vec_im, typename TC_vec>
__global__ void R2C_kernel(size_t N, T_vec_im arrayR_im, TC_vec arrayC)
{


    unsigned int j=blockDim.x * blockIdx.x + threadIdx.x;
    
    if(j>=N) return;
    
    TC val = TC(0, arrayR_im[j]);
    arrayC[j+1] = val;

    arrayC[0]=TC(0,0);

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


template<typename TC, typename T_vec_im, typename TC_vec>
void C2R_(unsigned int BLOCK_SIZE, size_t Nx, size_t My, TC_vec& arrayC, T_vec_im& arrayR_im)
{
    size_t N = Nx*My-1;
    dim3 threads(BLOCK_SIZE);
    int blocks_x=(N+BLOCK_SIZE)/BLOCK_SIZE;
    dim3 blocks(blocks_x);

    C2R_kernel<TC, T_vec_im, TC_vec><<< blocks, threads>>>(N, arrayC, arrayR_im);

}

template<typename TC, typename T_vec_im, typename TC_vec>
void R2C_(unsigned int BLOCK_SIZE, size_t Nx, size_t My, T_vec_im& arrayR_im, TC_vec& arrayC)
{
    size_t N = Nx*My-1;
    dim3 threads(BLOCK_SIZE);
    int blocks_x=(N+BLOCK_SIZE)/BLOCK_SIZE;
    dim3 blocks(blocks_x);

    R2C_kernel<TC, T_vec_im, TC_vec><<< blocks, threads>>>(N, arrayR_im, arrayC);

}

//explicit instantiation
template void gradient_Fourier<thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y);
template void gradient_Fourier<thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y);
template void Laplace_Fourier<thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y, thrust::complex<float>*& Laplce);
template void Laplace_Fourier<thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y, thrust::complex<double>*& Laplce);
template void biharmonic_Fourier<thrust::complex<float> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y, thrust::complex<float>*& biharmonic);
template void biharmonic_Fourier<thrust::complex<double> >(dim3 dimGrid, dim3 dimBlock, size_t Nx, size_t My, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y, thrust::complex<double>*& biharmonic);

template void C2R_<thrust::complex<float>, float*, thrust::complex<float>* >(unsigned int BLOCK_SIZE, size_t Nx, size_t My, thrust::complex<float>*& arrayC, float*& arrayR_im);
template void C2R_<thrust::complex<double>, double*, thrust::complex<double>* >(unsigned int BLOCK_SIZE, size_t Nx, size_t My, thrust::complex<double>*& arrayC, double*& arrayR_im);

template void R2C_<thrust::complex<float>, float*, thrust::complex<float>* >(unsigned int BLOCK_SIZE, size_t Nx, size_t My, float*& arrayR_im, thrust::complex<float>*& arrayC);
template void R2C_<thrust::complex<double>, double*, thrust::complex<double>* >(unsigned int BLOCK_SIZE, size_t Nx, size_t My, double*& arrayR_im, thrust::complex<double>*& arrayC);
