#include <cufft_test_kernels.h>




template <typename T_R, typename T_C>
__global__ void kernel_set_values(size_t N, T_C *array)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        array[j].x=sin(3.14159265358979*T_R(j)/N);
        array[j].y=T_R(0.0);
    }

}

template <typename T_R, typename T_C>
__global__ void kernel_check_values(size_t N, T_C *array1, T_C *array2)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        array1[j].x-=array2[j].x/N;
        array1[j].y-=array2[j].y/N;


    }

}

template <typename T_R, typename T_C>
__global__ void kernel_convert_2R_values(size_t N, T_C* arrayC, T_R* arrayRR, T_R* arrayRI)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        arrayRR[j] = arrayC[j].x;
        arrayRI[j] = arrayC[j].y;


    }

}


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
__global__ void gradient_Fourier_kernel(int Nx, int My, int Ny, T_C *gradient_x, T_C *gradient_y)
{

    unsigned int j=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k=blockDim.y * blockIdx.y + threadIdx.y;
    
    if((j>=Nx)||(k>=Ny)) return;

    int m=j;
    if(j>=Nx/2)
        m=j-Nx;
    
    int n=k;
    if(k>=Ny/2)
        n=k-Ny;
    
    gradient_x[I2(j,k,Nx)]=T_C(0,m);
    gradient_y[I2(j,k,Nx)]=T_C(0,n);

}


template <typename T_R, typename T_C>
__global__ void complexreal_to_real_kernel(int N, const T_C* arrayC, T_R* arrayRR)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        arrayRR[j] = arrayC[j].real();
    }

}


//========= kernels wraps ============

template <typename T_R, typename T_C>
void set_values(size_t N, T_C* array)
{

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlock(BLOCKSIZE, 1);
    dim3 dimGrid( k1, 1 );
    kernel_set_values<T_R,T_C><<<dimGrid, dimBlock>>>(N, array);

}


template <typename T_R, typename T_C>
void check_values(size_t N, T_C* array1, T_C* array2)
{

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlock(BLOCKSIZE, 1);
    dim3 dimGrid( k1, 1 );
    kernel_check_values<T_R,T_C><<<dimGrid, dimBlock>>>(N, array1, array2);

}


template <typename T_R, typename T_C>
void convert_2R_values(size_t N, T_C* arrayC, T_R* arrayRR, T_R* arrayRI)
{

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlock(BLOCKSIZE, 1);
    dim3 dimGrid( k1, 1 );
    kernel_convert_2R_values<T_R,T_C><<<dimGrid, dimBlock>>>(N, arrayC, arrayRR, arrayRI);

}

template<typename T_C>
void gradient_Fourier(size_t Nx, size_t My, T_C*& gradient_x, T_C*& gradient_y)
{
    dim3 dimBlock(64, 16);
    unsigned int blocks_x=floor(Nx/( 64 ))+1;
    unsigned int blocks_y=floor(My/( 16 ))+1;
    dim3 dimGrid( blocks_x, blocks_y);
    gradient_Fourier_kernel<T_C><<<dimGrid, dimBlock>>>(Nx,My,gradient_x,gradient_y);
}

template<typename T_C>
void gradient_Fourier(size_t Nx, size_t My, size_t Ny, T_C*& gradient_x, T_C*& gradient_y)
{
    dim3 dimBlock(64, 16);
    unsigned int blocks_x=floor(Nx/( 64 ))+1;
    unsigned int blocks_y=floor(Ny/( 16 ))+1;
    dim3 dimGrid( blocks_x, blocks_y);
    gradient_Fourier_kernel<T_C><<<dimGrid, dimBlock>>>(Nx,My,Ny,gradient_x,gradient_y);
}

template <typename T_R, typename T_C>
void complexreal_to_real(size_t N, const T_C*& arrayC, T_R*& arrayRR)
{

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlock(BLOCKSIZE, 1);
    dim3 dimGrid( k1, 1 );
    complexreal_to_real_kernel<T_R,T_C><<<dimGrid, dimBlock>>>(N, arrayC, arrayRR);

}

//all precompile variants:
template void set_values<double, cufftDoubleComplex>(size_t N, cufftDoubleComplex* t);
template void set_values<float, cufftComplex>(size_t N, cufftComplex* t);
template void check_values<double, cufftDoubleComplex>(size_t N, cufftDoubleComplex* t1, cufftDoubleComplex* t2);
template void check_values<float, cufftComplex>(size_t N, cufftComplex* t1, cufftComplex* t2);
template void convert_2R_values<double, cufftDoubleComplex>(size_t N, cufftDoubleComplex* t, double* t1, double* t2);
template void convert_2R_values<float, cufftComplex>(size_t N, cufftComplex* t, float* t1, float* t2);
template void gradient_Fourier<thrust::complex<float> >(size_t Nx, size_t My, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y);
template void gradient_Fourier<thrust::complex<double> >(size_t Nx, size_t My, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y);
template void gradient_Fourier<thrust::complex<float> >(size_t Nx, size_t My, size_t Ny, thrust::complex<float>*& gradient_x, thrust::complex<float>*& gradient_y);
template void gradient_Fourier<thrust::complex<double> >(size_t Nx, size_t My, size_t Ny, thrust::complex<double>*& gradient_x, thrust::complex<double>*& gradient_y);
template void complexreal_to_real<float, thrust::complex<float> >(size_t N, const thrust::complex<float>*& arrayC, float*& arrayRR);
template void complexreal_to_real<double, thrust::complex<double> >(size_t N, const thrust::complex<double>*& arrayC, double*& arrayRR);