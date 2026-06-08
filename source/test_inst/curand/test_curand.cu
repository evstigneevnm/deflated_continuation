#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <random>
#include <iostream>
#include <thrust/complex.h>

#include <utils/cuda_safe_call.h>
#include <utils/curand_safe_call.h>


template<typename T>
void curandGenerateUniformDistribution(curandGenerator_t gen, T*& vector, size_t size);

template<>
void curandGenerateUniformDistribution<float>(curandGenerator_t gen, float*& vector, size_t size)
{
    CURAND_SAFE_CALL( curandGenerateUniform(gen, vector, size) );
}

template<>
void curandGenerateUniformDistribution<double>(curandGenerator_t gen, double*& vector, size_t size)
{
    CURAND_SAFE_CALL( curandGenerateUniformDouble(gen, vector, size) );
}

template<>
void curandGenerateUniformDistribution<thrust::complex<float>>(curandGenerator_t gen, thrust::complex<float>*& vector, size_t size)
{
    
   std::cout << "complex type not supported yet!\n";
}

template<>
void curandGenerateUniformDistribution<thrust::complex<double>>(curandGenerator_t gen, thrust::complex<double>*& vector, size_t size)
{
    std::cout << "complex type not supported yet!\n";
}



template<class T>
void assign_random(T*& vec, size_t size)
{
    //vec is on the device!!!

    std::random_device r;
    curandGenerator_t gen;
    /* Create pseudo-random number generator */
    CURAND_SAFE_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    /* Set seed */
    CURAND_SAFE_CALL( curandSetPseudoRandomGeneratorSeed(gen, r()) );
    /* Generate n doubles on device */
    curandGenerateUniformDistribution<T>(gen, vec, size);    
    CURAND_SAFE_CALL(curandDestroyGenerator(gen));
}

int main(int argc, char *argv[])
{

    //#define type thrust::complex<double>
    #define type float


    size_t n = 20;
    size_t i;
    
    type *devData, *hostData;

    

    /* Allocate n doubles on host */
    hostData = (type *)calloc(n, sizeof(type));

    /* Allocate n doubles on device */
    CUDA_SAFE_CALL(cudaMalloc((void **)&devData, n*sizeof(type)));
    
    assign_random<type>(devData, n);
        

    /* Copy device memory to host */
    CUDA_SAFE_CALL( cudaMemcpy(hostData, devData, n * sizeof(type), cudaMemcpyDeviceToHost) );

    /* Show result */
    for(i = 0; i < n; i++) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    /* Cleanup */
    CUDA_SAFE_CALL(cudaFree(devData));
    free(hostData);    
    return 0;
}