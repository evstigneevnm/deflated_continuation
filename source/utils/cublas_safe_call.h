#ifndef __CUBLAS_SAFE_CALL_H__
#define __CUBLAS_SAFE_CALL_H__


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <sstream>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "The operation completed successfully.";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup.";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "An unsupported value or parameter was passed to the function (a negative vector size, for example).";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";

        case CUBLAS_STATUS_EXECUTION_FAILED:   
            return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "The functionnality requested is not supported.";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.";

    }

    return "<unknown>";
}

#define CUBLAS_SAFE_CALL(X)                    \
        do {   \
                cublasStatus_t status = (X);   \
                cudaError_t cuda_res = cudaDeviceSynchronize(); \
                if (status != CUBLAS_STATUS_SUCCESS) { \
                        std::stringstream ss;  \
                        ss << std::string("CUBLAS_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") << std::string(_cublasGetErrorEnum(status)); \
                        std::string str = ss.str();  \
                        throw std::runtime_error(str); \
                }      \
                if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CUBLAS_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed cudaDeviceSynchronize: ") + std::string(cudaGetErrorString(cuda_res)));   \
        } while (0)




#endif  