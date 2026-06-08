#ifndef __CURAND_SAFE_CALL_H__
#define __CURAND_SAFE_CALL_H__

#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <string>
#include <sstream>


#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

static const char *_curandGetErrorEnum(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "The operation completed successfully.";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "Header file and linked library version do not match.";    
        case CURAND_STATUS_NOT_INITIALIZED:
            return "Generator not initialized";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case CURAND_STATUS_TYPE_ERROR:
            return "Generator is wrong type";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "Argument out of range";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "Length requested is not a multple of dimension";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "GPU does not have double precision required by MRG32k3a";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "Kernel launch failure";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "Preexisting failure on library entry";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "Initialization of CUDA failed";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "Architecture mismatch, GPU does not support requested feature";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "Internal library error";
    }
    return "<unknown>";
}
  
#define CURAND_SAFE_CALL(X)                    \
        do {   \
                curandStatus_t status = (X);   \
                cudaError_t cuda_res = cudaDeviceSynchronize(); \
                if (status != CURAND_STATUS_SUCCESS) { \
                        std::stringstream ss;  \
                        ss << std::string("CURAND_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") << std::string(_curandGetErrorEnum(status)); \
                        std::string str = ss.str();  \
                        throw std::runtime_error(str); \
                }      \
                if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CURAND_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed cudaDeviceSynchronize: ") + std::string(cudaGetErrorString(cuda_res)));   \
        } while (0)  
  

  
#endif 