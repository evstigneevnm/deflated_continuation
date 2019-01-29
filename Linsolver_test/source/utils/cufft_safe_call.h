#ifndef __CUFFT_SAFE_CALL_H__
#define __CUFFT_SAFE_CALL_H__

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <sstream>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

static const char *_cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "The cuFFT operation was successful";

        case CUFFT_INVALID_PLAN:
            return "cuFFT was passed an invalid plan handle";

        case CUFFT_ALLOC_FAILED:
            return "cuFFT failed to allocate GPU or CPU memory";

        case CUFFT_INVALID_TYPE:
            return "Invalid type";

        case CUFFT_INVALID_VALUE:
            return "User specified an invalid pointer or parameter";

        case CUFFT_INTERNAL_ERROR:
            return "Driver or internal cuFFT library error";

        case CUFFT_EXEC_FAILED:
            return "Failed to execute an FFT on the GPU";

        case CUFFT_SETUP_FAILED:
            return "The cuFFT library failed to initialize";

        case CUFFT_INVALID_SIZE:
            return "User specified an invalid transform size";

        case CUFFT_UNALIGNED_DATA:
            return "Data is unaligned";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "Missing parameters in call";
        
        case CUFFT_INVALID_DEVICE:
            return "Execution of a plan was on different GPU than plan creation";
        
        case CUFFT_PARSE_ERROR:
            return "Internal plan database error";

        case CUFFT_NO_WORKSPACE:
            return "No workspace has been provided prior to plan execution";

    }

    return "<unknown>";
}

#define CUFFT_SAFE_CALL(X)                    \
        do {   \
                cufftResult status = (X);   \
                cudaError_t cuda_res = cudaDeviceSynchronize(); \
                if (status != CUFFT_SUCCESS) { \
                        std::stringstream ss;  \
                        ss << std::string("CUFFT_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") << std::string(_cufftGetErrorEnum(status)); \
                        std::string str = ss.str();  \
                        throw std::runtime_error(str); \
                }      \
                if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CUFFT_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed cudaDeviceSynchronize: ") + std::string(cudaGetErrorString(cuda_res)));   \
        } while (0)



#endif
