#ifndef __CUSOLVER_SAFE_CALL_H__
#define __CUSOLVER_SAFE_CALL_H__

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>
#include <sstream>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "The operation completed successfully.";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "the matrix type is not supported.";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "The cusolver library was not initialized. This is usually caused by the lack of a prior cusolverCreate() call, an error in the CUDA Runtime API called by the cusolver routine, or an error in the hardware setup.";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "Resource allocation failed inside the cusolver library. This is usually caused by a cudaMalloc() failure.";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "An unsupported value or parameter was passed to the function (a negative vector size, for example).";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.";

        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";

        case CUSOLVER_STATUS_EXECUTION_FAILED:   
            return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.";

        case CUSOLVER_STATUS_NOT_SUPPORTED:
            return "The functionnality requested is not supported.";

        case CUSOLVER_STATUS_ZERO_PIVOT:
            return "a zero pivot was encountered during the computation.";
        case CUSOLVER_STATUS_INVALID_LICENSE:
            return "invalid licence?!?";
        case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
            return "The configuration parameter gels_irs_params structure was not created.";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
            return "One of the configuration parameter in the gels_irs_params structure is not valid.";
        case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
            return "One of the configuration parameter in the gels_irs_params structure is not supported. For example if nrhs >1, and refinement solver was set to CUSOLVER_IRS_REFINE_GMRES";
        case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
            return "Numerical error related to niters <0, seeniters description for more details.";
        case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
            return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
        case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
            return "The information structure gesv_irs_infos was not created.";
    }

    return "<unknown>";
}

#define CUSOLVER_SAFE_CALL(X)                    \
        do {   \
                cusolverStatus_t status = (X);   \
                cudaError_t cuda_res = cudaDeviceSynchronize(); \
                if (status != CUSOLVER_STATUS_SUCCESS) { \
                        std::stringstream ss;  \
                        ss << std::string("CUSOLVER_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") << std::string(_cusolverGetErrorEnum(status)); \
                        std::string str = ss.str();  \
                        throw std::runtime_error(str); \
                }      \
                if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CUSOLVER_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed cudaDeviceSynchronize: ") + std::string(cudaGetErrorString(cuda_res)));   \
        } while (0)





#endif
