#ifndef __LINSOLVER_EXTERNAL_LAPACK_WRAP_H__
#define __LINSOLVER_EXTERNAL_LAPACK_WRAP_H__

#include <cuda_runtime.h>
#include <scfd/external_libraries/lapack_wrap.h>
#include <scfd/external_libraries/lapack_wrap_device.h>
#include <scfd/utils/cuda_safe_call.h>

namespace linsolver_external_libraries_compat
{
struct cuda_memory
{
    using pointer_type = void *;
    using const_pointer_type = const void *;

    static const bool is_host_visible = false;

    static void copy_to_host( size_t size, const_pointer_type src, pointer_type dst )
    {
        CUDA_SAFE_CALL( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost ) );
    }
};

struct cuda_backend
{
    using memory_type = cuda_memory;
};
}

template <class T>
using lapack_wrap = scfd::lapack_wrap_device<linsolver_external_libraries_compat::cuda_backend, T>;

#endif
