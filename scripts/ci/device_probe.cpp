// A working compiler/driver alone does not establish that a device kernel ran.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define GPU(call) hip##call
#else
#include <cuda_runtime.h>
#define GPU(call) cuda##call
#endif
#include <cstdio>

#define CHECK(call) do { auto status = GPU(call); if (status != GPU(Success)) { \
    std::fprintf(stderr, "%s: %s\n", #call, GPU(GetErrorString)(status)); return 1; } } while (false)

__global__ void write_result(int *value) { *value = 42; }

int main()
{
    int *device = nullptr, result = 0;
    CHECK(Malloc(reinterpret_cast<void **>(&device), sizeof(result)));
    write_result<<<1, 1>>>(device);
    CHECK(GetLastError());
    CHECK(DeviceSynchronize());
    CHECK(Memcpy(&result, device, sizeof(result), GPU(MemcpyDeviceToHost)));
    CHECK(Free(device));
    if (result != 42) { std::fprintf(stderr, "Wrong GPU result: %d\n", result); return 1; }
}
