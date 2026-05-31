#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <scfd/backend/copy/cuda.h>

#include <common/cpu_vector_operations.h>
#include <common/cuda_init_scfd.h>
#include <common/gpu_vector_operations.h>
#include <common/vector_snapshot_queue.h>
#include <external_libraries/cublas_wrap.h>

namespace
{

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

template<class Queue, class Value>
void push_fixed_two(Queue& queue, Value&& value)
{
    if(queue.size() == 2)
    {
        queue.pop_front();
    }
    queue.push_back(std::forward<Value>(value));
}

template<class T>
void require_host_vector_value(const std::vector<T>& x, T expected, const std::string& label)
{
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        if(x[i] != expected)
        {
            throw std::runtime_error(
                label + ": entry " + std::to_string(i) + " = " +
                std::to_string(static_cast<double>(x[i])) + ", expected " +
                std::to_string(static_cast<double>(expected))
            );
        }
    }
}

template<class T>
void fill_device(T* device, std::size_t n, T value)
{
    std::vector<T> host(n, value);
    scfd::cuda_copy<std::size_t>()(n, host.data(), device);
}

template<class T>
void require_device_vector_value(const T* device, std::size_t n, T expected, const std::string& label)
{
    std::vector<T> host(n);
    scfd::cuda_copy<std::size_t>()(n, device, host.data());
    require_host_vector_value(host, expected, label);
}

template<class T>
void test_metadata_queue()
{
    std::deque<T> lambdas;
    std::deque<std::pair<int, int>> dims;

    push_fixed_two(lambdas, T(10));
    push_fixed_two(dims, std::make_pair(1, 0));
    push_fixed_two(lambdas, T(20));
    push_fixed_two(dims, std::make_pair(2, 1));
    push_fixed_two(lambdas, T(30));
    push_fixed_two(dims, std::make_pair(3, 2));

    require(lambdas.size() == 2, "lambda queue should keep two values");
    require(dims.size() == 2, "dimension queue should keep two values");
    require(lambdas.at(0) == T(20), "oldest lambda after third push is wrong");
    require(lambdas.at(1) == T(30), "newest lambda after third push is wrong");
    require(dims.at(0) == std::make_pair(2, 1), "oldest dimensions after third push are wrong");
    require(dims.at(1) == std::make_pair(3, 2), "newest dimensions after third push are wrong");
}

template<class T>
void test_cpu_snapshots(std::size_t n)
{
    cpu_vector_operations<T> vec_ops(n);
    typename cpu_vector_operations<T>::vector_type x;
    vec_ops.init_vector(x);
    vec_ops.start_use_vector(x);

    common::vector_snapshot_queue<cpu_vector_operations<T>> snapshots(&vec_ops, 2);

    std::fill(x.begin(), x.end(), T(1));
    snapshots.push(x);
    std::fill(x.begin(), x.end(), T(2));
    snapshots.push(x);

    require(snapshots.is_queue_filled(), "CPU snapshot queue should be filled after two pushes");
    std::fill(x.begin(), x.end(), T(3));
    require_host_vector_value(snapshots.at(0), T(1), "CPU first stored vector after source mutation");
    require_host_vector_value(snapshots.at(1), T(2), "CPU second stored vector after source mutation");

    snapshots.push(x);
    std::fill(x.begin(), x.end(), T(4));
    require_host_vector_value(snapshots.at(0), T(2), "CPU oldest vector after third push");
    require_host_vector_value(snapshots.at(1), T(3), "CPU newest vector after third push");

    snapshots.clear();
    require(snapshots.size() == 0, "CPU snapshot queue should be empty after clear");
    require(!snapshots.is_queue_filled(), "CPU snapshot queue should not be filled after clear");

    vec_ops.stop_use_vector(x);
    vec_ops.free_vector(x);
}

template<class T>
void test_gpu_snapshots(std::size_t n, cublas_wrap& cublas)
{
    gpu_vector_operations<T> vec_ops(n, &cublas);
    typename gpu_vector_operations<T>::vector_type x;
    vec_ops.init_vector(x);
    vec_ops.start_use_vector(x);

    common::vector_snapshot_queue<gpu_vector_operations<T>> snapshots(&vec_ops, 2);

    fill_device(x, n, T(1));
    snapshots.push(x);
    fill_device(x, n, T(2));
    snapshots.push(x);

    require(snapshots.is_queue_filled(), "GPU snapshot queue should be filled after two pushes");
    fill_device(x, n, T(3));
    require_device_vector_value(snapshots.at(0), n, T(1), "GPU first stored vector after source mutation");
    require_device_vector_value(snapshots.at(1), n, T(2), "GPU second stored vector after source mutation");

    snapshots.push(x);
    fill_device(x, n, T(4));
    require_device_vector_value(snapshots.at(0), n, T(2), "GPU oldest vector after third push");
    require_device_vector_value(snapshots.at(1), n, T(3), "GPU newest vector after third push");

    snapshots.clear();
    require(snapshots.size() == 0, "GPU snapshot queue should be empty after clear");
    require(!snapshots.is_queue_filled(), "GPU snapshot queue should not be filled after clear");

    vec_ops.stop_use_vector(x);
    vec_ops.free_vector(x);
}

template<class T>
void run_cpu_type_tests(const std::string& type_name)
{
    test_metadata_queue<T>();
    const std::vector<std::size_t> sizes = {1, 7, 1025};
    for(auto n : sizes)
    {
        test_cpu_snapshots<T>(n);
        std::cout << "PASS CPU " << type_name << " n=" << n << std::endl;
    }
}

template<class T>
void run_gpu_type_tests(const std::string& type_name, cublas_wrap& cublas)
{
    const std::vector<std::size_t> sizes = {1, 7, 1025};
    for(auto n : sizes)
    {
        test_gpu_snapshots<T>(n, cublas);
        std::cout << "PASS GPU " << type_name << " n=" << n << std::endl;
    }
}

}

int main(int argc, char** argv)
{
    try
    {
        run_cpu_type_tests<float>("float");
        run_cpu_type_tests<double>("double");

        const std::string device_selector = (argc > 1) ? argv[1] : "auto";
        const int device = common::init_cuda_from_scfd_selector(device_selector);
        std::cout << "CUDA device: " << device << std::endl;

        cublas_wrap cublas;
        run_gpu_type_tests<float>("float", cublas);
        run_gpu_type_tests<double>("double", cublas);

        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }
    catch(const std::exception& e)
    {
        std::cerr << "FAILED: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
