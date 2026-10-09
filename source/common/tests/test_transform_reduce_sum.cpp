#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#ifdef TEST_VECTOR_BACKEND_CUDA
#include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda;
#elif defined(TEST_VECTOR_BACKEND_OMP)
#include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp;
#else
#include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu;
#endif
#include <common/scfd_vector_operations.h>
#include <nmfd/detail/vector_wrap.h>

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template<class T>
struct identity_mapping
{
    __DEVICE_TAG__ T operator()(T value) const
    {
        return value;
    }
};

template<class T>
struct product_mapping
{
    __DEVICE_TAG__ T operator()(T x, T y) const
    {
        return x * y;
    }

    __DEVICE_TAG__ T operator()(T x, T y, T z) const
    {
        return x * y + T(.5) * z;
    }
};

template<class T>
void check_sum()
{
    using operations_type = scfd_vector_operations<backend_type, T>;
    operations_type operations(3);
    nmfd::detail::vector_wrap<operations_type> x(operations), y(operations), z(operations);
    x.start_use();
    y.start_use();
    z.start_use();
    const scfd::static_vec::vec<T, 3> hx{1, -2, 3}, hy{4, 5, -6}, hz{7, 8, 9};
    operations.set(hx.d, *x);
    operations.set(hy.d, *y);
    operations.set(hz.d, *z);
    require(operations.transform_reduce_sum(identity_mapping<T>{}, *x) == 2, "Unary transformed sum");
    require(operations.transform_reduce_sum(product_mapping<T>{}, *x, *y) == -24, "Binary transformed sum");
    require(operations.transform_reduce_sum(product_mapping<T>{}, *x, *y, *z) == -12, "Ternary transformed sum");
    require(operations.transform_reduce_max(product_mapping<T>{}, *x, *y, *z) == T(7.5),
        "Shared transform kernel preserves maximum reduction");
    {
        const auto view = operations.view(*x);
        require(view(0) == hx[0] && view(1) == hx[1] && view(2) == hx[2], "Transform reductions preserve inputs");
    }

    const T large = std::is_same_v<T, float> ? T(1e8) : T(1e16);
    const scfd::static_vec::vec<T, 3> cancellation{large, 1, -large};
    operations.set(cancellation.d, *x);
    operations.use_high_precision();
    require(operations.transform_reduce_sum(identity_mapping<T>{}, *x) == 1 &&
                operations.last_reduction_used_high_precision(),
        "Transformed sum honors compensated reduction");
    operations.set_regular_precision();
    operations.assign_scalar(1, *x);
    require(operations.transform_reduce_sum(identity_mapping<T>{}, *x) == 3 &&
                !operations.last_reduction_used_high_precision(),
        "Transformed sum restores regular reduction");

    for (const T invalid : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
    {
        for (int index = 0; index < 3; ++index)
        {
            scfd::static_vec::vec<T, 3> values{1, 2, 3};
            values[index] = invalid;
            operations.set(values.d, *x);
            const auto sum = operations.transform_reduce_sum(identity_mapping<T>{}, *x);
            require(std::isnan(invalid) ? std::isnan(sum) : std::isinf(sum), "Invalid components propagate in sum");
        }
    }
    operations_type other(2);
    nmfd::detail::vector_wrap<operations_type> mismatched(other);
    mismatched.start_use();
    bool refused = false;
    try
    {
        operations.transform_reduce_sum(product_mapping<T>{}, *x, *mismatched);
    }
    catch (const std::invalid_argument&)
    {
        refused = true;
    }
    require(refused, "Transformed sum rejects mismatched vector dimensions");

    operations_type empty_operations(0);
    typename operations_type::vector_type empty;
    require(empty_operations.transform_reduce_sum(identity_mapping<T>{}, empty) == 0,
        "Empty transformed sum returns the additive identity");
}

int main()
try
{
    backend_type::init_device();
    check_sum<float>();
    check_sum<double>();
    std::cout << "Generic transformed sum, precision, invalid values, and dimensions: PASS\n";
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
