#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <common/scfd_serial_cpu_vector_operations.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>

namespace
{

int checks = 0;
int failures = 0;

using real = double;
using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
using adapter_t = symmetry::fourier::real_packed_fourier_slice_1d_adapter<vec_ops_t>;
using vector_t = typename vec_ops_t::vector_type;

void require_true(const std::string& label, const bool value)
{
    ++checks;
    if(!value)
    {
        ++failures;
        std::cerr << "FAIL " << label << std::endl;
    }
}

void require_close(const std::string& label, const real value, const real expected, const real tolerance)
{
    ++checks;
    const real error = std::abs(value - expected);
    if(error > tolerance)
    {
        ++failures;
        std::cerr << "FAIL " << label
                  << " value=" << value
                  << " expected=" << expected
                  << " error=" << error
                  << " tolerance=" << tolerance << std::endl;
    }
}

void set_vector(vec_ops_t& vec_ops, vector_t& x, const std::vector<real>& values)
{
    vec_ops.set(values.data(), x, values.size());
}

std::vector<real> get_vector(vec_ops_t& vec_ops, const vector_t& x)
{
    std::vector<real> values(vec_ops.get_size(x), real(0));
    vec_ops.get(x, values.data(), values.size());
    return values;
}

void require_vector_close(
    vec_ops_t& vec_ops,
    const std::string& label,
    const vector_t& value,
    const vector_t& expected,
    const real tolerance)
{
    const auto value_host = get_vector(vec_ops, value);
    const auto expected_host = get_vector(vec_ops, expected);
    require_true(label + " size", value_host.size() == expected_host.size());
    const std::size_t n = std::min(value_host.size(), expected_host.size());
    for(std::size_t i = 0; i < n; ++i)
    {
        require_close(label + " [" + std::to_string(i) + "]", value_host[i], expected_host[i], tolerance);
    }
}

struct vector_bundle
{
    vector_t x;
    vector_t y;
    vector_t z;
    vector_t w;
};

void init_bundle(vec_ops_t& vec_ops, vector_bundle& bundle)
{
    vec_ops.init_vector(bundle.x);
    vec_ops.init_vector(bundle.y);
    vec_ops.init_vector(bundle.z);
    vec_ops.init_vector(bundle.w);
    vec_ops.start_use_vector(bundle.x);
    vec_ops.start_use_vector(bundle.y);
    vec_ops.start_use_vector(bundle.z);
    vec_ops.start_use_vector(bundle.w);
}

void free_bundle(vec_ops_t& vec_ops, vector_bundle& bundle)
{
    vec_ops.stop_use_vector(bundle.w);
    vec_ops.free_vector(bundle.w);
    vec_ops.stop_use_vector(bundle.z);
    vec_ops.free_vector(bundle.z);
    vec_ops.stop_use_vector(bundle.y);
    vec_ops.free_vector(bundle.y);
    vec_ops.stop_use_vector(bundle.x);
    vec_ops.free_vector(bundle.x);
}

void test_shifted_copies_have_one_canonical_representative()
{
    vec_ops_t vec_ops(8);
    adapter_t adapter(&vec_ops, 4);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {1.0, 0.3, -0.2, 0.5, 0.05, -0.1, 0.03, 0.02});
    adapter.apply_shift(v.x, v.y, 0.73);
    adapter.stabilize_canonical(v.x, v.z);
    adapter.stabilize_canonical(v.y, v.w);

    require_vector_close(vec_ops, "canonical shifted copy", v.z, v.w, 2e-12);
    const auto host = get_vector(vec_ops, v.z);
    require_true("canonical selected mode 1", adapter.last_slice_data().mode == 1);
    require_true("canonical mode 1 real positive", host[0] > 0.0);
    require_close("canonical mode 1 imag", host[1], 0.0, 1e-12);

    free_bundle(vec_ops, v);
}

void test_canonical_ignores_chart_history()
{
    vec_ops_t vec_ops(8);
    adapter_t adapter(&vec_ops, 4);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0});
    adapter.stabilize(v.x, v.y);
    require_true("history selected mode 3", adapter.last_slice_data().mode == 3);

    set_vector(vec_ops, v.z, {1.0, 0.2, 0.1, -0.3, 0.5, 0.25, -0.05, 0.07});
    adapter.stabilize_canonical(v.z, v.w);
    const auto canonical = get_vector(vec_ops, v.w);
    require_true("canonical ignores preferred mode and selects mode 1", adapter.last_slice_data().mode == 1);
    require_true("canonical mode 1 real positive after history", canonical[0] > 0.0);
    require_close("canonical mode 1 imag after history", canonical[1], 0.0, 1e-12);

    free_bundle(vec_ops, v);
}

void test_residual_group_representatives_collapse()
{
    vec_ops_t vec_ops(8);
    adapter_t adapter(&vec_ops, 4);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {0.0, 0.0, 1.0, 0.4, 0.35, -0.2, -0.1, 0.07});
    adapter.apply_shift(v.x, v.y, 0.37);
    adapter.stabilize_canonical(v.x, v.z);
    adapter.stabilize_canonical(v.y, v.w);

    require_true("canonical selected first active mode 2", adapter.last_slice_data().mode == 2);
    require_vector_close(vec_ops, "canonical arbitrary shift with mode 2 active", v.z, v.w, 2e-12);
    const auto canonical = get_vector(vec_ops, v.z);
    require_true("canonical mode 2 real positive", canonical[2] > 0.0);
    require_close("canonical mode 2 imag", canonical[3], 0.0, 1e-12);

    adapter.apply_shift(v.x, v.y, std::acos(real(-1)));
    adapter.stabilize_canonical(v.y, v.w);
    require_vector_close(vec_ops, "canonical residual group copy", v.z, v.w, 2e-12);

    free_bundle(vec_ops, v);
}

} // namespace

int main()
{
    test_shifted_copies_have_one_canonical_representative();
    test_canonical_ignores_chart_history();
    test_residual_group_representatives_collapse();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
