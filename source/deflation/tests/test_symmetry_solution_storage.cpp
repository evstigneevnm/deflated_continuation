#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <common/scfd_serial_cpu_vector_operations.h>
#include <deflation/symmetry_solution_storage.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>

namespace
{

int checks = 0;
int failures = 0;

void require_close(const char* label, const double value, const double expected, const double tolerance)
{
    ++checks;
    const double error = std::abs(value - expected);
    if(error > tolerance)
    {
        ++failures;
        std::cerr << "FAIL " << label << " value=" << value
                  << " expected=" << expected
                  << " error=" << error
                  << " tolerance=" << tolerance << std::endl;
    }
}

void require_true(const char* label, const bool value)
{
    ++checks;
    if(!value)
    {
        ++failures;
        std::cerr << "FAIL " << label << std::endl;
    }
}

template<class VecOps>
void set_vector(VecOps& vec_ops, typename VecOps::vector_type& x, const std::vector<typename VecOps::scalar_type>& values)
{
    vec_ops.set(values.data(), x, values.size());
}

template<class VecOps>
std::vector<typename VecOps::scalar_type> get_vector(VecOps& vec_ops, const typename VecOps::vector_type& x)
{
    std::vector<typename VecOps::scalar_type> values(vec_ops.get_size(x), typename VecOps::scalar_type(0));
    vec_ops.get(x, values.data(), values.size());
    return values;
}

} // namespace

int main()
{
    using real = double;
    using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
    using adapter_t = symmetry::fourier::real_packed_fourier_slice_1d_adapter<vec_ops_t>;
    using storage_t = deflation::symmetry_solution_storage<vec_ops_t, adapter_t>;

    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    storage_t storage(&vec_ops, 8, real(1), real(2), &adapter, 1e-10);

    vec_ops_t::vector_type x;
    vec_ops_t::vector_type shifted;
    vec_ops_t::vector_type stabilized;
    vec_ops_t::vector_type query;
    vec_ops_t::vector_type shifted_query;
    vec_ops_t::vector_type c;
    vec_ops.init_vector(x);
    vec_ops.init_vector(shifted);
    vec_ops.init_vector(stabilized);
    vec_ops.init_vector(query);
    vec_ops.init_vector(shifted_query);
    vec_ops.init_vector(c);
    vec_ops.start_use_vector(x);
    vec_ops.start_use_vector(shifted);
    vec_ops.start_use_vector(stabilized);
    vec_ops.start_use_vector(query);
    vec_ops.start_use_vector(shifted_query);
    vec_ops.start_use_vector(c);

    set_vector(vec_ops, x, {1.0, 0.0, 0.25, 0.5});
    adapter.apply_shift(x, shifted, 0.73);
    vec_ops.assign(shifted, stabilized);
    storage.stabilize_in_place(stabilized);
    const auto stabilized_host = get_vector(vec_ops, stabilized);
    require_close("stabilized shifted mode1 real", stabilized_host[0], 1.0, 1e-12);
    require_close("stabilized shifted mode1 imag", stabilized_host[1], 0.0, 1e-12);
    require_close("stabilized shifted mode2 real", stabilized_host[2], 0.25, 1e-12);
    require_close("stabilized shifted mode2 imag", stabilized_host[3], 0.5, 1e-12);

    storage.push_back(x);
    storage.push_back(shifted);
    require_true("translated copy is skipped as duplicate", storage.get_size() == 1);

    set_vector(vec_ops, query, {1.1, 0.0, 0.25, 0.5});
    storage.set_ignore_zero();
    real beta = real(0);
    storage.calc_distance(query, beta, c);
    require_close("deflation beta in slice coordinates", beta, 401.0, 1e-10);
    const auto c_host = get_vector(vec_ops, c);
    require_close("deflation gradient mode1 real", c_host[0], 32000.0, 1e-8);
    require_close("deflation gradient mode1 imag", c_host[1], 0.0, 1e-12);
    require_close("deflation gradient mode2 real", c_host[2], 0.0, 1e-12);
    require_close("deflation gradient mode2 imag", c_host[3], 0.0, 1e-12);

    adapter.apply_shift(query, shifted_query, 0.41);
    storage.calc_distance(shifted_query, beta, c);
    require_close("shifted query beta", beta, 401.0, 1e-10);
    const auto shifted_query_host = get_vector(vec_ops, shifted_query);
    const auto shifted_c_host = get_vector(vec_ops, c);
    double tangent_dot = 0.0;
    for(std::size_t mode = 1; mode <= 2; ++mode)
    {
        const std::size_t offset = 2*(mode - 1);
        const double real_part = shifted_query_host[offset];
        const double imag_part = shifted_query_host[offset + 1];
        const double tangent_real = -static_cast<double>(mode)*imag_part;
        const double tangent_imag = static_cast<double>(mode)*real_part;
        tangent_dot += shifted_c_host[offset]*tangent_real + shifted_c_host[offset + 1]*tangent_imag;
    }
    require_close("pulled deflation gradient is orbit-orthogonal", tangent_dot, 0.0, 1e-8);

    storage.clear();
    storage.set_known_solution(x);
    storage.calc_distance(query, beta, c);
    require_close("known solution beta", beta, 401.0, 1e-10);

    vec_ops.stop_use_vector(c);
    vec_ops.free_vector(c);
    vec_ops.stop_use_vector(query);
    vec_ops.free_vector(query);
    vec_ops.stop_use_vector(shifted_query);
    vec_ops.free_vector(shifted_query);
    vec_ops.stop_use_vector(stabilized);
    vec_ops.free_vector(stabilized);
    vec_ops.stop_use_vector(shifted);
    vec_ops.free_vector(shifted);
    vec_ops.stop_use_vector(x);
    vec_ops.free_vector(x);

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
