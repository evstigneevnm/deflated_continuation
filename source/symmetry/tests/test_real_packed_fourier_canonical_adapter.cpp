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

void test_negative_reflection_representatives_collapse_when_enabled()
{
    vec_ops_t vec_ops(8);
    adapter_t adapter(&vec_ops, 4);
    adapter.enable_negative_reflection_symmetry();
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {1.0, 0.3, -0.2, 0.5, 0.05, -0.1, 0.03, 0.02});
    adapter.apply_negative_reflection(v.x, v.y);
    adapter.apply_shift(v.y, v.w, 0.41);
    adapter.stabilize_canonical(v.x, v.z);
    adapter.stabilize_canonical(v.w, v.y);

    require_vector_close(vec_ops, "canonical negative-reflection shifted copy", v.z, v.y, 2e-12);

    free_bundle(vec_ops, v);
}

void test_closest_to_reference_uses_negative_reflection_action()
{
    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    adapter.enable_negative_reflection_symmetry();
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {1.0, 0.0, 0.25, 0.4});
    adapter.apply_negative_reflection(v.x, v.y);
    adapter.stabilize_closest_to_reference(v.x, v.y, v.z);

    require_vector_close(vec_ops, "closest-to-reference negative reflection", v.z, v.x, 2e-12);
    require_true(
        "closest-to-reference selected negative reflection",
        adapter.last_discrete_action() ==
            symmetry::fourier::real_packed_fourier_1d_discrete_action::negative_reflection);

    free_bundle(vec_ops, v);
}

void test_continuation_chart_uses_negative_reflection_action()
{
    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    adapter.enable_negative_reflection_symmetry();
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {1.0, 0.0, 0.25, 0.4});
    adapter.apply_negative_reflection(v.x, v.y);
    adapter.apply_shift(v.y, v.z, 0.43);
    set_vector(vec_ops, v.w, {0.0, 0.0, 0.0, 0.0});
    adapter.stabilize_continuation_chart(v.x, v.w, v.z, v.y);

    require_vector_close(vec_ops, "continuation-chart negative reflection", v.y, v.x, 2e-12);
    require_true(
        "continuation-chart selected negative reflection",
        adapter.last_discrete_action() ==
            symmetry::fourier::real_packed_fourier_1d_discrete_action::negative_reflection);

    free_bundle(vec_ops, v);
}

void test_relative_active_mode_threshold_skips_tiny_low_mode()
{
    vec_ops_t vec_ops(6);
    adapter_t adapter(&vec_ops, 3);
    vector_bundle v;
    init_bundle(vec_ops, v);

    adapter.set_relative_active_mode_tolerance(1e-4);
    set_vector(vec_ops, v.x, {1.0e-6, 0.0, 1.0, 0.25, 0.1, 0.05});
    adapter.stabilize_canonical(v.x, v.z);
    require_true("relative threshold selects mode 2", adapter.last_slice_data().mode == 2);
    const auto canonical = get_vector(vec_ops, v.z);
    require_true("mode 2 real positive after relative threshold", canonical[2] > 0.0);
    require_close("mode 2 imag after relative threshold", canonical[3], 0.0, 1e-12);

    free_bundle(vec_ops, v);
}

void test_continuation_hysteresis_switches_before_mode_vanishes()
{
    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    adapter.set_relative_active_mode_tolerance(1e-12);
    adapter.set_continuation_mode_switch_ratio(0.25);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {1.0, 0.0, 0.0, 0.0});
    adapter.stabilize(v.x, v.w);
    require_true("history starts on mode 1", adapter.last_slice_data().mode == 1);

    set_vector(vec_ops, v.x, {0.01, 0.0, 1.0, 0.25});
    set_vector(vec_ops, v.y, {0.0, 0.0, 0.0, 0.0});
    adapter.begin_continuation_chart(v.x, v.y);
    require_true("continuation switches to better conditioned mode 2", adapter.last_slice_data().mode == 2);
    require_true("continuation mode 2 has strong slice matrix", adapter.last_slice_data().slice_matrix > 1.0);

    free_bundle(vec_ops, v);
}

void test_continuation_hysteresis_keeps_usable_current_mode()
{
    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    adapter.set_relative_active_mode_tolerance(1e-12);
    adapter.set_continuation_mode_switch_ratio(0.25);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {1.0, 0.0, 0.0, 0.0});
    adapter.stabilize(v.x, v.w);
    require_true("history starts on mode 1 for keep test", adapter.last_slice_data().mode == 1);

    set_vector(vec_ops, v.x, {0.6, 0.0, 1.0, 0.0});
    set_vector(vec_ops, v.y, {0.0, 0.0, 0.0, 0.0});
    adapter.begin_continuation_chart(v.x, v.y);
    require_true("continuation keeps sufficiently conditioned mode 1", adapter.last_slice_data().mode == 1);

    free_bundle(vec_ops, v);
}

void test_continuation_residual_copy_uses_tangent_direction()
{
    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    adapter.set_relative_active_mode_tolerance(1e-12);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {0.0, 0.0, 1.0, 0.0});
    set_vector(vec_ops, v.z, {0.1, 0.0, 1.0, 0.0});

    set_vector(vec_ops, v.y, {1.0, 0.0, 0.0, 0.0});
    adapter.stabilize_continuation_chart(v.x, v.y, v.z, v.w);
    auto positive = get_vector(vec_ops, v.w);
    require_true("positive tangent selects positive residual copy", positive[0] > 0.0);
    require_close("positive tangent mode 2 imag", positive[3], 0.0, 1e-12);

    set_vector(vec_ops, v.y, {-1.0, 0.0, 0.0, 0.0});
    adapter.stabilize_continuation_chart(v.x, v.y, v.z, v.w);
    auto negative = get_vector(vec_ops, v.w);
    require_true("negative tangent selects negative residual copy", negative[0] < 0.0);
    require_close("negative tangent mode 2 imag", negative[3], 0.0, 1e-12);

    free_bundle(vec_ops, v);
}

void set_group_tangent_from_state(vec_ops_t& vec_ops, const vector_t& state, vector_t& tangent)
{
    const auto source = get_vector(vec_ops, state);
    std::vector<real> values(source.size(), real(0));
    for(std::size_t mode = 1; 2*(mode - 1) + 1 < source.size(); ++mode)
    {
        const std::size_t offset = 2*(mode - 1);
        const real re = source[offset];
        const real im = source[offset + 1];
        values[offset] = -static_cast<real>(mode)*im;
        values[offset + 1] = static_cast<real>(mode)*re;
    }
    vec_ops.set(values.data(), tangent, values.size());
}

void test_lsq_continuation_collapses_high_mode_shift_with_c3_residual()
{
    vec_ops_t vec_ops(18);
    adapter_t adapter(&vec_ops, 9);
    adapter.set_stabilizer_policy(symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::lsq_multimode);
    adapter.set_lsq_mode_range(1, 9);
    adapter.set_lsq_max_active_modes(6);
    adapter.set_lsq_grid_points(96);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(
        vec_ops,
        v.x,
        {
            0.0, 0.0,
            0.0, 0.0,
            2.0, 0.3,
            0.0, 0.0,
            0.0, 0.0,
            -0.4, 0.1,
            0.0, 0.0,
            0.0, 0.0,
            0.05, -0.02
        });
    adapter.apply_shift(v.x, v.y, 0.41);
    set_vector(vec_ops, v.z, std::vector<real>(18, real(0)));
    adapter.stabilize_continuation_chart(v.x, v.z, v.y, v.w);

    require_vector_close(vec_ops, "LSQ C3 shifted copy", v.w, v.x, 1e-10);
    require_true("LSQ C3 active", adapter.last_slice_data().active());
    require_true("LSQ C3 residual group", adapter.last_slice_data().residual_group_order() == 3);
    require_true("LSQ C3 uses multiple modes", adapter.last_slice_data().active_modes.size() >= 3);

    set_group_tangent_from_state(vec_ops, v.y, v.z);
    adapter.stabilizer_differential_from_last(v.z, v.w);
    require_close("LSQ C3 group tangent projected out", vec_ops.norm_l2(v.w), 0.0, 1e-10);

    free_bundle(vec_ops, v);
}

void test_lsq_continuation_primitive_modes_remove_residual_group()
{
    vec_ops_t vec_ops(8);
    adapter_t adapter(&vec_ops, 4);
    adapter.set_stabilizer_policy(symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::lsq_multimode);
    adapter.set_lsq_mode_range(1, 4);
    adapter.set_lsq_max_active_modes(4);
    adapter.set_lsq_grid_points(96);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(
        vec_ops,
        v.x,
        {
            0.0, 0.0,
            0.0, 0.0,
            1.0, 0.25,
            -0.7, 0.15
        });
    adapter.apply_shift(v.x, v.y, -0.27);
    set_vector(vec_ops, v.z, std::vector<real>(8, real(0)));
    adapter.stabilize_continuation_chart(v.x, v.z, v.y, v.w);

    require_vector_close(vec_ops, "LSQ primitive shifted copy", v.w, v.x, 1e-10);
    require_true("LSQ primitive residual group is trivial", adapter.last_slice_data().residual_group_order() == 1);
    require_true("LSQ primitive has modes 3 and 4", adapter.last_slice_data().active_modes.size() == 2);

    free_bundle(vec_ops, v);
}

} // namespace

int main()
{
    test_shifted_copies_have_one_canonical_representative();
    test_canonical_ignores_chart_history();
    test_residual_group_representatives_collapse();
    test_negative_reflection_representatives_collapse_when_enabled();
    test_closest_to_reference_uses_negative_reflection_action();
    test_continuation_chart_uses_negative_reflection_action();
    test_relative_active_mode_threshold_skips_tiny_low_mode();
    test_continuation_hysteresis_switches_before_mode_vanishes();
    test_continuation_hysteresis_keeps_usable_current_mode();
    test_continuation_residual_copy_uses_tangent_direction();
    test_lsq_continuation_collapses_high_mode_shift_with_c3_residual();
    test_lsq_continuation_primitive_modes_remove_residual_group();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
