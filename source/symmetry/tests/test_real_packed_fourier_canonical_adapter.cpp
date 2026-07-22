#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <common/scfd_serial_cpu_vector_operations.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_policy_json.h>

namespace
{

int checks = 0;
int failures = 0;

using real = double;
using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
using adapter_t = symmetry::fourier::real_packed_fourier_slice_1d_adapter<vec_ops_t>;
using policy_t = typename adapter_t::policy_type;
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
    set_vector(vec_ops, v.y, {0.0, 0.0, 0.0, 0.0});
    adapter.prepare_continuation_seed(v.x, v.w);
    require_true("continuation state starts on mode 1", adapter.continuation_chart_state().data.mode == 1);

    set_vector(vec_ops, v.x, {0.01, 0.0, 1.0, 0.25});
    adapter.accept_continuation_step(v.x, v.y);
    require_true(
        "accepted-point transition switches to better conditioned mode 2",
        adapter.continuation_chart_state().data.mode == 2);
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
    set_vector(vec_ops, v.y, {0.0, 0.0, 0.0, 0.0});
    adapter.prepare_continuation_seed(v.x, v.w);
    require_true(
        "continuation state starts on mode 1 for keep test",
        adapter.continuation_chart_state().data.mode == 1);

    set_vector(vec_ops, v.x, {0.6, 0.0, 1.0, 0.0});
    adapter.accept_continuation_step(v.x, v.y);
    require_true(
        "accepted-point transition keeps sufficiently conditioned mode 1",
        adapter.continuation_chart_state().data.mode == 1);

    free_bundle(vec_ops, v);
}

void test_continuation_chart_restore_recovers_anchor_mode()
{
    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    adapter.set_relative_active_mode_tolerance(1e-12);
    adapter.set_continuation_mode_switch_ratio(0.25);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {1.0, 0.0, 0.0, 0.0});
    set_vector(vec_ops, v.y, {0.0, 0.0, 0.0, 0.0});
    adapter.prepare_continuation_seed(v.x, v.w);
    adapter.begin_continuation_chart(v.x, v.y);
    require_true("continuation restore anchor starts on mode 1", adapter.last_slice_data().mode == 1);

    set_vector(vec_ops, v.z, {0.01, 0.0, 1.0, 0.25});
    adapter.stabilize_continuation_chart(v.x, v.y, v.z, v.w);
    require_true("continuation trial keeps frozen anchor mode", adapter.last_slice_data().mode == 1);

    adapter.restore_continuation_chart();
    require_true("continuation restore recovers anchor mode", adapter.last_slice_data().mode == 1);
    set_vector(vec_ops, v.z, {0.0, 1.0, 0.0, 0.0});
    adapter.stabilizer_differential_from_last(v.z, v.w);
    require_close(
        "continuation restore recovers anchor differential state",
        vec_ops.norm_l2(v.w),
        0.0,
        1e-12);

    free_bundle(vec_ops, v);
}

void test_prepared_seed_and_predictor_share_one_chart()
{
    vec_ops_t vec_ops(6);
    adapter_t adapter(&vec_ops, 3);
    adapter.set_relative_active_mode_tolerance(1e-12);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {0.12, -0.04, 2.0, 0.7, -0.2, 0.1});
    adapter.prepare_continuation_seed(v.x, v.z);
    require_true("prepared seed initializes chart state", adapter.continuation_chart_state().initialized);
    require_true("prepared seed selects well-conditioned mode 2", adapter.continuation_chart_state().data.mode == 2);

    set_vector(vec_ops, v.y, {0.03, -0.01, 0.02, 0.01, -0.015, 0.005});
    adapter.begin_continuation_chart(v.z, v.y);
    vec_ops.assign_mul(real(1), v.z, real(0.02), v.y, v.w);
    adapter.stabilize_continuation_chart(v.z, v.y, v.w, v.x);

    vec_ops.assign_mul(real(1), v.x, real(-1), v.w, v.w);
    require_true(
        "prepared predictor chart displacement remains local",
        vec_ops.norm_l2(v.w) < 1e-3);
    require_true("predictor keeps prepared mode", adapter.last_slice_data().mode == 2);

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

void test_continuation_residual_copy_obeys_hard_locality()
{
    vec_ops_t vec_ops(4);
    adapter_t adapter(&vec_ops, 2);
    adapter.set_relative_active_mode_tolerance(1e-2);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(vec_ops, v.x, {0.002, 0.0, 1.0, 0.0});
    set_vector(vec_ops, v.z, {0.012, 0.0, 1.0, 0.0});
    set_vector(vec_ops, v.y, {-1.0, 0.0, 0.0, 0.0});

    adapter.stabilize_continuation_chart(v.x, v.y, v.z, v.w);
    const auto stabilized = get_vector(vec_ops, v.w);
    require_true("hard-local continuation keeps nearby passive odd mode", stabilized[0] > 0.0);
    require_close("hard-local continuation keeps nearby passive odd value", stabilized[0], 0.012, 1e-12);
    require_close("hard-local continuation mode 2 imag", stabilized[3], 0.0, 1e-12);
    require_true("hard-local continuation uses mode 2", adapter.last_slice_data().mode == 2);
    require_true(
        "hard-local continuation retains C2 residual group",
        adapter.last_slice_data().residual_group_order() == 2);

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

void test_lsq_prefers_reliable_coprime_mode_when_truncated()
{
    vec_ops_t vec_ops(8);
    adapter_t adapter(&vec_ops, 4);
    adapter.set_stabilizer_policy(symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::lsq_multimode);
    adapter.set_lsq_mode_range(1, 4);
    adapter.set_lsq_max_active_modes(2);
    adapter.set_lsq_grid_points(96);
    vector_bundle v;
    init_bundle(vec_ops, v);

    set_vector(
        vec_ops,
        v.x,
        {
            0.0, 0.0,
            2.0, 0.0,
            0.5, 0.0,
            1.5, 0.0
        });
    adapter.apply_shift(v.x, v.y, 0.17);
    set_vector(vec_ops, v.z, std::vector<real>(8, real(0)));
    adapter.stabilize_continuation_chart(v.x, v.z, v.y, v.w);

    require_vector_close(vec_ops, "LSQ reliable coprime shifted copy", v.w, v.x, 1e-10);
    require_true(
        "LSQ reliable coprime mode removes residual group",
        adapter.last_slice_data().residual_group_order() == 1);
    require_true(
        "LSQ reliable coprime mode respects active-mode limit",
        adapter.last_slice_data().active_modes.size() == 2);

    free_bundle(vec_ops, v);
}

void test_policy_configures_independent_adapters()
{
    vec_ops_t vec_ops(8);
    adapter_t storage_adapter(&vec_ops, 4);
    adapter_t continuation_adapter(&vec_ops, 4);
    policy_t storage_policy;
    storage_policy.relative_active_mode_tolerance = 1e-5;
    policy_t continuation_policy = storage_policy;
    continuation_policy.stabilizer =
        symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::lsq_multimode;
    continuation_policy.continuation_mode_switch_ratio = 0.3;
    continuation_policy.tangent_continuity_weight = 0.4;
    continuation_policy.tangent_backward_penalty = 5.0;
    continuation_policy.lsq.mode_min = 2;
    continuation_policy.lsq.mode_max = 4;
    continuation_policy.lsq.max_active_modes = 3;
    continuation_policy.lsq.grid_points = 96;
    continuation_policy.lsq.newton_iterations = 10;
    continuation_policy.lsq.prefer_trivial_residual_group = false;
    continuation_policy.lsq.minimum_coprime_relative_score = 0.1;
    continuation_policy.local_representative_relative_tolerance = 0.02;

    storage_adapter.configure(storage_policy);
    continuation_adapter.configure(continuation_policy);
    require_true("storage adapter policy equals source", storage_adapter.configuration() == storage_policy);
    require_true(
        "storage and continuation policies are intentionally independent",
        continuation_adapter.configuration() != storage_adapter.configuration());

    vector_bundle v;
    init_bundle(vec_ops, v);
    set_vector(vec_ops, v.x, {0.0, 0.0, 1.0, 0.25, 0.4, -0.1, -0.2, 0.05});
    set_vector(vec_ops, v.z, std::vector<real>(8, real(0)));
    storage_adapter.stabilize_canonical(v.x, v.y);
    require_true("storage adapter has independent chart history", storage_adapter.last_slice_data().active());
    require_true(
        "continuation adapter history is unchanged",
        !continuation_adapter.last_slice_data().active());
    require_true("mutable history does not change storage policy", storage_adapter.configuration() == storage_policy);
    require_true(
        "mutable history does not change continuation policy",
        continuation_adapter.configuration() == continuation_policy);

    continuation_adapter.begin_continuation_chart(v.x, v.z);
    bool rejected_reconfiguration = false;
    try
    {
        continuation_adapter.configure(continuation_policy);
    }
    catch(const std::logic_error&)
    {
        rejected_reconfiguration = true;
    }
    require_true("policy reconfiguration after chart start is rejected", rejected_reconfiguration);
    free_bundle(vec_ops, v);
}

void test_policy_json_parsing_and_validation()
{
    const auto root = nlohmann::json::parse(R"json(
    {
        "symmetry_stabilizer": {
            "type": "lsq_multimode",
            "relative_active_mode_tolerance": 1.0e-6,
            "continuation_mode_switch_ratio": 0.35,
            "tangent_continuity_weight": 0.45,
            "tangent_backward_penalty": 6.0,
            "mode_min": 2,
            "mode_max": 9,
            "max_active_modes": 5,
            "grid_points": 128,
            "newton_iterations": 11,
            "prefer_trivial_residual_group": false,
            "minimum_coprime_relative_score": 0.15,
            "local_representative_relative_tolerance": 0.025
        },
        "continuation_symmetry_stabilizer": {
            "type": "single_mode",
            "continuation_mode_switch_ratio": 0.2
        }
    })json");

    const auto policy =
        symmetry::fourier::read_real_packed_fourier_slice_1d_policy<real>(root);
    require_true(
        "JSON policy selects LSQ",
        policy.stabilizer == symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::lsq_multimode);
    require_close("JSON relative active tolerance", policy.relative_active_mode_tolerance, 1e-6, 0.0);
    require_close("JSON mode switch ratio", policy.continuation_mode_switch_ratio, 0.35, 0.0);
    require_close("JSON tangent continuity weight", policy.tangent_continuity_weight, 0.45, 0.0);
    require_close("JSON tangent backward penalty", policy.tangent_backward_penalty, 6.0, 0.0);
    require_true("JSON LSQ mode range", policy.lsq.mode_min == 2 && policy.lsq.mode_max == 9);
    require_true("JSON LSQ active mode count", policy.lsq.max_active_modes == 5);
    require_true("JSON LSQ grid points", policy.lsq.grid_points == 128);
    require_true("JSON LSQ Newton iterations", policy.lsq.newton_iterations == 11);
    require_true("JSON residual-group preference", !policy.lsq.prefer_trivial_residual_group);
    require_close("JSON coprime score", policy.lsq.minimum_coprime_relative_score, 0.15, 0.0);
    require_close(
        "JSON local representative tolerance",
        policy.local_representative_relative_tolerance,
        0.025,
        0.0);
    const auto continuation_policy =
        symmetry::fourier::read_real_packed_fourier_slice_1d_policy<real>(
            root,
            "continuation_symmetry_stabilizer",
            policy_t());
    require_true(
        "named JSON policy selects single mode",
        continuation_policy.stabilizer ==
            symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::single_mode);
    require_close(
        "named JSON policy mode switch ratio",
        continuation_policy.continuation_mode_switch_ratio,
        0.2,
        0.0);
    policy_t invalid = policy;
    invalid.lsq.grid_points = 7;
    bool rejected_invalid_policy = false;
    try
    {
        invalid.validate();
    }
    catch(const std::invalid_argument&)
    {
        rejected_invalid_policy = true;
    }
    require_true("invalid policy is rejected", rejected_invalid_policy);
}

} // namespace

int main()
{
    test_shifted_copies_have_one_canonical_representative();
    test_canonical_ignores_chart_history();
    test_residual_group_representatives_collapse();
    test_relative_active_mode_threshold_skips_tiny_low_mode();
    test_continuation_hysteresis_switches_before_mode_vanishes();
    test_continuation_hysteresis_keeps_usable_current_mode();
    test_continuation_chart_restore_recovers_anchor_mode();
    test_prepared_seed_and_predictor_share_one_chart();
    test_continuation_residual_copy_uses_tangent_direction();
    test_continuation_residual_copy_obeys_hard_locality();
    test_lsq_continuation_collapses_high_mode_shift_with_c3_residual();
    test_lsq_continuation_primitive_modes_remove_residual_group();
    test_lsq_prefers_reliable_coprime_mode_when_truncated();
    test_policy_configures_independent_adapters();
    test_policy_json_parsing_and_validation();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
