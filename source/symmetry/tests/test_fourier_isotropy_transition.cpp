#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <common/scfd_serial_cpu_vector_operations.h>
#include <continuation/chart_helpers.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>

namespace
{

using real = double;
using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
using adapter_t = symmetry::fourier::real_packed_fourier_slice_1d_adapter<vec_ops_t>;
using vector_t = typename vec_ops_t::vector_type;
using policy_t = symmetry::continuation::isotropy_transition_policy<real>;

int checks = 0;
int failures = 0;

void require_true(const std::string& label, const bool value)
{
    ++checks;
    if(!value)
    {
        ++failures;
        std::cerr << "FAIL " << label << std::endl;
    }
}

void require_close(
    const std::string& label,
    const real value,
    const real expected,
    const real tolerance)
{
    ++checks;
    if(std::abs(value - expected) > tolerance)
    {
        ++failures;
        std::cerr << "FAIL " << label
                  << " value=" << value
                  << " expected=" << expected
                  << " tolerance=" << tolerance << std::endl;
    }
}

struct vectors
{
    vector_t previous;
    vector_t candidate;
    vector_t shifted_previous;
    vector_t shifted_candidate;
};

void init_vectors(vec_ops_t& vec_ops, vectors& value)
{
    vec_ops.init_vector(value.previous);
    vec_ops.init_vector(value.candidate);
    vec_ops.init_vector(value.shifted_previous);
    vec_ops.init_vector(value.shifted_candidate);
    vec_ops.start_use_vector(value.previous);
    vec_ops.start_use_vector(value.candidate);
    vec_ops.start_use_vector(value.shifted_previous);
    vec_ops.start_use_vector(value.shifted_candidate);
}

void free_vectors(vec_ops_t& vec_ops, vectors& value)
{
    vec_ops.stop_use_vector(value.shifted_candidate);
    vec_ops.free_vector(value.shifted_candidate);
    vec_ops.stop_use_vector(value.shifted_previous);
    vec_ops.free_vector(value.shifted_previous);
    vec_ops.stop_use_vector(value.candidate);
    vec_ops.free_vector(value.candidate);
    vec_ops.stop_use_vector(value.previous);
    vec_ops.free_vector(value.previous);
}

void set_vector(vec_ops_t& vec_ops, vector_t& value, const std::vector<real>& host)
{
    vec_ops.set(host.data(), value, host.size());
}

policy_t test_policy()
{
    policy_t policy;
    policy.enabled = true;
    policy.relative_mode_tolerance = 1.0e-8;
    policy.maximum_order = 16;
    policy.maximum_refinements = 4;
    policy.refinement_step_factor = 0.25;
    return policy;
}

void test_generic_to_c2_transition_is_detected_and_shift_invariant()
{
    vec_ops_t vec_ops(16);
    adapter_t adapter(&vec_ops, 8);
    vectors value;
    init_vectors(vec_ops, value);

    set_vector(
        vec_ops,
        value.previous,
        {1.0e-4, 2.0e-5, 1.0, -0.2, 0.0, 0.0, 0.3, 0.1,
         0.0, 0.0, 0.1, -0.05, 0.0, 0.0, 0.03, 0.02});
    set_vector(
        vec_ops,
        value.candidate,
        {1.0e-12, -2.0e-12, 1.0, -0.2, 0.0, 0.0, 0.3, 0.1,
         0.0, 0.0, 0.1, -0.05, 0.0, 0.0, 0.03, 0.02});

    const auto result = adapter.detect_continuation_isotropy_transition(
        value.previous,
        value.candidate,
        test_policy());
    require_true("C1 to C2 supported", result.supported);
    require_true("C1 to C2 detected", result.detected);
    require_true("C1 previous order", result.previous_order == 1);
    require_true("C2 candidate order", result.candidate_order == 2);
    require_true(
        "C2 orbit descriptor is populated",
        result.candidate_orbit_type.group_dimension == 1 &&
            result.candidate_orbit_type.finite_isotropy_order() == 2);
    require_true(
        "C1 transverse component is resolved",
        result.previous_transverse_ratio > test_policy().relative_mode_tolerance);
    require_true(
        "C2 transverse component is below threshold",
        result.candidate_transverse_ratio <= test_policy().relative_mode_tolerance);

    adapter.apply_shift(value.previous, value.shifted_previous, 0.731);
    adapter.apply_shift(value.candidate, value.shifted_candidate, 0.731);
    const auto shifted_result = adapter.detect_continuation_isotropy_transition(
        value.shifted_previous,
        value.shifted_candidate,
        test_policy());
    require_true("shifted C1 to C2 detected", shifted_result.detected);
    require_true(
        "shifted orders unchanged",
        shifted_result.previous_order == result.previous_order &&
            shifted_result.candidate_order == result.candidate_order);
    require_close(
        "shifted previous transverse ratio",
        shifted_result.previous_transverse_ratio,
        result.previous_transverse_ratio,
        1.0e-14);

    free_vectors(vec_ops, value);
}

void test_higher_and_nested_isotropy_transitions()
{
    vec_ops_t vec_ops(16);
    adapter_t adapter(&vec_ops, 8);
    vectors value;
    init_vectors(vec_ops, value);

    set_vector(
        vec_ops,
        value.previous,
        {0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0, 1.0, 0.2,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.05});
    set_vector(
        vec_ops,
        value.candidate,
        {0.0, 0.0, 1.0e-12, 0.0, 0.0, 0.0, 1.0, 0.2,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.05});
    const auto c2_to_c4 = adapter.detect_continuation_isotropy_transition(
        value.previous,
        value.candidate,
        test_policy());
    require_true("C2 to C4 detected", c2_to_c4.detected);
    require_true("C2 previous order", c2_to_c4.previous_order == 2);
    require_true("C4 candidate order", c2_to_c4.candidate_order == 4);

    set_vector(
        vec_ops,
        value.previous,
        {2.0e-4, 0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0,
         0.0, 0.0, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0});
    set_vector(
        vec_ops,
        value.candidate,
        {1.0e-12, 0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0,
         0.0, 0.0, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0});
    const auto c1_to_c3 = adapter.detect_continuation_isotropy_transition(
        value.previous,
        value.candidate,
        test_policy());
    require_true("C1 to C3 detected", c1_to_c3.detected);
    require_true("C3 candidate order", c1_to_c3.candidate_order == 3);

    free_vectors(vec_ops, value);
}

void test_non_transitions_are_ignored()
{
    vec_ops_t vec_ops(16);
    adapter_t adapter(&vec_ops, 8);
    vectors value;
    init_vectors(vec_ops, value);

    const std::vector<real> c2_state{
        0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.3, -0.1,
        0.0, 0.0, 0.1, 0.02, 0.0, 0.0, 0.05, -0.01};
    set_vector(vec_ops, value.previous, c2_state);
    set_vector(vec_ops, value.candidate, c2_state);
    auto result = adapter.detect_continuation_isotropy_transition(
        value.previous,
        value.candidate,
        test_policy());
    require_true("same C2 stratum is not an event", !result.detected);

    set_vector(
        vec_ops,
        value.previous,
        {1.0e-4, 0.0, 1.0, 0.2, 0.0, 0.0, 0.3, -0.1,
         0.0, 0.0, 0.1, 0.02, 0.0, 0.0, 0.05, -0.01});
    set_vector(
        vec_ops,
        value.candidate,
        {2.0e-5, 0.0, 1.0, 0.2, 0.0, 0.0, 0.3, -0.1,
         0.0, 0.0, 0.1, 0.02, 0.0, 0.0, 0.05, -0.01});
    result = adapter.detect_continuation_isotropy_transition(
        value.previous,
        value.candidate,
        test_policy());
    require_true("resolved transverse mode is not an event", !result.detected);

    set_vector(vec_ops, value.candidate, std::vector<real>(16, 0.0));
    result = adapter.detect_continuation_isotropy_transition(
        value.previous,
        value.candidate,
        test_policy());
    require_true("zero solution is not assigned an artificial Cn order", !result.detected);
    require_true("zero solution order remains trivial", result.candidate_order == 1);

    struct plain_operator
    {
    } plain;
    require_true(
        "Fourier adapter advertises isotropy monitoring",
        continuation::chart::has_isotropy_transition<adapter_t, vector_t, real>::value);
    require_true(
        "ordinary operator does not advertise isotropy monitoring",
        !continuation::chart::has_isotropy_transition<plain_operator, vector_t, real>::value);
    const auto unsupported = continuation::chart::detect_isotropy_transition(
        &plain,
        value.previous,
        value.candidate,
        test_policy());
    require_true("ordinary operator reports unsupported", !unsupported.supported);
    require_true("ordinary operator has no event", !unsupported.detected);

    free_vectors(vec_ops, value);
}

} // namespace

int main()
{
    test_generic_to_c2_transition_is_detected_and_shift_invariant();
    test_higher_and_nested_isotropy_transitions();
    test_non_transitions_are_ignored();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
