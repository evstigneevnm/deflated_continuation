#include <cstdlib>
#include <iostream>
#include <string>

#include <continuation/continuation_endpoint_state.h>
#include <continuation/pending_branch_event.h>

namespace
{

int checks = 0;
int failures = 0;

void require_true(const bool value, const std::string& label)
{
    ++checks;
    if(!value)
    {
        ++failures;
        std::cerr << "FAIL " << label << std::endl;
    }
}

void test_endpoint_precedence()
{
    using reason_t = container::curve_endpoint_reason;
    continuation::continuation_endpoint_state state;
    state.observe(reason_t::known_branch);
    state.observe(reason_t::boundary_max);
    require_true(
        state.pending_reason() == reason_t::known_branch,
        "known branch is not overwritten by a lower-priority boundary");
    state.observe(reason_t::hard_failure);
    require_true(
        state.pending_reason() == reason_t::hard_failure,
        "hard failure has terminal precedence");
    require_true(state.incomplete(), "hard failure marks curve incomplete");

    state.reset_step();
    require_true(state.pending_reason() == reason_t::none, "step reset clears reason");
    require_true(state.incomplete(), "step reset preserves curve completeness state");
    state.reset_curve();
    require_true(!state.incomplete(), "curve reset clears completeness state");

    continuation::continuation_endpoint_state analytical_state;
    analytical_state.observe(reason_t::analytical_branch);
    analytical_state.observe(reason_t::boundary_max);
    require_true(
        analytical_state.pending_reason() == reason_t::analytical_branch,
        "analytical branch is not overwritten by a boundary");
}

void test_pending_branch_event()
{
    continuation::pending_branch_event<double> event;
    container::curve_provenance analytical_target;
    analytical_target.origin = container::curve_origin::analytical;
    analytical_target.analytical_branch_id = 7;
    event.update(6.25, 4, 19, analytical_target);
    event.increment_refinements();
    event.increment_refinements();
    require_true(event.active(), "branch event is active");
    require_true(event.matches(4, 19), "branch event identity matches");
    require_true(event.refinements() == 2, "branch refinement count");
    require_true(event.lambda() == 6.25, "branch event lambda");
    require_true(
        event.target_provenance().is_analytical() &&
            event.target_provenance().analytical_branch_id == 7,
        "pending event preserves analytical branch identity");

    event.update(6.2, 4, 19, analytical_target);
    require_true(event.refinements() == 2, "same event keeps refinement count");
    require_true(
        event.prediction_is_stable(1.0e-2),
        "analytical event accepts a stable repeated parameter prediction");
    require_true(
        !event.prediction_is_stable(1.0e-4),
        "analytical event rejects an unstable parameter prediction");
    event.update(7.0, 5, 3);
    require_true(event.refinements() == 0, "new event resets refinement count");
    require_true(
        !event.prediction_is_stable(1.0),
        "new branch event has no previous prediction");
    require_true(!event.note_miss(2), "first event miss is tolerated");
    require_true(!event.note_miss(2), "second event miss is tolerated");
    require_true(event.note_miss(2), "third event miss expires event");
    require_true(!event.active(), "expired branch event is cleared");
}

} // namespace

int main()
{
    test_endpoint_precedence();
    test_pending_branch_event();
    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
