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
}

void test_pending_branch_event()
{
    continuation::pending_branch_event<double> event;
    event.update(6.25, 4, 19);
    event.increment_refinements();
    event.increment_refinements();
    require_true(event.active(), "branch event is active");
    require_true(event.matches(4, 19), "branch event identity matches");
    require_true(event.refinements() == 2, "branch refinement count");
    require_true(event.lambda() == 6.25, "branch event lambda");

    event.update(6.2, 4, 19);
    require_true(event.refinements() == 2, "same event keeps refinement count");
    event.update(7.0, 5, 3);
    require_true(event.refinements() == 0, "new event resets refinement count");
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
