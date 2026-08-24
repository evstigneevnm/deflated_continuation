#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <models/KS_2D/KS2D_stability_cli.h>

namespace
{

using options_type = ks2d_stability_model::command_line_options;

options_type parse(std::vector<std::string> arguments)
{
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for(auto& argument: arguments)
        argv.push_back(argument.data());
    return ks2d_stability_model::parse_command_line(
        static_cast<int>(argv.size()),
        argv.data(),
        "default.json");
}

template<class Function>
bool throws_invalid_argument(Function&& function)
{
    try
    {
        function();
    }
    catch(const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    int checks = 0;
    int failures = 0;
    const auto check = [&checks, &failures](bool condition, const char* name)
    {
        ++checks;
        if(!condition)
        {
            ++failures;
            std::cerr << "FAIL: " << name << '\n';
        }
    };

    const auto replay = parse({
        "test",
        "project.json",
        "--curve-transition",
        "11",
        "50",
        "100",
        "--confirm",
        "--quiet"});
    check(replay.config_file == "project.json", "config file");
    check(replay.curve_transition, "curve replay enabled");
    check(replay.curve_transition_curve == 11, "curve number");
    check(
        replay.curve_transition_lower_source == std::uint64_t(50),
        "lower source point");
    check(
        replay.curve_transition_upper_source == std::uint64_t(100),
        "upper source point");
    check(replay.confirm, "confirmation enabled");
    check(replay.quiet, "quiet enabled");

    check(
        throws_invalid_argument([]
        {
            parse({
                "test",
                "project.json",
                "--curve-transition",
                "11",
                "50",
                "100"});
        }),
        "curve replay requires confirmation");
    check(
        throws_invalid_argument([]
        {
            parse({
                "test",
                "project.json",
                "--curve-transition",
                "11",
                "100",
                "50",
                "--confirm"});
        }),
        "curve replay requires ordered sources");
    check(
        throws_invalid_argument([]
        {
            parse({
                "test",
                "project.json",
                "--curve-transition",
                "-1",
                "50",
                "100",
                "--confirm"});
        }),
        "curve replay rejects negative curve");
    check(
        throws_invalid_argument([]
        {
            parse({
                "test",
                "project.json",
                "--curve-transition",
                "11",
                "50",
                "100",
                "--confirm",
                "--state-file",
                "state.dat",
                "--parameter",
                "17.9"});
        }),
        "curve replay rejects another replay mode");

    std::cout
        << "Checks: " << checks
        << ", failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
