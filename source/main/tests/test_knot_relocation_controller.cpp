#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <main/deflation_continuation/knot_relocation_controller.h>

namespace
{

struct settings
{
    bool enabled = true;
    unsigned int candidate_count = 4;
    double min_shift_abs = 0.1;
    double max_shift_abs = 0.2;
    bool prefer_positive_shift = true;
    bool require_all_intersections = true;
    bool save_registry = true;
    std::string registry_file = "knot_registry.json";
};

struct intersection_status
{
    unsigned int added = 0;
    unsigned int failed = 0;
    unsigned int missing_data = 0;
    unsigned int skipped_discontinuous = 0;
    unsigned int skipped_incomplete = 0;

    bool ok() const
    {
        return failed == 0 && missing_data == 0 && skipped_incomplete == 0;
    }
};

struct fake_registry
{
    void set(
        double requested_,
        double effective_,
        const std::string& reason_,
        const intersection_status& status_)
    {
        requested = requested_;
        effective = effective_;
        reason = reason_;
        status = status_;
        set_called = true;
    }
    bool save()
    {
        save_called = true;
        return true;
    }

    bool set_called = false;
    bool save_called = false;
    double requested = 0.0;
    double effective = 0.0;
    std::string reason;
    intersection_status status;
};

struct fake_log
{
    template<class... Args>
    void warning_f(const char*, Args&&...) {}
    template<class... Args>
    void info_f(const char*, Args&&...) {}
};

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

bool close_value(double left, double right)
{
    return std::abs(left - right) < 1.0e-14;
}

}

int main()
{
    try
    {
        fake_log log;
        using controller_type =
            main_classes::deflation_continuation_detail::knot_relocation_controller<
                double,
                intersection_status,
                settings,
                fake_log>;
        const std::vector<double> knots{4.0, 6.0, 8.0};
        controller_type controller(settings{}, knots, "project/", &log);
        require(controller.registry_file_name() == "project/knot_registry.json", "registry path");
        require(controller.bounds(6.0) == std::make_pair(4.0, 8.0), "neighbor bounds");
        require(close_value(controller.candidate(6.0, 0), 6.1), "positive first candidate");
        require(close_value(controller.candidate(6.0, 1), 5.9), "negative first candidate");
        require(close_value(controller.candidate(6.0, 2), 6.2), "positive outer candidate");

        intersection_status failed;
        failed.failed = 1;
        fake_registry registry;
        double effective = 0.0;
        intersection_status effective_status;
        const bool relocated = controller.relocate_restart_intersection(
            6.0,
            failed,
            registry,
            effective,
            effective_status,
            [](double candidate)
            {
                intersection_status status;
                if(std::abs(candidate - 5.9) < 1.0e-14)
                {
                    status.added = 1;
                }
                else
                {
                    status.failed = 1;
                }
                return status;
            });
        require(relocated, "restart relocation");
        require(close_value(effective, 5.9), "restart effective knot");
        require(registry.set_called && registry.save_called, "restart registry persistence");

        fake_registry active_registry;
        std::vector<double> effective_value;
        effective = 0.0;
        const bool active_relocated = controller.relocate_active_intersection(
            6.0,
            5.0,
            std::vector<double>{5.0},
            7.0,
            std::vector<double>{7.0},
            active_registry,
            effective,
            effective_value,
            [](double candidate, double, const std::vector<double>&, double,
               const std::vector<double>&, std::vector<double>& output)
            {
                output = {candidate};
                return true;
            });
        require(active_relocated, "active relocation");
        require(close_value(effective, 6.1), "active effective knot");
        require(effective_value == std::vector<double>({6.1}), "active interpolated value");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
