#include "explicit_lorenz_problem.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>
#include <scfd/static_vec/vec.h>
#ifdef TEST_VECTOR_BACKEND_CUDA
#include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda;
constexpr const char* backend_name = "cuda";
#elif defined(TEST_VECTOR_BACKEND_OMP)
#include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp;
constexpr const char* backend_name = "omp";
#else
#include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu;
constexpr const char* backend_name = "serial_cpu";
#endif
#include <common/scfd_vector_operations.h>
#include <contrib/json/nlohmann/json.hpp>
#include <time_stepper/integration/time_integrator.h>
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/integration/time_step_adaptation_scipy.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
template<class VectorOperations>
class external_manager
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params
    {
        std::string filename;
    };

    external_manager(VectorOperations& operations, const params& p, const nlohmann::json& metadata) :
        operations_(operations), params_(p), metadata_(metadata)
    {
        trajectory_.reserve(1024);
    }

    void set_time_interval(const scalar_type& start, scalar_type& end)
    {
        start_ = start;
        end_ = end;
    }

    bool apply(const vector_type& in, vector_type& out)
    {
        if (trajectory_.empty())
        {
            record(start_, in);
        }
        record(end_, out);
        return false;
    }

    integration_status get_status() const
    {
        return integration_status::running;
    }

    void complete(std::size_t accepted_steps)
    {
        if (trajectory_.size() != accepted_steps + 1)
        {
            throw std::runtime_error("Trajectory row count differs from accepted steps");
        }
        std::ofstream output;
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.imbue(std::locale::classic());
        output.open(params_.filename);
        output << std::setprecision(std::numeric_limits<scalar_type>::max_digits10);
        output << "# " << metadata_.dump() << '\n';
        output << "# time x y z\n";
        for (const auto& sample : trajectory_)
        {
            output << sample.time << ' ' << sample.state[0] << ' ' << sample.state[1] << ' ' << sample.state[2] << '\n';
        }
        const nlohmann::json footer = {{"status", "completed"}, {"rows", trajectory_.size()}};
        output << "# " << footer.dump() << '\n';
        output.close();
    }

private:
    struct sample
    {
        scalar_type time;
        scfd::static_vec::vec<scalar_type, 3> state;
    };

    VectorOperations& operations_;
    params params_;
    nlohmann::json metadata_;
    std::vector<sample> trajectory_;
    scalar_type start_ = 0, end_ = 0;

    void record(scalar_type time, const vector_type& state)
    {
        const auto view = operations_.view(state);
        trajectory_.push_back({time, {view(0), view(1), view(2)}});
    }
};
}
}
}

template<class Adaptation>
void run_simulation(
    double end_time, const std::string& filename, const std::string& method, const std::string& controller)
{
    using operations_type = scfd_vector_operations<backend_type, double>;
    using problem_type = nmfd::time_steppers::tests::explicit_lorenz_problem<operations_type>;
    using adaptation_type = Adaptation;
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, problem_type, adaptation_type>;
    using external_type = nmfd::time_steppers::tests::external_manager<operations_type>;
    using integrator_type =
        nmfd::time_steppers::integration::time_integrator<operations_type, step_type, external_type>;

    operations_type operations(3);
    problem_type problem;
    nmfd::detail::vector_wrap<operations_type> initial(operations), result(operations);
    initial.start_use();
    result.start_use();
    const scfd::static_vec::vec<double, 3> initial_values{2.2, 30.5, 2.5};
    operations.set(initial_values.d, *initial);

    typename adaptation_type::params adaptation_params;
    adaptation_params.relative_tolerance = 1e-10;
    adaptation_params.absolute_tolerance = 1e-12;
    adaptation_type adaptation(operations, adaptation_params);
    step_type step(operations, problem, adaptation, {method});
    const nlohmann::json metadata = {{"format", "nmfd.lorenz_trajectory.v1"}, {"problem", "lorenz"}, {"method", method},
        {"backend", backend_name}, {"controller", controller}, {"start_time", 0}, {"end_time", end_time},
        {"initial_state", {initial_values[0], initial_values[1], initial_values[2]}},
        {"parameters", {{"sigma", problem.sigma}, {"rho", problem.rho}, {"beta", problem.beta},
                           {"epsilon", problem.epsilon}, {"delta", problem.delta}}},
        {"adaptation",
            {{"relative_tolerance", adaptation_params.relative_tolerance},
                {"absolute_tolerance", adaptation_params.absolute_tolerance},
                {"initial_step", adaptation_params.initial_step}, {"minimum_step", adaptation_params.minimum_step},
                {"maximum_step", adaptation_params.maximum_step}}}};
    external_type external(operations, {filename}, metadata);
    integrator_type integrator(operations, step, {}, &external);
    integrator.set_time_interval(0, end_time);
    integrator.apply(*initial, *result);
    if (integrator.get_status() != nmfd::time_steppers::integration_status::completed ||
        integrator.get_final_time() != end_time)
    {
        throw std::runtime_error("Lorenz simulation failed; trajectory is incomplete");
    }
    external.complete(integrator.get_steps());
    std::cout << "lorenz: " << method << ", " << controller << ", " << backend_name << ", " << integrator.get_steps()
              << " accepted steps, final time " << end_time << ", trajectory " << filename << '\n';
}

int main(int argc, char** argv)
try
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0] << " end_time trajectory.dat [RK45|RK23|DOP853] [matlab|scipy]\n";
        return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
    }
    const std::string method = argc >= 4 ? argv[3] : "RK45";
    const std::string controller = argc == 5 ? argv[4] : method == "DOP853" ? "scipy" : "matlab";
    if ((method != "RK45" && method != "RK23" && method != "DOP853") ||
        (controller != "matlab" && controller != "scipy"))
    {
        throw std::invalid_argument("Expected RK45, RK23 or DOP853 and matlab or scipy adaptation");
    }
    if (method == "DOP853" && controller != "scipy")
    {
        throw std::invalid_argument("Adaptive DOP853 requires scipy adaptation for its combined estimator");
    }
    std::size_t parsed = 0;
    const double end_time = std::stod(argv[1], &parsed);
    if (parsed != std::string(argv[1]).size() || !std::isfinite(end_time) || end_time <= 0)
    {
        throw std::invalid_argument("Integration end time must be positive and finite");
    }
    backend_type::init_device();
    using operations_type = scfd_vector_operations<backend_type, double>;
    if (controller == "scipy")
    {
        run_simulation<nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>>(
            end_time, argv[2], method, controller);
    }
    else
    {
        run_simulation<nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>>(
            end_time, argv[2], method, controller);
    }
}
catch (const std::exception& e)
{
    std::cerr << "Lorenz simulation: " << e.what() << '\n';
    return 1;
}
