#include "explicit_lorenz_problem.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
#ifdef TEST_VECTOR_BACKEND_CUDA
#include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda;
#elif defined(TEST_VECTOR_BACKEND_OMP)
#include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp;
#else
#include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu;
#endif
#include <common/scfd_vector_operations.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/integration/time_integrator.h>

int main()
try
{
    backend_type::init_device();
    using operations_type = scfd_vector_operations<backend_type, double>;
    operations_type operations(3);
    nmfd::detail::vector_wrap<operations_type> initial_wrap(operations), result_wrap(operations);
    initial_wrap.start_use(); result_wrap.start_use();
    auto& initial = *initial_wrap;
    auto& result = *result_wrap;
    const scfd::static_vec::vec<double,3> initial_values{2.2,30.5,2.5};
    // SciPy DOP853, rtol=3e-14, atol=1e-14; independently cross-checked with Radau.
    const scfd::static_vec::vec<double,3> reference{21.527600219626638,31.740750381438271,45.939726836031042};
    operations.set(initial_values.d, initial);
    nmfd::time_steppers::tests::explicit_lorenz_problem<operations_type> problem;
    using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>;
    adaptation_type::params p;
    p.relative_tolerance = 1e-10;
    p.absolute_tolerance = 1e-12;
    p.initial_step = .1;
    adaptation_type adaptation(operations, p);
    using step_type = nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, decltype(problem), adaptation_type>;
    step_type step(operations, problem, adaptation);
    nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator(operations, step);
    integrator.set_time_interval(0, .1);
    for (int run = 0; run < 2; ++run)
    {
        integrator.apply(initial, result);
        scfd::static_vec::vec<double,3> values;
        operations.get(result, values.d);
        if (integrator.get_status() != nmfd::time_steppers::integration_status::completed ||
            integrator.get_final_time() != .1)
            throw std::runtime_error("lorenz: integration failed");
        for (int j = 0; j < 3; ++j)
            if (std::abs(values[j]-reference[j]) > 2e-8)
                throw std::runtime_error("lorenz: independent reference mismatch");
    }
    using constant_type = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
    using fixed_step_type = nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, decltype(problem), constant_type>;
    for (const auto* method : {"EE","HE","RK33SSP","RK43SSP","RK64SSP","DOPRI54"})
    {
        double errors[2]{};
        for (int resolution = 0; resolution < 2; ++resolution)
        {
            constant_type constant({.1/(100*(resolution+1))});
            fixed_step_type fixed_step(operations, problem, constant, {method});
            nmfd::time_steppers::integration::time_integrator<operations_type, fixed_step_type> fixed(operations, fixed_step);
            fixed.set_time_interval(0, .1);
            fixed.apply(initial, result);
            if (fixed.get_status() != nmfd::time_steppers::integration_status::completed)
                throw std::runtime_error("lorenz: fixed integration failed");
            scfd::static_vec::vec<double,3> values;
            operations.get(result, values.d);
            for (int j = 0; j < 3; ++j)
                errors[resolution] = std::max(errors[resolution], std::abs(values[j]-reference[j]));
        }
        if (errors[1] > std::max(1e-10, .8*errors[0]))
            throw std::runtime_error("lorenz: refinement did not improve "+std::string(method));
    }
    std::cout << "lorenz: adaptive restart/reference and six fixed RK methods PASS\n";
}
catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
