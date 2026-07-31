#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <scfd/utils/log.h>

#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include <contrib/nmfd-linsolvers/test/solvers/linear_operator_advection.h>
#include <contrib/nmfd-linsolvers/test/solvers/linear_operator_diffusion.h>
#include <contrib/nmfd-linsolvers/test/solvers/linear_operator_elliptic.h>
#include <contrib/nmfd-linsolvers/test/solvers/preconditioner_advection.h>
#include <contrib/nmfd-linsolvers/test/solvers/preconditioner_diffusion.h>
#include <contrib/nmfd-linsolvers/test/solvers/preconditioner_elliptic.h>
#include <contrib/nmfd-linsolvers/test/solvers/residual_regularization_test.h>

#include "nmfd_cpu_reference_vector_space.h"

namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL," << message << std::endl;
    }
}

template<class Base>
class counting_operator : public Base
{
public:
    using Base::Base;
    using vector_type = typename Base::vector_type;

    void apply(const vector_type& source, vector_type& destination) const
    {
        ++calls_;
        Base::apply(source, destination);
    }

    std::size_t calls() const
    {
        return calls_;
    }

    void reset_calls() const
    {
        calls_ = 0;
    }

private:
    mutable std::size_t calls_ = 0;
};

template<class Base>
class counting_preconditioner : public Base
{
public:
    using Base::Base;
    using vector_type = typename Base::T_vec;

    void apply(vector_type& vector) const
    {
        ++calls_;
        Base::apply(vector);
    }

    std::size_t calls() const
    {
        return calls_;
    }

    void reset_calls() const
    {
        calls_ = 0;
    }

private:
    mutable std::size_t calls_ = 0;
};

struct solve_measurement
{
    bool converged = false;
    int iterations = 0;
    double monitor_residual = 0.0;
    double monitor_residual_out = 0.0;
    double physical_residual = 0.0;
    double physical_relative_residual = 0.0;
    std::size_t operator_calls = 0;
    std::size_t preconditioner_calls = 0;
    std::size_t history_entries = 0;
};

template<
    class Solver,
    class Operator,
    class VectorSpace,
    class ResetPreconditioner,
    class GetPreconditionerCalls>
solve_measurement measure_solve(
    const std::string& case_name,
    Solver& solver,
    const std::shared_ptr<Operator>& matrix_operator,
    const std::shared_ptr<VectorSpace>& vector_space,
    const typename VectorSpace::vector_type& rhs,
    typename VectorSpace::vector_type& solution,
    ResetPreconditioner reset_preconditioner,
    GetPreconditionerCalls get_preconditioner_calls)
{
    using vector_type = typename VectorSpace::vector_type;

    matrix_operator->reset_calls();
    reset_preconditioner();
    solve_measurement measurement;
    measurement.converged = solver.solve(rhs, solution);
    measurement.iterations = solver.monitor().iters_performed();
    measurement.monitor_residual = solver.monitor().resid_norm();
    measurement.monitor_residual_out = solver.monitor().resid_norm_out();
    measurement.operator_calls = matrix_operator->calls();
    measurement.preconditioner_calls = get_preconditioner_calls();
    measurement.history_entries =
        solver.monitor().convergence_history().size();

    vector_type residual;
    vector_space->init_vector(residual);
    vector_space->start_use_vector(residual);
    matrix_operator->apply(solution, residual);
    vector_space->add_lin_comb(1.0, rhs, -1.0, residual);
    measurement.physical_residual = vector_space->norm(residual);
    const double rhs_norm = vector_space->norm(rhs);
    measurement.physical_relative_residual =
        measurement.physical_residual/std::max(rhs_norm, 1e-300);
    vector_space->stop_use_vector(residual);
    vector_space->free_vector(residual);

    std::cout
        << "RESULT," << case_name
        << "," << (measurement.converged ? 1 : 0)
        << "," << measurement.iterations
        << "," << measurement.monitor_residual
        << "," << measurement.monitor_residual_out
        << "," << measurement.physical_residual
        << "," << measurement.physical_relative_residual
        << "," << measurement.operator_calls
        << "," << measurement.preconditioner_calls
        << "," << measurement.history_entries
        << std::endl;

    require(measurement.converged, case_name + ",did_not_converge");
    require(
        measurement.physical_relative_residual <= 1e-8,
        case_name + ",physical_residual_too_large");
    return measurement;
}

template<class VectorSpace>
typename VectorSpace::vector_type make_rhs(
    const std::shared_ptr<VectorSpace>& vector_space,
    bool double_frequency,
    bool zero_boundaries)
{
    typename VectorSpace::vector_type rhs;
    vector_space->init_vector(rhs);
    vector_space->start_use_vector(rhs);
    const std::size_t count = static_cast<std::size_t>(vector_space->size());
    const double frequency = double_frequency ? 2.0 : 1.0;
    for(std::size_t index = 0; index < count; ++index)
    {
        rhs[index] = std::sin(
            frequency*pi*static_cast<double>(index)/
            static_cast<double>(count - 1));
    }
    if(zero_boundaries)
        rhs[0] = rhs[count - 1] = 0.0;
    return rhs;
}

template<class VectorSpace>
void release_vector(
    const std::shared_ptr<VectorSpace>& vector_space,
    typename VectorSpace::vector_type& vector)
{
    vector_space->stop_use_vector(vector);
    vector_space->free_vector(vector);
}

template<class Solver, class Operator, class Preconditioner, class VectorSpace>
void run_preconditioned_pair(
    const std::string& prefix,
    const std::shared_ptr<VectorSpace>& vector_space,
    const std::shared_ptr<Operator>& matrix_operator,
    const std::shared_ptr<Preconditioner>& preconditioner,
    const typename Solver::params& parameters,
    typename VectorSpace::vector_type& rhs,
    double reuse_factor)
{
    typename VectorSpace::vector_type solution;
    vector_space->init_vector(solution);
    vector_space->start_use_vector(solution);
    vector_space->assign_scalar(0.0, solution);

    Solver solver(
        matrix_operator,
        vector_space,
        nullptr,
        parameters,
        preconditioner);
    auto reset_preconditioner =
        [&preconditioner]() { preconditioner->reset_calls(); };
    auto get_preconditioner_calls =
        [&preconditioner]() { return preconditioner->calls(); };

    measure_solve(
        prefix + "_zero",
        solver,
        matrix_operator,
        vector_space,
        rhs,
        solution,
        reset_preconditioner,
        get_preconditioner_calls);
    vector_space->add_mul_scalar(0.0, reuse_factor, solution);
    measure_solve(
        prefix + "_reused",
        solver,
        matrix_operator,
        vector_space,
        rhs,
        solution,
        reset_preconditioner,
        get_preconditioner_calls);

    release_vector(vector_space, solution);
}

template<class Solver, class Operator, class VectorSpace>
void run_unpreconditioned_pair(
    const std::string& prefix,
    const std::shared_ptr<VectorSpace>& vector_space,
    const std::shared_ptr<Operator>& matrix_operator,
    const typename Solver::params& parameters,
    typename VectorSpace::vector_type& rhs,
    double reuse_factor)
{
    typename VectorSpace::vector_type solution;
    vector_space->init_vector(solution);
    vector_space->start_use_vector(solution);
    vector_space->assign_scalar(0.0, solution);

    Solver solver(matrix_operator, vector_space, nullptr, parameters);
    auto reset_preconditioner = []() {};
    auto get_preconditioner_calls = []() { return std::size_t(0); };

    measure_solve(
        prefix + "_zero",
        solver,
        matrix_operator,
        vector_space,
        rhs,
        solution,
        reset_preconditioner,
        get_preconditioner_calls);
    vector_space->add_mul_scalar(0.0, reuse_factor, solution);
    measure_solve(
        prefix + "_reused",
        solver,
        matrix_operator,
        vector_space,
        rhs,
        solution,
        reset_preconditioner,
        get_preconditioner_calls);

    release_vector(vector_space, solution);
}

} // namespace

int main()
{
    using scalar_type = double;
    using log_type = scfd::utils::log_std;
    using vector_space_type =
        nmfd::tests::cpu_reference_vector_space<scalar_type>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;

    std::cout << std::setprecision(17);
    std::cout
        << "HEADER,case,converged,iterations,monitor_residual,"
        << "monitor_residual_out,physical_residual,"
        << "physical_relative_residual,operator_calls,"
        << "preconditioner_calls,history_entries"
        << std::endl;

    {
        using base_operator =
            tests::linear_operator_diffusion<vector_space_type, log_type>;
        using operator_type = counting_operator<base_operator>;
        using base_preconditioner =
            tests::preconditioner_diffusion<
                vector_space_type,
                operator_type,
                log_type>;
        using preconditioner_type =
            counting_preconditioner<base_preconditioner>;
        using solver_type =
            nmfd::solvers::gmres<
                vector_space_type,
                monitor_type,
                log_type,
                operator_type,
                preconditioner_type>;

        auto vector_space =
            std::make_shared<vector_space_type>(500);
        auto matrix_operator =
            std::make_shared<operator_type>(*vector_space, 1.0);
        auto preconditioner =
            std::make_shared<preconditioner_type>(vector_space, 15);
        auto rhs = make_rhs(vector_space, false, true);

        typename solver_type::params parameters;
        parameters.basis_size = 25;
        parameters.batch_size = 5;
        parameters.monitor.rel_tol = 1e-10;
        parameters.monitor.max_iters_num = 300;
        parameters.monitor.save_convergence_history = true;
        parameters.orthogonalization = "mgs";
        parameters.reorthogonalization_policy = "none";

        parameters.preconditioner_side = 'L';
        run_preconditioned_pair<solver_type>(
            "diffusion_500_left",
            vector_space,
            matrix_operator,
            preconditioner,
            parameters,
            rhs,
            0.99999);
        parameters.preconditioner_side = 'R';
        run_preconditioned_pair<solver_type>(
            "diffusion_500_right",
            vector_space,
            matrix_operator,
            preconditioner,
            parameters,
            rhs,
            0.99999);
        release_vector(vector_space, rhs);
    }

    {
        using base_operator =
            tests::linear_operator_advection<vector_space_type, log_type>;
        using operator_type = counting_operator<base_operator>;
        using base_preconditioner =
            tests::preconditioner_advection<
                vector_space_type,
                operator_type,
                log_type>;
        using preconditioner_type =
            counting_preconditioner<base_preconditioner>;
        using solver_type =
            nmfd::solvers::gmres<
                vector_space_type,
                monitor_type,
                log_type,
                operator_type,
                preconditioner_type>;

        auto vector_space =
            std::make_shared<vector_space_type>(500);
        auto matrix_operator =
            std::make_shared<operator_type>(*vector_space, 1.0, 1.0);
        auto preconditioner =
            std::make_shared<preconditioner_type>(vector_space, 1);
        auto rhs = make_rhs(vector_space, false, true);

        typename solver_type::params parameters;
        parameters.basis_size = 15;
        parameters.batch_size = 5;
        parameters.monitor.rel_tol = 1e-10;
        parameters.monitor.max_iters_num = 300;
        parameters.monitor.save_convergence_history = true;
        parameters.orthogonalization = "mgs";
        parameters.reorthogonalization_policy = "none";

        parameters.preconditioner_side = 'L';
        run_preconditioned_pair<solver_type>(
            "advection_500_left",
            vector_space,
            matrix_operator,
            preconditioner,
            parameters,
            rhs,
            0.9999999);
        parameters.preconditioner_side = 'R';
        run_preconditioned_pair<solver_type>(
            "advection_500_right",
            vector_space,
            matrix_operator,
            preconditioner,
            parameters,
            rhs,
            0.9999999);
        release_vector(vector_space, rhs);
    }

    {
        using diffusion_operator = counting_operator<
            tests::linear_operator_diffusion<vector_space_type, log_type>>;
        using diffusion_solver =
            nmfd::solvers::gmres<
                vector_space_type,
                monitor_type,
                log_type,
                diffusion_operator>;
        auto vector_space =
            std::make_shared<vector_space_type>(50);
        auto matrix_operator =
            std::make_shared<diffusion_operator>(*vector_space, 1.0);
        auto rhs = make_rhs(vector_space, false, true);

        typename diffusion_solver::params parameters;
        parameters.basis_size = 30;
        parameters.batch_size = 5;
        parameters.monitor.rel_tol = 1e-10;
        parameters.monitor.max_iters_num = 50;
        parameters.monitor.save_convergence_history = true;
        parameters.orthogonalization = "mgs";
        parameters.reorthogonalization_policy = "none";
        run_unpreconditioned_pair<diffusion_solver>(
            "diffusion_50_none",
            vector_space,
            matrix_operator,
            parameters,
            rhs,
            0.9999999);
        release_vector(vector_space, rhs);
    }

    {
        using advection_operator = counting_operator<
            tests::linear_operator_advection<vector_space_type, log_type>>;
        using advection_solver =
            nmfd::solvers::gmres<
                vector_space_type,
                monitor_type,
                log_type,
                advection_operator>;
        auto vector_space =
            std::make_shared<vector_space_type>(50);
        auto matrix_operator =
            std::make_shared<advection_operator>(*vector_space, 1.0, 1.0);
        auto rhs = make_rhs(vector_space, false, true);

        typename advection_solver::params parameters;
        parameters.basis_size = 50;
        parameters.batch_size = 5;
        parameters.monitor.rel_tol = 1e-10;
        parameters.monitor.max_iters_num = 50;
        parameters.monitor.save_convergence_history = true;
        parameters.orthogonalization = "mgs";
        parameters.reorthogonalization_policy = "none";
        run_unpreconditioned_pair<advection_solver>(
            "advection_50_none",
            vector_space,
            matrix_operator,
            parameters,
            rhs,
            0.9999999);
        release_vector(vector_space, rhs);
    }

    {
        using base_operator =
            tests::linear_operator_elliptic<vector_space_type, log_type>;
        using operator_type = counting_operator<base_operator>;
        using base_preconditioner =
            tests::preconditioner_elliptic<
                vector_space_type,
                operator_type,
                log_type>;
        using preconditioner_type =
            counting_preconditioner<base_preconditioner>;
        using regularization_type =
            nmfd::solvers::detail::residual_regularization_test<
                vector_space_type,
                log_type>;
        using solver_type =
            nmfd::solvers::gmres<
                vector_space_type,
                monitor_type,
                log_type,
                operator_type,
                preconditioner_type,
                regularization_type>;

        auto vector_space =
            std::make_shared<vector_space_type>(500);
        auto matrix_operator =
            std::make_shared<operator_type>(*vector_space);
        auto preconditioner =
            std::make_shared<preconditioner_type>(vector_space, 15);
        auto regularization =
            std::make_shared<regularization_type>(vector_space);
        auto rhs = make_rhs(vector_space, true, false);

        typename solver_type::params parameters;
        parameters.basis_size = 25;
        parameters.batch_size = 5;
        parameters.monitor.rel_tol = 1e-10;
        parameters.monitor.max_iters_num = 300;
        parameters.monitor.save_convergence_history = true;
        parameters.orthogonalization = "mgs";
        parameters.reorthogonalization_policy = "none";

        typename solver_type::vector_type solution;
        vector_space->init_vector(solution);
        vector_space->start_use_vector(solution);
        vector_space->assign_scalar(0.0, solution);

        parameters.preconditioner_side = 'L';
        {
            solver_type solver(
                matrix_operator,
                vector_space,
                nullptr,
                parameters,
                preconditioner,
                regularization);
            auto reset_preconditioner =
                [&preconditioner]() { preconditioner->reset_calls(); };
            auto get_preconditioner_calls =
                [&preconditioner]() { return preconditioner->calls(); };
            measure_solve(
                "elliptic_500_left_zero",
                solver,
                matrix_operator,
                vector_space,
                rhs,
                solution,
                reset_preconditioner,
                get_preconditioner_calls);
            vector_space->add_mul_scalar(0.0, 0.99999, solution);
            measure_solve(
                "elliptic_500_left_reused",
                solver,
                matrix_operator,
                vector_space,
                rhs,
                solution,
                reset_preconditioner,
                get_preconditioner_calls);
        }

        vector_space->assign_scalar(0.0, solution);
        parameters.preconditioner_side = 'R';
        {
            solver_type solver(
                matrix_operator,
                vector_space,
                nullptr,
                parameters,
                preconditioner,
                regularization);
            auto reset_preconditioner =
                [&preconditioner]() { preconditioner->reset_calls(); };
            auto get_preconditioner_calls =
                [&preconditioner]() { return preconditioner->calls(); };
            measure_solve(
                "elliptic_500_right_zero",
                solver,
                matrix_operator,
                vector_space,
                rhs,
                solution,
                reset_preconditioner,
                get_preconditioner_calls);
            vector_space->add_mul_scalar(0.0, 0.99999, solution);
            measure_solve(
                "elliptic_500_right_reused",
                solver,
                matrix_operator,
                vector_space,
                rhs,
                solution,
                reset_preconditioner,
                get_preconditioner_calls);
        }

        release_vector(vector_space, solution);
        release_vector(vector_space, rhs);
    }

    std::cout << "SUMMARY," << checks << "," << failures << std::endl;
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
