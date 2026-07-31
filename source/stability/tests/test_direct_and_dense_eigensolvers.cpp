#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <stability/analysis/dimension_guarded_eigensolver.h>
#include <stability/eigensolvers/direct_scalar_eigensolver.h>
#include <stability/eigensolvers/host_dense_operator_eigensolver.h>

#include "common/analytical_dense_operator.h"
#include "common/analytical_eigenproblem.h"

namespace
{

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

template<class VectorSpace>
class scalar_operator
{
public:
    using vector_type = typename VectorSpace::vector_type;

    scalar_operator(VectorSpace& vector_space, double value)
        : vector_space_(vector_space),
          value_(value)
    {
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        vector_space_.assign_mul(value_, source, destination);
        return true;
    }

private:
    VectorSpace& vector_space_;
    double value_;
};

template<class VectorSpace>
class scripted_eigensolver
{
public:
    using vector_type = typename VectorSpace::vector_type;
    using real_type = typename VectorSpace::norm_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit scripted_eigensolver(bool succeeds)
        : succeeds_(succeeds)
    {
    }

    result_type execute(const vector_type&) const
    {
        ++calls_;
        result_type result;
        result.status = succeeds_
            ? stability::eigensolvers::eigensolver_status::success
            : stability::eigensolvers::eigensolver_status::
                  inner_solver_failure;
        result.scans_succeeded = succeeds_ ? 1 : 0;
        result.coverage_complete = succeeds_;
        result.diagnostic =
            succeeds_ ? "scripted success" : "scripted failure";
        return result;
    }

    std::size_t calls() const
    {
        return calls_;
    }

private:
    bool succeeds_;
    mutable std::size_t calls_ = 0;
};

template<class Backend>
void run_backend(const std::string& label)
{
    using scalar_space_type =
        scfd_vector_operations<Backend, double>;
    scalar_space_type scalar_space(1);
    scalar_operator<scalar_space_type> scalar_op(
        scalar_space,
        -3.25);
    stability::eigensolvers::direct_scalar_eigensolver<
        scalar_space_type,
        scalar_operator<scalar_space_type>>
        direct(scalar_space, scalar_op);
    typename scalar_space_type::vector_type initial;
    scalar_space.init_vector(initial);
    scalar_space.start_use_vector(initial);
    scalar_space.assign_scalar(1.0, initial);
    const auto direct_result = direct.execute(initial);
    require(
        direct_result.succeeded() &&
            direct_result.eigenpairs.size() == 1 &&
            std::abs(
                direct_result.eigenpairs.front().value.real() +
                3.25) < 1.0e-13 &&
            direct_result.eigenpairs.front().residual < 1.0e-13,
        label + " direct scalar eigenvalue");
    scalar_space.stop_use_vector(initial);
    scalar_space.free_vector(initial);

    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    const auto problem =
        stability::tests::symmetric_eigenproblem<double>();
    vector_space_type vector_space(problem.dimension());
    stability::tests::analytical_dense_operator<
        vector_space_type,
        double>
        matrix_operator(vector_space, problem);
    nmfd::operations::linalg::host_small_dense_lapack<double>
        lapack;
    stability::eigensolvers::host_dense_operator_eigensolver<
        vector_space_type,
        decltype(matrix_operator),
        decltype(lapack)>
        dense(vector_space, matrix_operator, lapack);
    typename vector_space_type::vector_type dense_initial;
    vector_space.init_vector(dense_initial);
    vector_space.start_use_vector(dense_initial);
    vector_space.assign_scalar(1.0, dense_initial);
    const auto dense_result = dense.execute(dense_initial);
    require(
        dense_result.succeeded() &&
            dense_result.eigenpairs.size() == problem.dimension() &&
            dense_result.operator_calls == problem.dimension(),
        label + " dense operator oracle status");

    std::vector<double> actual;
    for(const auto& estimate : dense_result.eigenpairs)
    {
        require(
            estimate.converged &&
                std::abs(estimate.value.imag()) < 1.0e-12,
            label + " dense oracle residual");
        actual.push_back(estimate.value.real());
    }
    std::sort(actual.begin(), actual.end());
    std::vector<double> expected;
    for(const auto& pair : problem.eigenpairs())
        expected.push_back(pair.value.real());
    std::sort(expected.begin(), expected.end());
    require(
        actual.size() == expected.size(),
        label + " dense oracle eigenvalue count");
    for(std::size_t index = 0; index < actual.size(); ++index)
    {
        require(
            std::abs(actual[index] - expected[index]) < 1.0e-10,
            label + " dense oracle eigenvalue " +
                std::to_string(index));
    }

    scripted_eigensolver<vector_space_type> failing_primary(false);
    using guarded_type =
        stability::analysis::dimension_guarded_eigensolver<
            scripted_eigensolver<vector_space_type>,
            decltype(dense)>;
    guarded_type fallback(
        failing_primary,
        dense,
        problem.dimension(),
        problem.dimension(),
        false);
    const auto fallback_result = fallback.execute(dense_initial);
    require(
        failing_primary.calls() == 1 &&
            fallback_result.succeeded() &&
            fallback_result.eigenpairs.size() ==
                problem.dimension() &&
            fallback_result.diagnostic.find(
                "small-system recovery") != std::string::npos,
        label + " dimension-guarded fallback");

    scripted_eigensolver<vector_space_type> skipped_primary(false);
    guarded_type preferred(
        skipped_primary,
        dense,
        problem.dimension(),
        problem.dimension(),
        true);
    const auto preferred_result = preferred.execute(dense_initial);
    require(
        skipped_primary.calls() == 0 &&
            preferred_result.succeeded() &&
            preferred_result.diagnostic.find(
                "small-system eigensolver selected") !=
                std::string::npos,
        label + " preferred small-system eigensolver");

    scripted_eigensolver<vector_space_type>
        confirmation_primary(true);
    guarded_type confirmation(
        confirmation_primary,
        dense,
        problem.dimension(),
        problem.dimension(),
        false);
    const auto confirmation_result =
        confirmation.execute_classification_confirmation(
            dense_initial);
    require(
        confirmation.classification_confirmation_available() &&
            confirmation_primary.calls() == 0 &&
            confirmation_result.succeeded() &&
            confirmation_result.eigenpairs.size() ==
                problem.dimension() &&
            confirmation_result.diagnostic.find(
                "classification confirmation") !=
                std::string::npos,
        label + " dimension-guarded exact classification "
                "confirmation");

    scripted_eigensolver<vector_space_type> guarded_primary(false);
    guarded_type unavailable(
        guarded_primary,
        dense,
        problem.dimension(),
        problem.dimension() - 1,
        false);
    const auto unavailable_result =
        unavailable.execute(dense_initial);
    require(
        guarded_primary.calls() == 1 &&
            !unavailable_result.succeeded() &&
            unavailable_result.diagnostic == "scripted failure",
        label + " dimension guard preserves primary failure");
    require(
        !unavailable.classification_confirmation_available(),
        label + " dimension guard disables unavailable confirmation");

    vector_space.stop_use_vector(dense_initial);
    vector_space.free_vector(dense_initial);
}

} // namespace

int main()
{
    run_backend<scfd::backend::serial_cpu>("serial");
    run_backend<scfd::backend::omp>("OMP");
    std::cout
        << "Direct/dense eigensolver checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
