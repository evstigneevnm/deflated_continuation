#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/detail/vector_wrap.h>
#include <stability/analysis/eigenvector_rank_aggregator.h>

#include "common/recycled_ritz_subspace_test_suite.h"

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
void set_column(
    const VectorSpace& vector_space,
    stability::eigensolvers::ritz_vector_storage<VectorSpace>& storage,
    std::size_t column,
    const std::vector<double>& real,
    const std::vector<double>& imaginary)
{
    using ordinal_type = typename VectorSpace::ordinal_type;
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        real_vector(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        imaginary_vector(vector_space);
    vector_space.set(real.data(), *real_vector, real.size());
    vector_space.set(
        imaginary.data(),
        *imaginary_vector,
        imaginary.size());
    vector_space.assign(
        *real_vector,
        storage.real(),
        static_cast<ordinal_type>(storage.capacity()),
        static_cast<ordinal_type>(column));
    vector_space.assign(
        *imaginary_vector,
        storage.imaginary(),
        static_cast<ordinal_type>(storage.capacity()),
        static_cast<ordinal_type>(column));
}

stability::eigensolvers::eigenpair_estimate<double> estimate(
    std::complex<double> value,
    std::size_t vector_index,
    double relative_residual,
    bool converged = true)
{
    stability::eigensolvers::eigenpair_estimate<double> result;
    result.value = value;
    result.projected_index = vector_index;
    result.residual = relative_residual;
    result.relative_residual = relative_residual;
    result.converged = converged;
    return result;
}

template<class Backend>
void run_backend(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using storage_type =
        stability::eigensolvers::ritz_vector_storage<
            vector_space_type>;
    using aggregator_type =
        stability::analysis::eigenvector_rank_aggregator<
            vector_space_type>;
    using result_type =
        stability::eigensolvers::eigensolver_result<double>;

    vector_space_type vector_space(3);
    storage_type vectors(vector_space, 5);
    set_column(
        vector_space,
        vectors,
        0,
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0});
    set_column(
        vector_space,
        vectors,
        1,
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0});
    set_column(
        vector_space,
        vectors,
        2,
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0});
    set_column(
        vector_space,
        vectors,
        3,
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0});
    set_column(
        vector_space,
        vectors,
        4,
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0});
    vectors.set_size(5);

    aggregator_type aggregator(vector_space);
    result_type first;
    first.eigenpairs = {
        estimate({2.0, 0.0}, 0, 1.0e-6),
        estimate({-1.0, 0.0}, 3, 1.0e-8),
        estimate({9.0, 0.0}, 2, 1.0e-12, false)};
    aggregator.add(first, vectors);
    require(
        aggregator.size() == 2,
        label + " accepts converged nonzero vectors only");

    result_type second;
    second.eigenpairs = {
        estimate({2.0 + 1.0e-10, 0.0}, 1, 1.0e-12),
        estimate({2.0 - 1.0e-10, 0.0}, 2, 1.0e-10),
        estimate({4.0, 0.0}, 4, 1.0e-12)};
    aggregator.add(second, vectors);

    const auto estimates = aggregator.estimates();
    const auto repeated = std::count_if(
        estimates.begin(),
        estimates.end(),
        [](const auto& value)
        {
            return std::abs(value.value.real() - 2.0) < 1.0e-6;
        });
    require(
        estimates.size() == 3 && repeated == 2,
        label +
            " preserves independent vectors at a repeated eigenvalue");

    const auto improved = std::find_if(
        estimates.begin(),
        estimates.end(),
        [](const auto& value)
        {
            return
                std::abs(value.value.real() - 2.0) < 1.0e-6 &&
                value.relative_residual == 1.0e-12;
        });
    require(
        improved != estimates.end(),
        label +
            " phase-equivalent rediscovery improves estimate metadata");

    result_type invalid;
    invalid.eigenpairs = {
        estimate({3.0, 0.0}, vectors.size(), 1.0e-12)};
    bool rejected_invalid_index = false;
    try
    {
        aggregator.add(invalid, vectors);
    }
    catch(const std::out_of_range&)
    {
        rejected_invalid_index = true;
    }
    require(
        rejected_invalid_index,
        label + " rejects a missing Ritz-vector index");

    vector_space_type recycling_space(3);
    stability_tests::run_recycled_ritz_subspace_test_suite(
        recycling_space,
        label,
        require);
}

} // namespace

int main()
{
    run_backend<scfd::backend::serial_cpu>("serial");
    run_backend<scfd::backend::omp>("OMP");
    std::cout
        << "Eigenvector rank aggregation checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
