#ifndef __STABILITY_TESTS_RECYCLED_RITZ_SUBSPACE_TEST_SUITE_H__
#define __STABILITY_TESTS_RECYCLED_RITZ_SUBSPACE_TEST_SUITE_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <nmfd/detail/vector_wrap.h>

#include <stability/analysis/eigenvector_rank_aggregator.h>
#include <stability/analysis/recycled_ritz_subspace.h>

namespace stability_tests
{

template<class VectorSpace>
class recycled_diagonal_operator
{
public:
    using vector_type = typename VectorSpace::vector_type;

    recycled_diagonal_operator(
        VectorSpace& vector_space,
        const std::vector<double>& diagonal)
        : vector_space_(vector_space),
          diagonal_(vector_space)
    {
        vector_space_.set(
            diagonal.data(),
            *diagonal_,
            diagonal.size());
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        vector_space_.mul_pointwise(
            1.0,
            source,
            1.0,
            *diagonal_,
            destination);
        return vector_space_.check_is_valid_number(destination);
    }

private:
    VectorSpace& vector_space_;
    nmfd::detail::vector_wrap<VectorSpace, true, true> diagonal_;
};

template<class VectorSpace>
class recycled_coupled_operator
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    recycled_coupled_operator(
        VectorSpace& vector_space,
        const std::vector<double>& diagonal)
        : vector_space_(vector_space),
          diagonal_(vector_space, diagonal),
          source_coordinate_(vector_space),
          target_coordinate_(vector_space)
    {
        std::vector<double> source(
            vector_space_.get_default_size(), 0.0);
        std::vector<double> target(source.size(), 0.0);
        source.at(2) = 1.0;
        target.at(0) = 1.0;
        vector_space_.set(
            source.data(),
            *source_coordinate_,
            source.size());
        vector_space_.set(
            target.data(),
            *target_coordinate_,
            target.size());
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        if(!diagonal_.apply(source, destination))
            return false;
        const scalar_type coefficient =
            vector_space_.scalar_prod(
                *source_coordinate_,
                source);
        vector_space_.add_lin_comb(
            coefficient,
            *target_coordinate_,
            scalar_type(1),
            destination);
        return vector_space_.check_is_valid_number(destination);
    }

private:
    VectorSpace& vector_space_;
    recycled_diagonal_operator<VectorSpace> diagonal_;
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        source_coordinate_;
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        target_coordinate_;
};

template<class VectorSpace>
void set_recycled_column(
    const VectorSpace& vector_space,
    stability::eigensolvers::ritz_vector_storage<VectorSpace>& storage,
    std::size_t column,
    const std::vector<double>& values)
{
    using ordinal_type = typename VectorSpace::ordinal_type;
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        vector(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        zero(vector_space);
    vector_space.set(values.data(), *vector, values.size());
    vector_space.assign_scalar(0.0, *zero);
    vector_space.assign(
        *vector,
        storage.real(),
        static_cast<ordinal_type>(storage.capacity()),
        static_cast<ordinal_type>(column));
    vector_space.assign(
        *zero,
        storage.imaginary(),
        static_cast<ordinal_type>(storage.capacity()),
        static_cast<ordinal_type>(column));
}

template<class VectorSpace, class Require>
void run_recycled_ritz_subspace_test_suite(
    VectorSpace& vector_space,
    const std::string& label,
    Require&& require)
{
    using storage_type =
        stability::eigensolvers::ritz_vector_storage<VectorSpace>;
    using aggregator_type =
        stability::analysis::eigenvector_rank_aggregator<VectorSpace>;
    using cache_type =
        stability::analysis::recycled_ritz_subspace<VectorSpace>;
    using result_type =
        stability::eigensolvers::eigensolver_result<double>;

    storage_type vectors(vector_space, 3);
    set_recycled_column(
        vector_space, vectors, 0, {1.0, 0.0, 0.0});
    set_recycled_column(
        vector_space, vectors, 1, {0.0, 1.0, 0.0});
    set_recycled_column(
        vector_space, vectors, 2, {0.0, 0.0, 1.0});
    vectors.set_size(3);

    result_type spectrum;
    for(std::size_t index = 0; index < 3; ++index)
    {
        stability::eigensolvers::eigenpair_estimate<double>
            estimate;
        estimate.value = index < 2
            ? std::complex<double>(2.0, 0.0)
            : std::complex<double>(-1.0, 0.0);
        estimate.residual = 1.0e-14;
        estimate.relative_residual = 1.0e-14;
        estimate.converged = true;
        estimate.projected_index = index;
        spectrum.eigenpairs.push_back(estimate);
    }
    aggregator_type aggregator(vector_space);
    aggregator.add(spectrum, vectors);

    typename cache_type::options_type options;
    options.enabled = true;
    options.maximum_vectors = 3;
    options.innovation_weight = 0.2;
    options.absolute_residual_tolerance = 1.0e-12;
    options.relative_residual_tolerance = 1.0e-6;
    cache_type cache(vector_space, options);

    cache.begin_transaction();
    cache.stage(aggregator);
    require(
        cache.size() == 0,
        label + " staged vectors are not visible before commit");
    cache.rollback_transaction();
    require(
        cache.size() == 0,
        label + " rollback discards staged vectors");

    cache.begin_transaction();
    cache.stage(aggregator);
    cache.commit_transaction();
    require(
        cache.size() == 3,
        label + " commit publishes the recycled invariant subspace");

    recycled_diagonal_operator<VectorSpace> exact_operator(
        vector_space,
        {2.0, 2.0, -1.0});
    const auto exact = cache.validate(exact_operator);
    require(
        exact.accepted_indices.size() == 3 &&
            exact.rejected_vectors == 0,
        label + " exact Ritz vectors pass current-operator validation");

    recycled_diagonal_operator<VectorSpace> changed_operator(
        vector_space,
        {4.0, 5.0, -3.0});
    const auto changed = cache.validate(changed_operator);
    require(
        changed.accepted_indices.size() == 3 &&
            changed.rejected_vectors == 0,
        label + " current Rayleigh quotients track moving eigenvalues");

    recycled_coupled_operator<VectorSpace> coupled_operator(
        vector_space,
        {4.0, 5.0, -3.0});
    const auto coupled = cache.validate(coupled_operator);
    require(
        coupled.accepted_indices.size() == 2 &&
            coupled.rejected_vectors == 1,
        label + " vectors leaving the current invariant subspace are rejected");

    nmfd::detail::vector_wrap<VectorSpace, true, true>
        innovation(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        seed(vector_space);
    const double innovation_values[] = {1.0, -2.0, 3.0};
    vector_space.set(innovation_values, *innovation, 3);
    require(
        cache.make_seed(
            1,
            *innovation,
            coupled.accepted_indices,
            *seed),
        label + " validated subspace produces a recycled probe");
    require(
        std::abs(vector_space.norm(*seed) - 1.0) < 1.0e-11,
        label + " recycled probe is normalized");

    cache.clear();
    require(
        cache.size() == 0,
        label + " curve-boundary reset clears the cache");
}

} // namespace stability_tests

#endif
