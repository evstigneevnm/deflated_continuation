#ifndef STABILITY_TESTS_TRACKED_INVARIANT_SUBSPACE_TEST_SUITE_H
#define STABILITY_TESTS_TRACKED_INVARIANT_SUBSPACE_TEST_SUITE_H

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nmfd/detail/vector_wrap.h>

#include <stability/analysis/eigenvector_rank_aggregator.h>
#include <stability/tracking/tracked_invariant_subspace.h>

namespace stability_tests
{

template<class VectorSpace>
class tracked_dense_three_operator
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    tracked_dense_three_operator(
        VectorSpace& vector_space,
        std::array<double, 9> matrix)
        : vector_space_(vector_space),
          matrix_(std::move(matrix))
    {
        if(vector_space_.get_default_size() != 3)
        {
            throw std::invalid_argument(
                "tracked test operator requires dimension three");
        }
        for(std::size_t index = 0; index < 3; ++index)
        {
            coordinates_[index] =
                std::make_unique<vector_wrap_type>(vector_space_);
            std::vector<double> values(3, 0.0);
            values[index] = 1.0;
            vector_space_.set(
                values.data(),
                **coordinates_[index],
                values.size());
        }
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        std::array<scalar_type, 3> coefficients{};
        for(std::size_t col = 0; col < 3; ++col)
        {
            coefficients[col] = vector_space_.scalar_prod(
                **coordinates_[col],
                source);
        }
        vector_space_.assign_scalar(scalar_type{}, destination);
        for(std::size_t row = 0; row < 3; ++row)
        {
            scalar_type value{};
            for(std::size_t col = 0; col < 3; ++col)
            {
                value += scalar_type(matrix_[row + 3*col])*
                    coefficients[col];
            }
            vector_space_.add_lin_comb(
                value,
                **coordinates_[row],
                scalar_type(1),
                destination);
        }
        return vector_space_.check_is_valid_number(destination);
    }

private:
    using vector_wrap_type =
        nmfd::detail::vector_wrap<VectorSpace, true, true>;

    VectorSpace& vector_space_;
    std::array<double, 9> matrix_{};
    std::array<std::unique_ptr<vector_wrap_type>, 3> coordinates_;
};

template<class VectorSpace>
void set_tracked_column(
    const VectorSpace& vector_space,
    stability::eigensolvers::ritz_vector_storage<VectorSpace>& storage,
    std::size_t column,
    const std::vector<double>& real,
    const std::vector<double>& imaginary = {0.0, 0.0, 0.0})
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

inline stability::eigensolvers::eigenpair_estimate<double>
tracked_estimate(
    std::complex<double> value,
    std::size_t index)
{
    stability::eigensolvers::eigenpair_estimate<double> result;
    result.value = value;
    result.residual = 1.0e-14;
    result.relative_residual = 1.0e-14;
    result.converged = true;
    result.projected_index = index;
    return result;
}

template<class VectorSpace>
void populate_tracked_aggregator(
    stability::eigensolvers::ritz_vector_storage<VectorSpace>& storage,
    std::vector<stability::eigensolvers::eigenpair_estimate<double>>
        estimates,
    stability::analysis::eigenvector_rank_aggregator<VectorSpace>&
        aggregator)
{
    stability::eigensolvers::eigensolver_result<double> result;
    result.status =
        stability::eigensolvers::eigensolver_status::success;
    result.eigenpairs = std::move(estimates);
    aggregator.add(result, storage);
}

template<class VectorSpace, class Require>
void run_tracked_invariant_subspace_test_suite(
    VectorSpace& vector_space,
    const std::string& label,
    Require&& require)
{
    using storage_type =
        stability::eigensolvers::ritz_vector_storage<VectorSpace>;
    using tracker_type =
        stability::tracking::tracked_invariant_subspace<VectorSpace>;
    using aggregator_type =
        stability::analysis::eigenvector_rank_aggregator<VectorSpace>;

    typename tracker_type::options_type options;
    options.enabled = true;
    options.maximum_dimension = 3;
    options.maximum_seed_vectors = 2;
    options.seed_innovation_weight = 0.2;
    options.absolute_invariance_tolerance = 1.0e-12;
    options.relative_invariance_tolerance = 1.0e-10;
    options.dependence_tolerance = 1.0e-10;
    options.minimum_retained_residual_ratio = 0.05;
    options.principal_angle_rank_tolerance = 1.0e-8;

    storage_type coordinate_vectors(vector_space, 2);
    set_tracked_column(
        vector_space, coordinate_vectors, 0, {1.0, 0.0, 0.0});
    set_tracked_column(
        vector_space, coordinate_vectors, 1, {0.0, 1.0, 0.0});
    coordinate_vectors.set_size(2);
    aggregator_type coordinate_aggregator(vector_space);
    populate_tracked_aggregator(
        coordinate_vectors,
        {tracked_estimate({2.0, 0.0}, 0),
         tracked_estimate({2.0, 0.0}, 1)},
        coordinate_aggregator);

    tracker_type tracker(vector_space, options);
    tracker.begin_transaction();
    require(
        tracker.stage(coordinate_aggregator) == 2,
        label + " stages both directions of a repeated eigenspace");
    require(
        tracker.size() == 0 && tracker.seed_size() == 2,
        label +
            " staged invariant subspace is visible only inside its "
            "transaction");
    tracker.rollback_transaction();
    require(
        tracker.size() == 0 && tracker.seed_size() == 0,
        label + " invariant-subspace rollback discards staged columns");
    tracker.begin_transaction();
    tracker.stage(coordinate_aggregator);
    tracker.commit_transaction();
    require(
        tracker.size() == 2,
        label + " commit publishes the real invariant plane");

    nmfd::detail::vector_wrap<VectorSpace, true, true>
        innovation(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        first_seed(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true>
        second_seed(vector_space);
    const double innovation_values[] = {1.0, -2.0, 3.0};
    vector_space.set(innovation_values, *innovation, 3);
    require(
        tracker.make_seed(0, *innovation, *first_seed) &&
            tracker.make_seed(1, *innovation, *second_seed) &&
            std::abs(vector_space.norm(*first_seed) - 1.0) < 1.0e-11 &&
            std::abs(vector_space.norm(*second_seed) - 1.0) < 1.0e-11,
        label + " creates normalized seeds from separate tracked columns");
    require(
        std::abs(
            vector_space.scalar_prod(*first_seed, *second_seed)) < 0.1,
        label + " separate tracked directions remain distinguishable");

    // Column-major matrix with a rotating 2x2 block and a separate mode.
    tracked_dense_three_operator<VectorSpace> rotating_operator(
        vector_space,
        {2.0, 3.0, 0.0,
         -3.0, 2.0, 0.0,
         0.0, 0.0, -1.0});
    const auto rotating = tracker.validate(rotating_operator);
    require(
        rotating.accepted &&
            rotating.dimension == 2 &&
            rotating.operator_calls == 2 &&
            rotating.residual_frobenius < 1.0e-11,
        label +
            " validates a rotating plane although its columns are not "
            "individual real eigenvectors");
    require(
        std::abs(rotating.projected_operator(0, 1) + 3.0) < 1.0e-11 &&
            std::abs(rotating.projected_operator(1, 0) - 3.0) < 1.0e-11,
        label + " retains the projected dynamics inside the plane");

    tracked_dense_three_operator<VectorSpace> leaking_operator(
        vector_space,
        {2.0, 3.0, 1.0,
         -3.0, 2.0, 0.0,
         1.0, 0.0, -1.0});
    const auto leaking = tracker.validate(leaking_operator);
    require(
        !leaking.accepted && leaking.residual_frobenius > 0.5,
        label + " rejects a plane with a substantial external residual");

    storage_type mixed_vectors(vector_space, 2);
    set_tracked_column(
        vector_space, mixed_vectors, 0, {1.0, 0.0, 0.0});
    set_tracked_column(
        vector_space, mixed_vectors, 1, {0.0, 1.0, 0.0});
    mixed_vectors.set_size(2);
    aggregator_type mixed_aggregator(vector_space);
    populate_tracked_aggregator(
        mixed_vectors,
        {tracked_estimate({2.0, 0.0}, 0),
         tracked_estimate({-1.0, 0.0}, 1)},
        mixed_aggregator);
    tracker_type mixed_tracker(vector_space, options);
    mixed_tracker.stage(mixed_aggregator);
    tracked_dense_three_operator<VectorSpace> partially_stale_operator(
        vector_space,
        {2.0, 0.0, 1.0,
         0.0, -1.0, 0.0,
         0.0, 0.0, -2.0});
    const auto partially_valid =
        mixed_tracker.validate(partially_stale_operator);
    require(
        partially_valid.accepted &&
            partially_valid.groups.size() == 2 &&
            partially_valid.accepted_indices.size() == 1 &&
            partially_valid.accepted_indices.front() == 1,
        label +
            " block validation retains a valid eigendirection while "
            "rejecting a stale block");
    require(
        mixed_tracker.make_seed(
            0,
            *innovation,
            partially_valid.accepted_indices,
            *first_seed),
        label + " accepted block can seed a new Krylov probe");
    const auto accepted_alignment = vector_space.scalar_prod(
        mixed_vectors.real(),
        static_cast<typename VectorSpace::ordinal_type>(
            mixed_vectors.capacity()),
        static_cast<typename VectorSpace::ordinal_type>(1),
        *first_seed);
    require(
        std::abs(accepted_alignment) > 0.95,
        label + " block seed uses the selected valid eigendirection");

    const double inverse_sqrt_two = 1.0/std::sqrt(2.0);
    storage_type rotated_vectors(vector_space, 2);
    set_tracked_column(
        vector_space,
        rotated_vectors,
        0,
        {inverse_sqrt_two, inverse_sqrt_two, 0.0});
    set_tracked_column(
        vector_space,
        rotated_vectors,
        1,
        {-inverse_sqrt_two, inverse_sqrt_two, 0.0});
    rotated_vectors.set_size(2);
    aggregator_type rotated_aggregator(vector_space);
    populate_tracked_aggregator(
        rotated_vectors,
        {tracked_estimate({2.0, 0.0}, 0),
         tracked_estimate({2.0, 0.0}, 1)},
        rotated_aggregator);
    tracker_type rotated_tracker(vector_space, options);
    rotated_tracker.stage(rotated_aggregator);
    const auto same_plane = tracker.overlap_with(rotated_tracker);
    require(
        same_plane.dimension_gap == 0 &&
            same_plane.numerical_rank == 2 &&
            same_plane.maximum_angle < 1.0e-7,
        label + " principal angles ignore an internal basis rotation");
    tracker.begin_transaction();
    tracker.stage(rotated_aggregator);
    require(
        tracker.pending_overlap() &&
            tracker.pending_overlap()->maximum_angle < 1.0e-7,
        label + " stages basis-invariant continuation diagnostics");
    tracker.rollback_transaction();
    require(
        tracker.size() == 2 && !tracker.pending_overlap(),
        label + " rollback preserves the committed tracking plane");

    const double small_angle = 0.01;
    storage_type slightly_tilted_vectors(vector_space, 2);
    set_tracked_column(
        vector_space,
        slightly_tilted_vectors,
        0,
        {std::cos(small_angle), 0.0, std::sin(small_angle)});
    set_tracked_column(
        vector_space,
        slightly_tilted_vectors,
        1,
        {0.0, 1.0, 0.0});
    slightly_tilted_vectors.set_size(2);
    aggregator_type slightly_tilted_aggregator(vector_space);
    populate_tracked_aggregator(
        slightly_tilted_vectors,
        {tracked_estimate({2.0, 0.0}, 0),
         tracked_estimate({2.0, 0.0}, 1)},
        slightly_tilted_aggregator);
    tracker.begin_transaction();
    tracker.stage(slightly_tilted_aggregator);
    tracker.copy_column(0, *first_seed);
    const auto tilted_alignment = vector_space.scalar_prod(
        slightly_tilted_vectors.real(),
        static_cast<typename VectorSpace::ordinal_type>(
            slightly_tilted_vectors.capacity()),
        static_cast<typename VectorSpace::ordinal_type>(0),
        *first_seed);
    require(
        tracker.pending_retained_columns() == 0 &&
            std::abs(tilted_alignment) > 1.0 - 1.0e-10,
        label +
            " small subspace rotation keeps the fresh basis instead "
            "of retaining every stale column");
    tracker.rollback_transaction();

    storage_type incomplete_vectors(vector_space, 1);
    set_tracked_column(
        vector_space,
        incomplete_vectors,
        0,
        {1.0, 0.0, 0.0});
    incomplete_vectors.set_size(1);
    aggregator_type incomplete_aggregator(vector_space);
    populate_tracked_aggregator(
        incomplete_vectors,
        {tracked_estimate({2.0, 0.0}, 0)},
        incomplete_aggregator);
    tracker.begin_transaction();
    tracker.stage(incomplete_aggregator);
    require(
        tracker.seed_size() == 2 &&
            tracker.pending_retained_columns() == 1 &&
            tracker.pending_overlap() &&
            tracker.pending_overlap()->dimension_gap == 0,
        label +
            " incomplete recovery retains the missing previous "
            "subspace direction");
    tracker.rollback_transaction();

    storage_type partial_vectors(vector_space, 2);
    set_tracked_column(
        vector_space, partial_vectors, 0, {0.0, 1.0, 0.0});
    set_tracked_column(
        vector_space, partial_vectors, 1, {0.0, 0.0, 1.0});
    partial_vectors.set_size(2);
    aggregator_type partial_aggregator(vector_space);
    populate_tracked_aggregator(
        partial_vectors,
        {tracked_estimate({2.0, 0.0}, 0),
         tracked_estimate({-1.0, 0.0}, 1)},
        partial_aggregator);
    tracker_type partial_tracker(vector_space, options);
    partial_tracker.stage(partial_aggregator);
    const auto partial_overlap = tracker.overlap_with(partial_tracker);
    require(
        partial_overlap.numerical_rank == 1 &&
            std::abs(
                partial_overlap.maximum_angle -
                std::acos(-1.0)/2.0) < 1.0e-7,
        label + " principal angles detect a lost subspace direction");

    tracker.begin_transaction();
    tracker.stage(partial_aggregator);
    tracker.stage(incomplete_aggregator);
    require(
        tracker.seed_size() == 3 &&
            tracker.pending_retained_columns() == 2,
        label +
            " independent confirmation runs accumulate directions in "
            "the staged transaction");
    tracker.rollback_transaction();

    storage_type complex_vectors(vector_space, 2);
    set_tracked_column(
        vector_space,
        complex_vectors,
        0,
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0});
    set_tracked_column(
        vector_space,
        complex_vectors,
        1,
        {1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0});
    complex_vectors.set_size(2);
    aggregator_type complex_aggregator(vector_space);
    populate_tracked_aggregator(
        complex_vectors,
        {tracked_estimate({2.0, 3.0}, 0),
         tracked_estimate({2.0, -3.0}, 1)},
        complex_aggregator);
    tracker_type complex_tracker(vector_space, options);
    complex_tracker.stage(complex_aggregator);
    require(
        complex_tracker.size() == 2 &&
            !complex_tracker.descriptor(0).imaginary_component &&
            complex_tracker.descriptor(1).imaginary_component,
        label + " stores one real plane for a conjugate eigenvalue pair");

    storage_type negative_complex_vector(vector_space, 1);
    set_tracked_column(
        vector_space,
        negative_complex_vector,
        0,
        {1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0});
    negative_complex_vector.set_size(1);
    aggregator_type negative_complex_aggregator(vector_space);
    populate_tracked_aggregator(
        negative_complex_vector,
        {tracked_estimate({2.0, -3.0}, 0)},
        negative_complex_aggregator);
    tracker_type negative_complex_tracker(vector_space, options);
    negative_complex_tracker.stage(negative_complex_aggregator);
    require(
        negative_complex_tracker.size() == 2,
        label +
            " stores a real plane when only the negative conjugate is "
            "available");
}

} // namespace stability_tests

#endif
