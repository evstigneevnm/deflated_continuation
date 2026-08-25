#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(TEST_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#include <external_libraries/fft_facade_cufft.h>
#include <scfd/backend/cuda.h>
#else
#include <external_libraries/fft_facade_fftw.h>
#include <scfd/backend/omp.h>
#endif

#include <common/NMFD-operations/nmfd/operations/scfd_vector_operations.h>
#include <deflation/symmetry_solution_storage.h>
#include <discretization/fourier/codecs/full_real_field.h>
#include <discretization/fourier/codecs/inversion_odd_field.h>
#include <discretization/fourier/codecs/translation_equivariant_real_field.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h>
#include <nonlinear_operators/adjoint_jacobian_capability.h>
#include <nonlinear_operators/tests/adjoint_jacobian_test.h>
#include <nonlinear_operators/tests/linear_nonlinear_decomposition_test.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>
#include <symmetry/fourier/translation_action.h>

namespace
{

class report
{
public:
    void check(const bool value, const std::string& message)
    {
        ++checks_;
        if(!value)
        {
            ++failures_;
            std::cerr << "FAIL: " << message << std::endl;
        }
    }

    int finish(const std::string& backend) const
    {
        std::cout << backend << " KS2D operator: " << checks_ << " checks, "
                  << failures_ << " failures" << std::endl;
        return failures_ == 0 ? 0 : 1;
    }

private:
    std::size_t checks_ = 0;
    std::size_t failures_ = 0;
};

template<class T>
T relative_error(const std::vector<T>& actual, const std::vector<T>& expected)
{
    T error_squared = T(0);
    T expected_squared = T(0);
    for(std::size_t index = 0; index < actual.size(); ++index)
    {
        const T difference = actual[index] - expected[index];
        error_squared += difference*difference;
        expected_squared += expected[index]*expected[index];
    }
    return std::sqrt(error_squared)/(std::sqrt(expected_squared) + T(1));
}

template<class Backend, class FFTBackend, template<class, class> class StateCodec>
void run_case(
    report& result,
    const std::string& case_name,
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t state_size,
    const bool expect_inversion_odd,
    const bool expect_translation_equivariance,
    const std::size_t expected_finite_group_order = 1
)
{
    using scalar_type = double;
    using vector_operations_type = scfd_vector_operations<Backend, scalar_type>;
    using operator_type = nonlinear_operators::kuramoto_sivashinskiy_2d<
        vector_operations_type,
        FFTBackend,
        StateCodec
    >;
    using vector_type = typename vector_operations_type::vector_type;
    using copy_type = typename vector_operations_type::copy_type;

    const scalar_type lambda = 7.3;
    const scalar_type epsilon = 2.0e-7;
    vector_operations_type operations(state_size);
    operator_type ks2d(2.0, 4.0, nx, ny, &operations);
    const auto message = [&](const std::string& text)
    {
        return case_name + ": " + text;
    };

    result.check(ks2d.size() == state_size, message("correct state dimension"));
    result.check(ks2d.physical_size() == nx*ny, message("correct physical dimension"));

    vector_type zero;
    vector_type state;
    vector_type direction;
    vector_type cotangent;
    vector_type state_epsilon;
    vector_type f_zero;
    vector_type f_state;
    vector_type linear_state;
    vector_type nonlinear_state;
    vector_type split_state;
    vector_type f_perturbed;
    vector_type jacobian_direction;
    vector_type alpha_direction;
    vector_type f_lambda_perturbed;
    vector_type recovered;
    vector_type physical;
    vector_type seed_a;
    vector_type seed_a_repeat;
    vector_type seed_b;
    operations.init_vectors(
        zero,
        state,
        direction,
        cotangent,
        state_epsilon,
        f_zero,
        f_state,
        linear_state,
        nonlinear_state,
        split_state,
        f_perturbed,
        jacobian_direction,
        alpha_direction,
        f_lambda_perturbed,
        recovered,
        physical,
        seed_a,
        seed_a_repeat,
        seed_b
    );
    operations.start_use_vectors(
        zero,
        state,
        direction,
        cotangent,
        state_epsilon,
        f_zero,
        f_state,
        linear_state,
        nonlinear_state,
        split_state,
        f_perturbed,
        jacobian_direction,
        alpha_direction,
        f_lambda_perturbed,
        recovered,
        seed_a,
        seed_a_repeat,
        seed_b
    );
    operations.start_use_vector(physical, nx*ny);

    operations.assign_scalar(0.0, zero);
    ks2d.F(zero, lambda, f_zero);
    result.check(operations.norm_l2(f_zero) < 1.0e-14, message("zero state is an exact solution"));

    const scalar_type pi = std::acos(scalar_type(-1));
    const scalar_type amplitude = 0.1;
    typename operator_type::extent_type physical_extent(nx, ny);
    typename operator_type::transform_type reference_transform(physical_extent);
    typename operator_type::grid_type reference_grid(
        physical_extent,
        {scalar_type(2)*pi, scalar_type(2)*pi}
    );
    typename operator_type::wavevector_table_type reference_wavevectors(
        reference_grid,
        typename operator_type::index_space_type(physical_extent)
    );
    typename operator_type::physical_field_type reference_physical(physical_extent);
    typename operator_type::spectral_field_type reference_spectrum(
        typename operator_type::index_space_type(physical_extent).spectral_extent()
    );
    typename operator_type::spectral_field_type translated_spectrum(
        typename operator_type::index_space_type(physical_extent).spectral_extent()
    );
    typename operator_type::spectral_field_type residual_spectrum(
        typename operator_type::index_space_type(physical_extent).spectral_extent()
    );
    typename operator_type::spectral_field_type translated_residual_spectrum(
        typename operator_type::index_space_type(physical_extent).spectral_extent()
    );
    std::vector<scalar_type> reference_physical_host(nx*ny);
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        const scalar_type x = scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx);
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            reference_physical_host[ix*ny + iy] = amplitude*std::sin(x);
        }
    }
    copy_type()(
        static_cast<typename vector_operations_type::ordinal_type>(reference_physical_host.size()),
        reference_physical_host.data(),
        reference_physical.data()
    );
    reference_transform.forward(reference_physical, reference_spectrum);
    ks2d.state_codec().pack(reference_spectrum, state);
    ks2d.F(state, lambda, f_state);
    ks2d.physical_solution(f_state, physical);
    std::vector<scalar_type> analytical_residual_host(nx*ny);
    operations.get(physical, analytical_residual_host.data(), nx*ny);
    scalar_type maximum_analytical_error = 0.0;
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        const scalar_type x = scalar_type(2)*pi*static_cast<scalar_type>(ix)/static_cast<scalar_type>(nx);
        const scalar_type expected =
            amplitude*(scalar_type(4) - lambda)*std::sin(x) +
            lambda*amplitude*amplitude*std::sin(scalar_type(2)*x);
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            maximum_analytical_error = std::max(
                maximum_analytical_error,
                std::abs(analytical_residual_host[ix*ny + iy] - expected)
            );
        }
    }
    result.check(
        maximum_analytical_error < 2.0e-12,
        message("closed-form single-mode residual")
    );

    std::vector<scalar_type> state_host(state_size);
    std::vector<scalar_type> direction_host(state_size);
    std::vector<scalar_type> cotangent_host(state_size);
    std::vector<scalar_type> perturbed_host(state_size);
    for(std::size_t index = 0; index < state_size; ++index)
    {
        const scalar_type denominator = static_cast<scalar_type>((index + 1)*(index + 1)*(index + 1));
        state_host[index] = 0.05*std::sin(static_cast<scalar_type>(index + 1))/denominator;
        direction_host[index] = 0.03*std::cos(static_cast<scalar_type>(2*index + 1))/denominator;
        cotangent_host[index] =
            0.04*std::sin(static_cast<scalar_type>(3*index + 2))/
            static_cast<scalar_type>((index + 1)*(index + 2));
        perturbed_host[index] = state_host[index] + epsilon*direction_host[index];
    }
    operations.set(state_host.data(), state);
    operations.set(direction_host.data(), direction);
    operations.set(cotangent_host.data(), cotangent);
    operations.set(perturbed_host.data(), state_epsilon);

    ks2d.F(state, lambda, f_state);
    ks2d.linear_residual(state, lambda, linear_state);
    ks2d.nonlinear_residual(state, lambda, nonlinear_state);
    operations.assign_mul(
        scalar_type(1),
        linear_state,
        scalar_type(1),
        nonlinear_state,
        split_state);
    std::vector<scalar_type> f_state_host(state_size);
    std::vector<scalar_type> split_state_host(state_size);
    operations.get(f_state, f_state_host.data());
    operations.get(split_state, split_state_host.data());
    result.check(
        relative_error(split_state_host, f_state_host) < 2.0e-13,
        message("linear plus nonlinear residual equals F"));
    result.check(
        operations.norm_l2(nonlinear_state) > 1.0e-12,
        message("nonlinear residual is nonzero"));
    ks2d.set_linearization_point(state, lambda);
    ks2d.jacobian_u(direction, jacobian_direction);
    ks2d.F(state_epsilon, lambda, f_perturbed);
    std::vector<scalar_type> f_perturbed_host(state_size);
    std::vector<scalar_type> jacobian_host(state_size);
    operations.get(f_perturbed, f_perturbed_host.data());
    operations.get(jacobian_direction, jacobian_host.data());
    std::vector<scalar_type> finite_difference_host(state_size);
    for(std::size_t index = 0; index < state_size; ++index)
    {
        finite_difference_host[index] = (f_perturbed_host[index] - f_state_host[index])/epsilon;
    }
    result.check(
        relative_error(finite_difference_host, jacobian_host) < 2.0e-7,
        message("Jacobian finite-difference consistency")
    );

    ks2d.jacobian_alpha(alpha_direction);
    ks2d.F(state, lambda + epsilon, f_lambda_perturbed);
    std::vector<scalar_type> alpha_host(state_size);
    std::vector<scalar_type> lambda_perturbed_host(state_size);
    operations.get(alpha_direction, alpha_host.data());
    operations.get(f_lambda_perturbed, lambda_perturbed_host.data());
    for(std::size_t index = 0; index < state_size; ++index)
    {
        finite_difference_host[index] = (lambda_perturbed_host[index] - f_state_host[index])/epsilon;
    }
    result.check(
        relative_error(finite_difference_host, alpha_host) < 2.0e-8,
        message("parameter Jacobian finite-difference consistency")
    );

    nonlinear_operators::tests::check_linear_nonlinear_decomposition(
        operations,
        ks2d,
        state,
        direction,
        lambda,
        scalar_type(1.0e-6),
        scalar_type(2.0e-7),
        [&](const bool condition, const std::string& text)
        {
            result.check(condition, message(text));
        },
        "KS2D"
    );

    static_assert(
        nonlinear_operators::has_jacobian_u_adjoint_v<operator_type>,
        "KS2D must provide an adjoint Jacobian action");
    static_assert(
        nonlinear_operators::has_component_jacobian_u_adjoint_v<operator_type>,
        "KS2D must provide component adjoint Jacobian actions");
    static_assert(
        nonlinear_operators::has_affine_preconditioner_adjoint_pair_v<operator_type>,
        "KS2D must provide an affine preconditioner adjoint pair");
    nonlinear_operators::tests::check_adjoint_jacobian(
        operations,
        ks2d,
        state,
        direction,
        cotangent,
        lambda,
        scalar_type(8.0e-12),
        [&](const bool condition, const std::string& text)
        {
            result.check(condition, message(text));
        },
        "KS2D");
    nonlinear_operators::tests::check_affine_preconditioner_adjoint(
        operations,
        ks2d,
        state,
        direction,
        cotangent,
        lambda,
        scalar_type(0.7),
        scalar_type(1.1),
        scalar_type(8.0e-12),
        [&](const bool condition, const std::string& text)
        {
            result.check(condition, message(text));
        },
        "KS2D");

    if(expect_translation_equivariance)
    {
        const std::array<scalar_type, 2> translation{0.23, -0.31};
        ks2d.state_codec().unpack(state, reference_spectrum);
        symmetry::fourier::apply_translation<typename operator_type::backend_type>(
            reference_spectrum,
            reference_wavevectors,
            translation,
            translated_spectrum
        );
        ks2d.state_codec().pack(translated_spectrum, state_epsilon);
        ks2d.F(state_epsilon, lambda, f_perturbed);

        ks2d.state_codec().unpack(f_state, residual_spectrum);
        symmetry::fourier::apply_translation<typename operator_type::backend_type>(
            residual_spectrum,
            reference_wavevectors,
            translation,
            translated_residual_spectrum
        );
        ks2d.state_codec().pack(translated_residual_spectrum, jacobian_direction);
        operations.get(f_perturbed, f_perturbed_host.data());
        operations.get(jacobian_direction, jacobian_host.data());
        const scalar_type equivariance_error = relative_error(f_perturbed_host, jacobian_host);
        std::ostringstream equivariance_message;
        equivariance_message << "translation equivariance, relative error="
                             << std::scientific << equivariance_error;
        result.check(
            equivariance_error < 2.0e-12,
            message(equivariance_message.str())
        );
    }

    ks2d.set_linearization_point(zero, lambda);
    ks2d.jacobian_u(direction, recovered);
    ks2d.preconditioner_jacobian_u(recovered);
    std::vector<scalar_type> recovered_host(state_size);
    operations.get(recovered, recovered_host.data());
    result.check(
        relative_error(recovered_host, direction_host) < 2.0e-12,
        message("diagonal preconditioner inverts the zero-state Jacobian")
    );

    ks2d.physical_solution(state, physical);
    std::vector<scalar_type> physical_host(nx*ny);
    operations.get(physical, physical_host.data(), nx*ny);
    scalar_type mean = 0.0;
    scalar_type maximum_inversion_sum = 0.0;
    for(std::size_t ix = 0; ix < nx; ++ix)
    {
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            mean += physical_host[ix*ny + iy];
            const std::size_t reflected_x = (nx - ix)%nx;
            const std::size_t reflected_y = (ny - iy)%ny;
            maximum_inversion_sum = std::max(
                maximum_inversion_sum,
                std::abs(physical_host[ix*ny + iy] + physical_host[reflected_x*ny + reflected_y])
            );
        }
    }
    mean /= static_cast<scalar_type>(nx*ny);
    result.check(std::abs(mean) < 2.0e-14, message("physical solution is mean zero"));
    result.check(
        expect_inversion_odd ? maximum_inversion_sum < 2.0e-13 : maximum_inversion_sum > 1.0e-5,
        message(expect_inversion_odd
            ? "physical solution remains inversion odd"
            : "full codec does not impose inversion-odd symmetry")
    );

    if(expect_inversion_odd)
    {
        ks2d.randomize_vector(seed_a, scalar_type(4.333), std::uint64_t(0));
        ks2d.randomize_vector(seed_a_repeat, scalar_type(4.333), std::uint64_t(0));
        ks2d.randomize_vector(seed_b, scalar_type(4.333), std::uint64_t(1));
        std::vector<scalar_type> seed_a_host(state_size);
        std::vector<scalar_type> seed_a_repeat_host(state_size);
        std::vector<scalar_type> seed_b_host(state_size);
        operations.get(seed_a, seed_a_host.data());
        operations.get(seed_a_repeat, seed_a_repeat_host.data());
        operations.get(seed_b, seed_b_host.data());
        result.check(
            relative_error(seed_a_host, seed_a_repeat_host) == scalar_type(0),
            message("deflation seed is deterministic")
        );
        scalar_type seed_dot = scalar_type(0);
        scalar_type seed_a_norm_squared = scalar_type(0);
        scalar_type seed_b_norm_squared = scalar_type(0);
        for(std::size_t index = 0; index < state_size; ++index)
        {
            seed_dot += seed_a_host[index]*seed_b_host[index];
            seed_a_norm_squared += seed_a_host[index]*seed_a_host[index];
            seed_b_norm_squared += seed_b_host[index]*seed_b_host[index];
        }
        const scalar_type cosine = seed_dot/
            std::sqrt(seed_a_norm_squared*seed_b_norm_squared);
        result.check(
            std::abs(cosine) < scalar_type(0.99),
            message("deflation seed profiles span distinct directions")
        );
        result.check(
            operations.norm_l2(seed_a) >
                scalar_type(0.1)*std::sqrt(static_cast<scalar_type>(nx*ny)),
            message("deflation seed uses physical Fourier scaling")
        );
        ks2d.F(seed_a, scalar_type(4.333), f_state);
        result.check(
            std::isfinite(operations.norm_l2(f_state)),
            message("deflation seed has a finite residual")
        );

        using finite_actions_type =
            symmetry::finite_action_registry<vector_operations_type>;
        using identity_adapter_type =
            deflation::identity_symmetry_adapter<vector_operations_type>;
        using quotient_adapter_type = symmetry::finite_quotient_adapter<
            vector_operations_type,
            identity_adapter_type>;
        finite_actions_type finite_actions(&operations);
        ks2d.configure_finite_symmetry_actions(finite_actions);
        result.check(
            finite_actions.size() == expected_finite_group_order,
            message("complete compatible finite symmetry group is registered")
        );
        result.check(
            !finite_actions.definition_fingerprint().empty(),
            message("finite symmetry definition has a fingerprint")
        );

        std::vector<scalar_type> transformed_host(state_size);
        std::vector<scalar_type> transformed_residual_host(state_size);
        std::vector<scalar_type> expected_residual_host(state_size);
        for(std::size_t action_index = 0;
            action_index < finite_actions.size();
            ++action_index)
        {
            finite_actions.apply(action_index, seed_a, state_epsilon);
            finite_actions.pullback(action_index, state_epsilon, recovered);
            operations.get(recovered, transformed_host.data());
            result.check(
                relative_error(transformed_host, seed_a_host) < 2.0e-14,
                message("finite action pullback applies the inverse")
            );

            ks2d.F(state_epsilon, scalar_type(4.333), f_perturbed);
            finite_actions.apply(action_index, f_state, jacobian_direction);
            operations.get(f_perturbed, transformed_residual_host.data());
            operations.get(jacobian_direction, expected_residual_host.data());
            result.check(
                relative_error(
                    transformed_residual_host,
                    expected_residual_host) < 2.0e-12,
                message("finite half-period action is equivariant")
            );
        }

        identity_adapter_type identity_adapter(&operations);
        quotient_adapter_type quotient_adapter(
            &operations,
            &identity_adapter,
            &finite_actions);
        deflation::symmetry_solution_storage<
            vector_operations_type,
            quotient_adapter_type> quotient_storage(
                &operations,
                8,
                operations.get_l2_size(),
                scalar_type(2),
                &quotient_adapter,
                1.0e-10);
        const int half_shift_x =
            finite_actions.find("inversion_odd_half_shift_x");
        result.check(
            half_shift_x >= 0,
            message("x half-period action is named and discoverable")
        );
        quotient_storage.push_back(seed_a);
        if(half_shift_x >= 0)
        {
            finite_actions.apply(
                static_cast<std::size_t>(half_shift_x),
                seed_a,
                state_epsilon);
            quotient_storage.push_back(state_epsilon);
            result.check(
                quotient_storage.get_size() == 1,
                message("finite quotient storage removes half-period duplicates")
            );
            result.check(
                quotient_storage.nearest_stabilized_distance(state_epsilon) < 1.0e-12,
                message("finite quotient duplicate distance is zero")
            );
        }

        const int axis_swap = finite_actions.find("axis_swap");
        result.check(
            axis_swap >= 0 ? nx == ny : nx != ny,
            message("axis swap is registered exactly for compatible square grids")
        );
        if(axis_swap >= 0)
        {
            ks2d.physical_solution(seed_a, physical);
            std::vector<scalar_type> original_physical(nx*ny);
            operations.get(physical, original_physical.data(), nx*ny);
            finite_actions.apply(
                static_cast<std::size_t>(axis_swap),
                seed_a,
                state_epsilon);
            ks2d.physical_solution(state_epsilon, physical);
            std::vector<scalar_type> swapped_physical(nx*ny);
            operations.get(physical, swapped_physical.data(), nx*ny);
            scalar_type maximum_swap_error = scalar_type(0);
            for(std::size_t ix = 0; ix < nx; ++ix)
            {
                for(std::size_t iy = 0; iy < ny; ++iy)
                {
                    maximum_swap_error = std::max(
                        maximum_swap_error,
                        std::abs(
                            swapped_physical[ix*ny + iy] -
                            original_physical[iy*ny + ix]));
                }
            }
            result.check(
                maximum_swap_error < scalar_type(2.0e-12),
                message("axis-swap action exchanges physical coordinates")
            );
            quotient_storage.push_back(state_epsilon);
            result.check(
                quotient_storage.get_size() == 1,
                message("finite quotient storage removes axis-swap duplicates")
            );

            using inversion_codec_type =
                discretization::fourier::codecs::inversion_odd_field_2d<
                    vector_operations_type,
                    typename operator_type::complex_type>;
            if constexpr(std::is_same<
                typename operator_type::state_codec_type,
                inversion_codec_type>::value)
            {
                using residual_translation_aligner_type =
                    symmetry::fourier::residual_translation_orbit_aligner_2d<
                        vector_operations_type>;
                using residual_quotient_adapter_type =
                    symmetry::finite_quotient_adapter<
                        vector_operations_type,
                        identity_adapter_type,
                        residual_translation_aligner_type>;
                residual_translation_aligner_type residual_translation_aligner(
                    &operations,
                    ks2d.state_modes());
                residual_quotient_adapter_type residual_quotient_adapter(
                    &operations,
                    &identity_adapter,
                    &finite_actions,
                    &residual_translation_aligner);
                deflation::symmetry_solution_storage<
                    vector_operations_type,
                    residual_quotient_adapter_type> residual_storage(
                        &operations,
                        8,
                        operations.get_l2_size(),
                        scalar_type(2),
                        &residual_quotient_adapter,
                        1.0e-10);

                std::vector<scalar_type> residual_source(
                    state_size,
                    scalar_type(0));
                const auto& state_modes = ks2d.state_modes();
                for(std::size_t index = 0; index < state_size; ++index)
                {
                    const int difference =
                        state_modes[index][0] - state_modes[index][1];
                    if(difference%2 == 0)
                    {
                        residual_source[index] =
                            scalar_type(0.02)*
                            std::sin(static_cast<scalar_type>(5*index + 1));
                    }
                }
                operations.set(residual_source.data(), seed_a);
                finite_actions.apply(
                    static_cast<std::size_t>(axis_swap),
                    seed_a,
                    state_epsilon);
                operations.get(state_epsilon, transformed_host.data());
                for(std::size_t index = 0; index < state_size; ++index)
                {
                    const int difference =
                        state_modes[index][0] - state_modes[index][1];
                    if(difference%2 == 0 && (difference/2)%2 != 0)
                    {
                        transformed_host[index] = -transformed_host[index];
                    }
                }
                operations.set(transformed_host.data(), state_epsilon);
                residual_storage.push_back(seed_a);
                residual_storage.push_back(state_epsilon);
                result.check(
                    residual_storage.get_size() == 1,
                    message(
                        "finite quotient storage removes axis-swap plus residual quarter-shift duplicates")
                );
                result.check(
                    residual_storage.nearest_stabilized_distance(
                        state_epsilon) < scalar_type(2.0e-12),
                    message(
                        "residual translation orbit distance is backend independent")
                );
                result.check(
                    residual_quotient_adapter.
                        symmetry_definition_fingerprint().find(
                            "fourier_residual_translation_lattice_2d_v1") !=
                        std::string::npos,
                    message(
                        "residual translation policy enters the archive definition")
                );
            }
        }
    }

    operations.stop_use_vectors(
        zero,
        state,
        direction,
        cotangent,
        state_epsilon,
        f_zero,
        f_state,
        linear_state,
        nonlinear_state,
        split_state,
        f_perturbed,
        jacobian_direction,
        alpha_direction,
        f_lambda_perturbed,
        recovered,
        physical,
        seed_a,
        seed_a_repeat,
        seed_b
    );
    operations.free_vectors(
        zero,
        state,
        direction,
        cotangent,
        state_epsilon,
        f_zero,
        f_state,
        linear_state,
        nonlinear_state,
        split_state,
        f_perturbed,
        jacobian_direction,
        alpha_direction,
        f_lambda_perturbed,
        recovered,
        physical,
        seed_a,
        seed_a_repeat,
        seed_b
    );
}

template<class Backend, class FFTBackend>
void run_full_square_symmetry_case(report& result)
{
    using scalar_type = double;
    using vector_operations_type =
        scfd_vector_operations<Backend, scalar_type>;
    using operator_type = nonlinear_operators::kuramoto_sivashinskiy_2d<
        vector_operations_type,
        FFTBackend,
        discretization::fourier::codecs::full_real_field_2d>;
    using vector_type = typename vector_operations_type::vector_type;
    using finite_actions_type =
        symmetry::finite_action_registry<vector_operations_type>;
    using identity_adapter_type =
        deflation::identity_symmetry_adapter<vector_operations_type>;
    using quotient_adapter_type = symmetry::finite_quotient_adapter<
        vector_operations_type,
        identity_adapter_type>;

    constexpr std::size_t n = 12;
    constexpr std::size_t state_size = n*n - 1;
    vector_operations_type operations(state_size);
    operator_type ks2d(2.0, 4.0, n, n, &operations);
    vector_type state;
    vector_type transformed;
    vector_type recovered;
    vector_type residual;
    vector_type transformed_residual;
    vector_type expected_residual;
    vector_type physical;
    operations.init_vectors(
        state,
        transformed,
        recovered,
        residual,
        transformed_residual,
        expected_residual,
        physical);
    operations.start_use_vectors(
        state,
        transformed,
        recovered,
        residual,
        transformed_residual,
        expected_residual);
    operations.start_use_vector(physical, n*n);

    std::vector<scalar_type> state_host(state_size);
    for(std::size_t index = 0; index < state_size; ++index)
    {
        state_host[index] =
            0.03*std::sin(static_cast<scalar_type>(3*index + 1))/
            static_cast<scalar_type>((index + 1)*(index + 1));
    }
    operations.set(state_host.data(), state);
    ks2d.F(state, 5.25, residual);

    finite_actions_type finite_actions(&operations);
    const auto group = ks2d.finite_symmetry_group();
    const auto action_workspace =
        ks2d.configure_finite_symmetry_actions(
            finite_actions,
            group);
    (void)action_workspace;
    result.check(
        group.size() == 2 && finite_actions.size() == 2,
        "full_square: physical axis-swap group has order two");
    const int axis_swap = finite_actions.find("axis_swap");
    result.check(
        axis_swap >= 0,
        "full_square: axis swap is registered");

    if(axis_swap >= 0)
    {
        const std::size_t action_index =
            static_cast<std::size_t>(axis_swap);
        finite_actions.apply(action_index, state, transformed);
        finite_actions.pullback(
            action_index,
            transformed,
            recovered);
        std::vector<scalar_type> recovered_host(state_size);
        operations.get(recovered, recovered_host.data());
        result.check(
            relative_error(recovered_host, state_host) < 2.0e-14,
            "full_square: axis-swap pullback recovers the full state");

        ks2d.F(transformed, 5.25, transformed_residual);
        finite_actions.apply(
            action_index,
            residual,
            expected_residual);
        std::vector<scalar_type> transformed_residual_host(state_size);
        std::vector<scalar_type> expected_residual_host(state_size);
        operations.get(
            transformed_residual,
            transformed_residual_host.data());
        operations.get(
            expected_residual,
            expected_residual_host.data());
        result.check(
            relative_error(
                transformed_residual_host,
                expected_residual_host) < 2.0e-12,
            "full_square: nonlinear operator is axis-swap equivariant");

        ks2d.physical_solution(state, physical);
        std::vector<scalar_type> original_physical(n*n);
        operations.get(physical, original_physical.data(), n*n);
        ks2d.physical_solution(transformed, physical);
        std::vector<scalar_type> swapped_physical(n*n);
        operations.get(physical, swapped_physical.data(), n*n);
        scalar_type maximum_swap_error = scalar_type(0);
        for(std::size_t ix = 0; ix < n; ++ix)
        {
            for(std::size_t iy = 0; iy < n; ++iy)
            {
                maximum_swap_error = std::max(
                    maximum_swap_error,
                    std::abs(
                        swapped_physical[ix*n + iy] -
                        original_physical[iy*n + ix]));
            }
        }
        result.check(
            maximum_swap_error < 2.0e-12,
            "full_square: physical coordinates are exchanged");

        identity_adapter_type identity_adapter(&operations);
        quotient_adapter_type quotient_adapter(
            &operations,
            &identity_adapter,
            &finite_actions);
        deflation::symmetry_solution_storage<
            vector_operations_type,
            quotient_adapter_type> storage(
                &operations,
                4,
                operations.get_l2_size(),
                scalar_type(2),
                &quotient_adapter,
                1.0e-10);
        storage.push_back(state);
        storage.push_back(transformed);
        result.check(
            storage.get_size() == 1,
            "full_square: quotient storage removes axis-swapped full states");
    }

    operations.stop_use_vector(physical);
    operations.stop_use_vectors(
        state,
        transformed,
        recovered,
        residual,
        transformed_residual,
        expected_residual);
    operations.free_vectors(
        state,
        transformed,
        recovered,
        residual,
        transformed_residual,
        expected_residual,
        physical);
}

template<class Backend, class FFTBackend>
int run(const std::string& backend_name)
{
    constexpr std::size_t nx = 12;
    constexpr std::size_t ny = 16;
    report result;
    run_case<
        Backend,
        FFTBackend,
        discretization::fourier::codecs::inversion_odd_field_2d
    >(result, "inversion_odd_rectangular", nx, ny, nx*ny/2 - 2, true, false, 4);
    constexpr std::size_t square_size = 12;
    run_case<
        Backend,
        FFTBackend,
        discretization::fourier::codecs::inversion_odd_field_2d
    >(
        result,
        "inversion_odd_square",
        square_size,
        square_size,
        square_size*square_size/2 - 2,
        true,
        false,
        8);
    run_case<
        Backend,
        FFTBackend,
        discretization::fourier::codecs::full_real_field_2d
    >(result, "full_grid_mean_zero", nx, ny, nx*ny - 1, false, false);
    run_case<
        Backend,
        FFTBackend,
        discretization::fourier::codecs::translation_equivariant_real_field_2d
    >(result, "full_translation_equivariant", nx, ny, (nx - 1)*(ny - 1) - 1, false, true);
    run_full_square_symmetry_case<Backend, FFTBackend>(result);
    return result.finish(backend_name);
}

} // namespace

int main()
{
    try
    {
#if defined(TEST_VECTOR_BACKEND_CUDA)
        common::init_cuda_from_scfd_selector("auto");
        return run<scfd::backend::cuda, external_libraries::fft::cufft_backend>("scfd_cuda_cufft");
#else
        return run<scfd::backend::omp, external_libraries::fft::fftw_backend>("scfd_omp_fftw");
#endif
    }
    catch(const std::exception& error)
    {
        std::cerr << "KS2D operator test failed: " << error.what() << std::endl;
        return 1;
    }
}
