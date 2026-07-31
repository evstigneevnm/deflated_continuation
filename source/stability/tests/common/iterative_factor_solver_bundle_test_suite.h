#ifndef __STABILITY_TESTS_COMMON_ITERATIVE_FACTOR_SOLVER_BUNDLE_TEST_SUITE_H__
#define __STABILITY_TESTS_COMMON_ITERATIVE_FACTOR_SOLVER_BUNDLE_TEST_SUITE_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/utils/log.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <stability/eigensolvers/transformations/complex_affine_block_operator.h>
#include <stability/eigensolvers/transformations/complex_affine_preconditioner.h>
#include <stability/eigensolvers/transformations/iterative_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/stability_polynomial_factorization.h>

#include "analytical_dense_operator.h"
#include "analytical_eigenproblem.h"

namespace stability
{
namespace tests
{
namespace iterative_factor_solver_bundle_test
{

inline std::size_t checks = 0;
inline std::size_t failures = 0;

inline void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

template<class Function>
void require_throws(Function&& function, const std::string& message)
{
    bool threw = false;
    try
    {
        function();
    }
    catch(const std::exception&)
    {
        threw = true;
    }
    require(threw, message);
}

inline bool close(
    const std::complex<double>& actual,
    const std::complex<double>& expected,
    double tolerance = 2.0e-9)
{
    return std::abs(actual - expected) <=
        tolerance *
        std::max({1.0, std::abs(actual), std::abs(expected)});
}

template<class ProductSpace, class RealOperator>
class controlled_affine_operator
{
public:
    using vector_type = typename ProductSpace::vector_type;
    using factor_type =
        stability::eigensolvers::transformations::
            complex_affine_factor<
                typename ProductSpace::norm_type>;
    using implementation_type =
        stability::eigensolvers::transformations::
            complex_affine_block_operator<
                ProductSpace,
                RealOperator>;

    controlled_affine_operator(
        const ProductSpace& vector_space,
        const RealOperator& real_operator,
        const factor_type& factor,
        bool force_failure)
        : implementation_(vector_space, real_operator, factor),
          force_failure_(force_failure)
    {
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        ++operator_calls_;
        if(force_failure_)
        {
            ++forced_failures_;
            return false;
        }
        return implementation_.apply(source, destination);
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    std::size_t forced_failures() const
    {
        return forced_failures_;
    }

private:
    implementation_type implementation_;
    bool force_failure_;
    mutable std::size_t operator_calls_ = 0;
    mutable std::size_t forced_failures_ = 0;
};

template<class VectorSpace, class LinearOperator>
class counting_identity_preconditioner
{
public:
    using vector_type = typename VectorSpace::vector_type;
    using operator_type = LinearOperator;

    explicit counting_identity_preconditioner(
        std::shared_ptr<VectorSpace> vector_space)
        : vector_space_(std::move(vector_space))
    {
        if(!vector_space_)
            throw std::invalid_argument(
                "counting identity preconditioner requires a vector space");
    }

    void set_operator(std::shared_ptr<const operator_type>)
    {
    }

    bool apply(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++apply_calls_;
        vector_space_->assign(right_hand_side, solution);
        return true;
    }

    bool apply(vector_type&) const
    {
        ++apply_calls_;
        return true;
    }

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        return apply(right_hand_side, solution);
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

private:
    std::shared_ptr<VectorSpace> vector_space_;
    mutable std::size_t apply_calls_ = 0;
};

template<class ComponentSpace>
class analytical_affine_inverse_model
{
public:
    using scalar_type = typename ComponentSpace::scalar_type;
    using vector_type = typename ComponentSpace::vector_type;

    analytical_affine_inverse_model(
        const ComponentSpace& vector_space,
        std::vector<scalar_type> diagonal)
        : vector_space_(vector_space),
          diagonal_(std::move(diagonal))
    {
        if(diagonal_.size() != vector_space_.get_default_size())
            throw std::invalid_argument(
                "analytical affine inverse diagonal size mismatch");
    }

    bool preconditioner_jacobian_affine_u(
        vector_type& right_hand_side_to_solution,
        scalar_type jacobian_scale,
        scalar_type identity_shift) const
    {
        ++apply_calls_;
        if(apply_calls_ > fail_after_)
            return false;

        const std::size_t size = diagonal_.size();
        std::vector<scalar_type> host(size);
        vector_space_.get(
            right_hand_side_to_solution,
            host.data(),
            size);
        for(std::size_t index = 0; index < size; ++index)
        {
            const scalar_type denominator =
                jacobian_scale*diagonal_[index] +
                identity_shift;
            if(
                !(std::abs(denominator) >
                  std::numeric_limits<scalar_type>::min()))
            {
                return false;
            }
            host[index] /= denominator;
        }
        vector_space_.set(
            host.data(),
            right_hand_side_to_solution,
            size);
        return true;
    }

    void fail_after(std::size_t successful_applications)
    {
        fail_after_ = successful_applications;
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

private:
    const ComponentSpace& vector_space_;
    std::vector<scalar_type> diagonal_;
    mutable std::size_t apply_calls_ = 0;
    std::size_t fail_after_ =
        std::numeric_limits<std::size_t>::max();
};

template<class ProductSpace>
void set_complex_vector(
    const ProductSpace& vector_space,
    const std::vector<std::complex<double>>& host,
    typename ProductSpace::vector_type& destination)
{
    std::vector<double> packed(2 * host.size());
    for(std::size_t index = 0; index < host.size(); ++index)
    {
        packed[index] = host[index].real();
        packed[host.size() + index] = host[index].imag();
    }
    vector_space.set(packed.data(), destination, packed.size());
}

template<class ProductSpace>
std::vector<std::complex<double>> get_complex_vector(
    const ProductSpace& vector_space,
    const typename ProductSpace::vector_type& source)
{
    const std::size_t size = vector_space.first_size();
    std::vector<double> packed(2 * size);
    vector_space.get(source, packed.data(), packed.size());
    std::vector<std::complex<double>> result(size);
    for(std::size_t index = 0; index < size; ++index)
    {
        result[index] =
            std::complex<double>(
                packed[index],
                packed[size + index]);
    }
    return result;
}

template<class Solver>
typename Solver::params solver_parameters()
{
    typename Solver::params parameters;
    parameters.basis_size = 8;
    parameters.batch_size = 8;
    parameters.preconditioner_side = 'L';
    parameters.orthogonalization = "mgs";
    parameters.reorthogonalization_policy = "dgks";
    parameters.max_orthogonalization_passes = 2;
    parameters.monitor.rel_tol = 1.0e-12;
    parameters.monitor.abs_tol = 1.0e-13;
    parameters.monitor.max_iters_num = 24;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    return parameters;
}

template<class Bundle, class ProductSpace, class RealOperator>
std::unique_ptr<Bundle> make_bundle(
    const std::shared_ptr<ProductSpace>& vector_space,
    const RealOperator& real_operator,
    const std::vector<typename Bundle::factor_type>& factors,
    std::size_t failed_factor = std::numeric_limits<std::size_t>::max())
{
    using components_type = typename Bundle::components_type;
    using operator_type = typename Bundle::operator_type;
    using preconditioner_type = typename Bundle::preconditioner_type;

    auto factory =
        [&vector_space, &real_operator, failed_factor](
            const typename Bundle::factor_type& factor,
            std::size_t index)
        {
            return components_type{
                std::make_shared<operator_type>(
                    *vector_space,
                    real_operator,
                    factor,
                    index == failed_factor),
                std::make_shared<preconditioner_type>(
                    vector_space)};
        };
    return std::make_unique<Bundle>(
        vector_space,
        factors,
        solver_parameters<typename Bundle::solver_type>(),
        factory);
}

inline std::vector<std::complex<double>> linear_combination(
    std::complex<double> first_scale,
    const std::vector<std::complex<double>>& first,
    std::complex<double> second_scale,
    const std::vector<std::complex<double>>& second)
{
    std::vector<std::complex<double>> result(first.size());
    for(std::size_t index = 0; index < result.size(); ++index)
    {
        result[index] =
            first_scale * first[index] +
            second_scale * second[index];
    }
    return result;
}

template<class Bundle>
std::vector<std::complex<double>> expected_solution(
    const std::vector<double>& diagonal,
    const std::vector<std::complex<double>>& right_hand_side,
    const std::vector<typename Bundle::factor_type>& factors)
{
    std::vector<std::complex<double>> result(diagonal.size());
    for(std::size_t index = 0; index < diagonal.size(); ++index)
    {
        const auto denominator =
            stability::eigensolvers::transformations::
                evaluate_affine_factors(
                    factors,
                    std::complex<double>(diagonal[index], 0.0));
        result[index] = right_hand_side[index] / denominator;
    }
    return result;
}

template<class Backend>
void run_backend(const std::string& label)
{
    using component_space_type =
        scfd_vector_operations<Backend, double>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<
            component_space_type>;
    using real_operator_type =
        stability::tests::analytical_dense_operator<
            component_space_type,
            double>;
    using factor_operator_type =
        controlled_affine_operator<
            product_space_type,
            real_operator_type>;
    using preconditioner_type =
        counting_identity_preconditioner<
            product_space_type,
            factor_operator_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<
            product_space_type,
            log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            product_space_type,
            monitor_type,
            log_type,
            factor_operator_type,
            preconditioner_type>;
    using bundle_type =
        stability::eigensolvers::transformations::
            iterative_factor_solver_bundle<
                product_space_type,
                factor_operator_type,
                preconditioner_type,
                solver_type>;
    using affine_model_type =
        analytical_affine_inverse_model<
            component_space_type>;
    using affine_provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                component_space_type,
                affine_model_type>;
    using affine_preconditioner_type =
        stability::eigensolvers::transformations::
            complex_affine_preconditioner<
                product_space_type,
                factor_operator_type,
                affine_provider_type>;
    using affine_solver_type =
        nmfd::solvers::gmres<
            product_space_type,
            monitor_type,
            log_type,
            factor_operator_type,
            affine_preconditioner_type>;
    using affine_bundle_type =
        stability::eigensolvers::transformations::
            iterative_factor_solver_bundle<
                product_space_type,
                factor_operator_type,
                affine_preconditioner_type,
                affine_solver_type>;
    using vector_wrap_type =
        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true>;

    const auto problem =
        stability::tests::diagonal_eigenproblem<double>();
    const std::vector<double> diagonal{-4.0, -1.0, 2.0, 7.0};
    auto component_space =
        std::make_shared<component_space_type>(problem.dimension());
    auto product_space =
        std::make_shared<product_space_type>(
            *component_space,
            *component_space);
    real_operator_type real_operator(*component_space, problem);

    constexpr double step = 0.075;
    constexpr std::size_t repetitions = 3;
    const std::complex<double> shift =
        std::polar(1.02, 0.43);
    const auto factors =
        stability::eigensolvers::transformations::
            euler_denominator_factors(
                step,
                repetitions,
                shift);

    auto bundle = make_bundle<bundle_type>(
        product_space,
        real_operator,
        factors);
    require(
        bundle->factor_count() == factors.size(),
        label + " factor count");
    for(std::size_t index = 0; index < factors.size(); ++index)
    {
        require(
            bundle->factor(index).descriptor().operator_scale ==
                factors[index].operator_scale &&
            bundle->factor(index).descriptor().diagonal_shift ==
                factors[index].diagonal_shift,
            label + " descriptor ownership");
        require(
            bundle->factor(index).solver_parameters().basis_size == 8 &&
            bundle->factor(index).solver_parameters().batch_size == 8,
            label + " uniform solver parameters");
    }
    require_throws(
        [&]
        {
            (void)bundle->factor(factors.size());
        },
        label + " factor bounds check");

    vector_wrap_type right_first(*product_space);
    vector_wrap_type right_second(*product_space);
    vector_wrap_type right_combined(*product_space);
    vector_wrap_type solution_first(*product_space);
    vector_wrap_type solution_first_repeat(*product_space);
    vector_wrap_type solution_second(*product_space);
    vector_wrap_type solution_combined(*product_space);

    const std::vector<std::complex<double>> host_first{
        {1.0, -0.25},
        {-0.3, 0.8},
        {2.0, 1.5},
        {-1.2, -0.4}};
    const std::vector<std::complex<double>> host_second{
        {-0.5, 0.6},
        {1.1, -0.2},
        {0.4, -1.3},
        {2.2, 0.1}};
    const std::complex<double> first_scale(0.7, -0.2);
    const std::complex<double> second_scale(-0.3, 0.4);
    const auto host_combined =
        linear_combination(
            first_scale,
            host_first,
            second_scale,
            host_second);
    set_complex_vector(*product_space, host_first, *right_first);
    set_complex_vector(*product_space, host_second, *right_second);
    set_complex_vector(*product_space, host_combined, *right_combined);

    product_space->assign_scalar(17.0, *solution_first);
    require(
        bundle->solve(*right_first, *solution_first),
        label + " first solve");
    const auto actual_first =
        get_complex_vector(*product_space, *solution_first);
    const auto expected_first =
        expected_solution<bundle_type>(
            diagonal,
            host_first,
            factors);
    for(std::size_t index = 0; index < diagonal.size(); ++index)
    {
        require(
            close(actual_first[index], expected_first[index]),
            label + " first solve value");
    }

    product_space->assign_scalar(-23.0, *solution_first_repeat);
    require(
        bundle->solve(*right_first, *solution_first_repeat),
        label + " repeated solve");
    const auto repeated_first =
        get_complex_vector(*product_space, *solution_first_repeat);
    for(std::size_t index = 0; index < diagonal.size(); ++index)
    {
        require(
            close(repeated_first[index], actual_first[index]),
            label + " dirty output does not affect repeated solve");
    }

    require(
        bundle->solve(*right_second, *solution_second),
        label + " second solve");
    require(
        bundle->solve(*right_combined, *solution_combined),
        label + " combined solve");
    const auto actual_second =
        get_complex_vector(*product_space, *solution_second);
    const auto actual_combined =
        get_complex_vector(*product_space, *solution_combined);
    const auto linearized =
        linear_combination(
            first_scale,
            actual_first,
            second_scale,
            actual_second);
    for(std::size_t index = 0; index < diagonal.size(); ++index)
    {
        require(
            close(actual_combined[index], linearized[index], 8.0e-9),
            label + " factorized inverse linearity");
    }

    const auto success_statistics = bundle->statistics();
    require(
        success_statistics.solve_calls == 4 &&
        success_statistics.factor_solve_calls ==
            4 * factors.size() &&
        success_statistics.failed_solves == 0 &&
        success_statistics.last_failed_factor ==
            bundle_type::inverse_type::no_factor_index &&
        success_statistics.completed_factors_last_solve ==
            factors.size(),
        label + " aggregate success statistics");
    require(
        success_statistics.factors.size() == factors.size(),
        label + " per-factor statistics count");
    for(std::size_t index = 0; index < factors.size(); ++index)
    {
        const auto& statistics =
            success_statistics.factors[index];
        require(
            statistics.index == index &&
            statistics.solve_calls == 4 &&
            statistics.failed_solves == 0 &&
            statistics.total_iterations > 0 &&
            statistics.maximum_iterations > 0 &&
            statistics.last_residual_available,
            label + " per-factor success statistics");
        require(
            bundle->factor(index).linear_operator().operator_calls() > 0,
            label + " operator state access");
        require(
            bundle->factor(index).preconditioner().apply_calls() > 0,
            label + " preconditioner state access");
    }

    bundle->reset_statistics();
    const auto reset_statistics = bundle->statistics();
    require(
        reset_statistics.solve_calls == 0 &&
        reset_statistics.factor_solve_calls == 0 &&
        reset_statistics.failed_solves == 0 &&
        reset_statistics.last_failed_factor ==
            bundle_type::inverse_type::no_factor_index &&
        reset_statistics.completed_factors_last_solve == 0,
        label + " aggregate statistics reset");
    for(const auto& statistics : reset_statistics.factors)
    {
        require(
            statistics.solve_calls == 0 &&
            statistics.failed_solves == 0 &&
            statistics.total_iterations == 0 &&
            !statistics.last_residual_available,
            label + " per-factor statistics reset");
    }

    auto affine_model = std::make_shared<affine_model_type>(
        *component_space,
        diagonal);
    auto affine_provider =
        std::make_shared<affine_provider_type>(
            *component_space,
            *affine_model);
    using affine_components_type =
        typename affine_bundle_type::components_type;
    auto affine_factory =
        [&product_space, &real_operator, &affine_provider](
            const typename affine_bundle_type::factor_type& factor,
            std::size_t)
        {
            return affine_components_type{
                std::make_shared<factor_operator_type>(
                    *product_space,
                    real_operator,
                    factor,
                    false),
                std::make_shared<affine_preconditioner_type>(
                    *product_space,
                    affine_provider,
                    factor)};
        };
    auto factor_parameters =
        [](
            const typename affine_bundle_type::
                solver_parameters_type& defaults,
            const typename affine_bundle_type::factor_type&,
            std::size_t index)
        {
            auto parameters = defaults;
            parameters.basis_size =
                static_cast<unsigned>(8 + index);
            parameters.batch_size =
                std::min<unsigned>(
                    parameters.basis_size,
                    3);
            return parameters;
        };
    affine_bundle_type affine_bundle(
        product_space,
        factors,
        solver_parameters<affine_solver_type>(),
        affine_factory,
        factor_parameters);
    product_space->assign_scalar(31.0, *solution_first_repeat);
    require(
        affine_bundle.solve(
            *right_first,
            *solution_first_repeat),
        label + " model affine-preconditioned solve");
    const auto affine_actual =
        get_complex_vector(
            *product_space,
            *solution_first_repeat);
    for(std::size_t index = 0; index < diagonal.size(); ++index)
    {
        require(
            close(affine_actual[index], expected_first[index]),
            label + " model affine-preconditioned value");
    }

    const auto affine_statistics = affine_bundle.statistics();
    for(std::size_t index = 0; index < factors.size(); ++index)
    {
        const auto& factor_state = affine_bundle.factor(index);
        require(
            factor_state.solver_parameters().basis_size ==
                static_cast<unsigned>(8 + index) &&
            factor_state.solver_parameters().batch_size == 3,
            label + " factor-specific solver parameters");
        require(
            factor_state.preconditioner().factor().
                operator_scale ==
                factors[index].operator_scale &&
            factor_state.preconditioner().factor().
                diagonal_shift ==
                factors[index].diagonal_shift,
            label + " affine preconditioner descriptor");
        require(
            factor_state.preconditioner().apply_calls() > 0 &&
            factor_state.preconditioner().
                failed_applications() == 0,
            label + " affine preconditioner statistics");
    }
    require(
        affine_provider->apply_calls() > 0 &&
        affine_provider->failed_applications() == 0 &&
        affine_model->apply_calls() ==
            affine_provider->apply_calls(),
        label + " nonlinear-operator provider path");

    const std::vector<typename affine_bundle_type::factor_type>
        real_factors{
            {
                std::complex<double>(0.5, 0.0),
                std::complex<double>(1.25, 0.0)
            }};
    auto identity_real_bundle = make_bundle<bundle_type>(
        product_space,
        real_operator,
        real_factors);
    require(
        identity_real_bundle->solve(
            *right_first,
            *solution_first),
        label + " real factor identity-preconditioned solve");
    affine_bundle_type affine_real_bundle(
        product_space,
        real_factors,
        solver_parameters<affine_solver_type>(),
        affine_factory,
        factor_parameters);
    require(
        affine_real_bundle.solve(
            *right_first,
            *solution_first_repeat),
        label + " real factor affine-preconditioned solve");
    const auto identity_real_statistics =
        identity_real_bundle->statistics();
    const auto affine_real_statistics =
        affine_real_bundle.statistics();
    require(
        affine_real_statistics.factors[0].total_iterations <
            identity_real_statistics.factors[0].total_iterations,
        label +
            " exact real affine preconditioner reduces GMRES iterations");

    auto failing_affine_model =
        std::make_shared<affine_model_type>(
            *component_space,
            diagonal);
    failing_affine_model->fail_after(0);
    auto failing_affine_provider =
        std::make_shared<affine_provider_type>(
            *component_space,
            *failing_affine_model);
    auto failing_affine_factory =
        [
            &product_space,
            &real_operator,
            &failing_affine_provider
        ](
            const typename affine_bundle_type::factor_type& factor,
            std::size_t)
        {
            return affine_components_type{
                std::make_shared<factor_operator_type>(
                    *product_space,
                    real_operator,
                    factor,
                    false),
                std::make_shared<affine_preconditioner_type>(
                    *product_space,
                    failing_affine_provider,
                    factor)};
        };
    affine_bundle_type failing_affine_bundle(
        product_space,
        factors,
        solver_parameters<affine_solver_type>(),
        failing_affine_factory);
    const std::vector<std::complex<double>>
        affine_failure_sentinel(
            diagonal.size(),
            std::complex<double>(-13.0, 19.0));
    set_complex_vector(
        *product_space,
        affine_failure_sentinel,
        *solution_first_repeat);
    require(
        !failing_affine_bundle.solve(
            *right_first,
            *solution_first_repeat),
        label + " affine provider failure propagates");
    const auto after_affine_failure =
        get_complex_vector(
            *product_space,
            *solution_first_repeat);
    for(std::size_t index = 0;
        index < affine_failure_sentinel.size();
        ++index)
    {
        require(
            close(
                after_affine_failure[index],
                affine_failure_sentinel[index]),
            label + " affine provider failure is transactional");
    }
    require(
        failing_affine_bundle.last_failed_factor() == 0 &&
        failing_affine_provider->failed_applications() > 0 &&
        failing_affine_bundle.factor(0).
            preconditioner().failed_applications() > 0,
        label + " affine provider failure statistics");

    const std::vector<std::complex<double>> sentinel(
        diagonal.size(),
        std::complex<double>(7.0, -9.0));
    for(std::size_t failed_factor = 0;
        failed_factor < factors.size();
        ++failed_factor)
    {
        auto failing_bundle = make_bundle<bundle_type>(
            product_space,
            real_operator,
            factors,
            failed_factor);
        set_complex_vector(
            *product_space,
            sentinel,
            *solution_combined);
        require(
            !failing_bundle->solve(
                *right_first,
                *solution_combined),
            label + " forced factor failure");
        require(
            failing_bundle->last_failed_factor() == failed_factor &&
            failing_bundle->completed_factors_last_solve() ==
                failed_factor &&
            failing_bundle->factor_solve_calls() ==
                failed_factor + 1 &&
            failing_bundle->failed_solves() == 1,
            label + " forced failure location");

        const auto after_failure =
            get_complex_vector(*product_space, *solution_combined);
        for(std::size_t index = 0; index < sentinel.size(); ++index)
        {
            require(
                close(after_failure[index], sentinel[index]),
                label + " failed solve keeps output transactional");
        }

        const auto failure_statistics =
            failing_bundle->statistics();
        for(std::size_t index = 0; index < factors.size(); ++index)
        {
            const std::size_t expected_calls =
                index <= failed_factor ? 1 : 0;
            const std::size_t expected_failures =
                index == failed_factor ? 1 : 0;
            require(
                failure_statistics.factors[index].solve_calls ==
                    expected_calls &&
                failure_statistics.factors[index].failed_solves ==
                    expected_failures,
                label + " per-factor failure statistics");
        }
        require(
            failing_bundle
                ->factor(failed_factor)
                .linear_operator()
                .forced_failures() > 0,
            label + " forced operator failure reached");
    }

    using components_type = typename bundle_type::components_type;
    auto null_operator_factory =
        [product_space](
            const typename bundle_type::factor_type&,
            std::size_t)
        {
            return components_type{
                nullptr,
                std::make_shared<preconditioner_type>(
                    product_space)};
        };
    require_throws(
        [&]
        {
            bundle_type invalid(
                product_space,
                factors,
                solver_parameters<solver_type>(),
                null_operator_factory);
        },
        label + " rejects null factor operator");

    auto valid_factory =
        [product_space, &real_operator](
            const typename bundle_type::factor_type& factor,
            std::size_t)
        {
            return components_type{
                std::make_shared<factor_operator_type>(
                    *product_space,
                    real_operator,
                    factor,
                    false),
                std::make_shared<preconditioner_type>(
                    product_space)};
        };
    require_throws(
        [&]
        {
            bundle_type invalid(
                product_space,
                {},
                solver_parameters<solver_type>(),
                valid_factory);
        },
        label + " rejects empty factors");
}

inline int finish()
{
    std::cout << "Checks: " << checks
              << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        std::cout << "FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASSED\n";
    return EXIT_SUCCESS;
}

} // namespace iterative_factor_solver_bundle_test
} // namespace tests
} // namespace stability

#endif
