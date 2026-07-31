#ifndef __STABILITY_TESTS_COMMON_MATRIX_FREE_COMPLEXIFICATION_TEST_SUITE_H__
#define __STABILITY_TESTS_COMMON_MATRIX_FREE_COMPLEXIFICATION_TEST_SUITE_H__

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
#include <type_traits>
#include <utility>
#include <vector>

#include <scfd/utils/log.h>

#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_vector_operations.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/scfd_complex_vector_bridge.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <stability/eigensolvers/transformations/affine_pencil_operator.h>
#include <stability/eigensolvers/transformations/complex_affine_factor.h>
#include <stability/eigensolvers/transformations/complexified_real_affine_preconditioner.h>
#include <stability/eigensolvers/transformations/complexified_real_operator.h>
#include <stability/eigensolvers/transformations/identity_operator.h>
#include <stability/eigensolvers/transformations/iterative_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>

#include "analytical_dense_operator.h"
#include "analytical_eigenproblem.h"
#include "analytical_real_affine_inverse_model.h"

namespace stability
{
namespace tests
{
namespace matrix_free_complexification_test
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

template<class Real>
bool close(
    const std::complex<Real>& actual,
    const std::complex<Real>& expected,
    Real tolerance)
{
    return std::abs(actual - expected) <=
        tolerance *
        std::max({Real(1), std::abs(actual), std::abs(expected)});
}

template<class Backend, class Real>
using backend_complex_type =
    common::scfd_backend_ext::complex_t<Backend, Real>;

template<class ComplexSpace, class Real>
void set_complex_vector(
    const ComplexSpace& vector_space,
    const std::vector<std::complex<Real>>& host,
    typename ComplexSpace::vector_type& destination)
{
    using complex_type = typename ComplexSpace::scalar_type;
    using traits =
        common::scfd_backend_ext::complex_value_traits<complex_type>;
    std::vector<complex_type> converted(host.size());
    for(std::size_t index = 0; index < host.size(); ++index)
    {
        converted[index] = traits::make(
            host[index].real(),
            host[index].imag());
    }
    vector_space.set(
        converted.data(),
        destination,
        converted.size());
}

template<class ComplexSpace>
std::vector<std::complex<typename ComplexSpace::norm_type>>
get_complex_vector(
    const ComplexSpace& vector_space,
    const typename ComplexSpace::vector_type& source)
{
    using real_type = typename ComplexSpace::norm_type;
    using complex_type = typename ComplexSpace::scalar_type;
    using traits =
        common::scfd_backend_ext::complex_value_traits<complex_type>;
    std::vector<complex_type> converted(
        vector_space.get_default_size());
    vector_space.get(
        source,
        converted.data(),
        converted.size());
    std::vector<std::complex<real_type>> result(converted.size());
    for(std::size_t index = 0; index < converted.size(); ++index)
    {
        result[index] = std::complex<real_type>(
            traits::real(converted[index]),
            traits::imag(converted[index]));
    }
    return result;
}

template<class Real>
std::vector<std::complex<Real>> solve_dense(
    std::vector<std::complex<Real>> matrix,
    std::vector<std::complex<Real>> right_hand_side)
{
    const std::size_t dimension = right_hand_side.size();
    if(matrix.size() != dimension*dimension)
        throw std::invalid_argument("dense solve matrix size mismatch");

    const Real tolerance =
        Real(128)*std::numeric_limits<Real>::epsilon();
    for(std::size_t pivot = 0; pivot < dimension; ++pivot)
    {
        std::size_t selected = pivot;
        for(std::size_t row = pivot + 1; row < dimension; ++row)
        {
            if(
                std::abs(matrix[row*dimension + pivot]) >
                std::abs(matrix[selected*dimension + pivot]))
            {
                selected = row;
            }
        }
        if(!(std::abs(matrix[selected*dimension + pivot]) > tolerance))
            throw std::runtime_error("dense solve is singular");
        if(selected != pivot)
        {
            for(std::size_t column = pivot;
                column < dimension;
                ++column)
            {
                std::swap(
                    matrix[pivot*dimension + column],
                    matrix[selected*dimension + column]);
            }
            std::swap(
                right_hand_side[pivot],
                right_hand_side[selected]);
        }

        for(std::size_t row = pivot + 1;
            row < dimension;
            ++row)
        {
            const auto multiplier =
                matrix[row*dimension + pivot] /
                matrix[pivot*dimension + pivot];
            matrix[row*dimension + pivot] = {};
            for(std::size_t column = pivot + 1;
                column < dimension;
                ++column)
            {
                matrix[row*dimension + column] -=
                    multiplier *
                    matrix[pivot*dimension + column];
            }
            right_hand_side[row] -=
                multiplier*right_hand_side[pivot];
        }
    }

    for(std::size_t row = dimension; row-- > 0;)
    {
        for(std::size_t column = row + 1;
            column < dimension;
            ++column)
        {
            right_hand_side[row] -=
                matrix[row*dimension + column] *
                right_hand_side[column];
        }
        right_hand_side[row] /= matrix[row*dimension + row];
    }
    return right_hand_side;
}

template<class Real>
std::vector<std::complex<Real>> expected_factorized_solution(
    const analytical_eigenproblem<Real>& problem,
    const std::vector<
        stability::eigensolvers::transformations::
            complex_affine_factor<Real>>& factors,
    std::vector<std::complex<Real>> right_hand_side)
{
    const std::size_t dimension = problem.dimension();
    for(const auto& factor : factors)
    {
        std::vector<std::complex<Real>> matrix(
            dimension*dimension);
        for(std::size_t row = 0; row < dimension; ++row)
        {
            for(std::size_t column = 0;
                column < dimension;
                ++column)
            {
                matrix[row*dimension + column] =
                    factor.operator_scale *
                    problem.matrix(row, column);
                if(row == column)
                {
                    matrix[row*dimension + column] +=
                        factor.diagonal_shift;
                }
            }
        }
        right_hand_side =
            solve_dense(std::move(matrix), std::move(right_hand_side));
    }
    return right_hand_side;
}

template<class VectorSpace, class LinearOperator>
class identity_preconditioner
{
public:
    using vector_type = typename VectorSpace::vector_type;
    using operator_type = LinearOperator;

    explicit identity_preconditioner(
        std::shared_ptr<VectorSpace> vector_space)
        : vector_space_(std::move(vector_space))
    {
    }

    void set_operator(std::shared_ptr<const operator_type>)
    {
    }

    bool apply(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        vector_space_->assign(right_hand_side, solution);
        return true;
    }

    bool apply(vector_type&) const
    {
        return true;
    }

private:
    std::shared_ptr<VectorSpace> vector_space_;
};

template<class Backend, class Real>
void test_operator(const std::string& label)
{
    using real_space_type =
        scfd_vector_operations<Backend, Real>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<real_space_type>;
    using complex_type = backend_complex_type<Backend, Real>;
    using complex_space_type =
        scfd_vector_operations<Backend, complex_type>;
    using bridge_type =
        nmfd::operations::scfd_complex_vector_bridge<
            product_space_type,
            complex_space_type>;
    using real_operator_type =
        analytical_dense_operator<real_space_type, Real>;
    using operator_type =
        stability::eigensolvers::transformations::
            complexified_real_operator<
                product_space_type,
                complex_space_type,
                bridge_type,
                real_operator_type>;

    const auto problem = nonnormal_eigenproblem<Real>();
    real_space_type real_space(problem.dimension());
    product_space_type product_space(real_space, real_space);
    complex_space_type complex_space(problem.dimension());
    bridge_type bridge(product_space, complex_space);
    real_operator_type real_operator(real_space, problem);
    operator_type complexified(product_space, bridge, real_operator);

    nmfd::detail::vector_wrap<complex_space_type, true, true> source(
        complex_space);
    nmfd::detail::vector_wrap<complex_space_type, true, true> destination(
        complex_space);
    const std::vector<std::complex<Real>> input{
        {Real(1), Real(0.5)},
        {Real(-2), Real(0.25)},
        {Real(0.75), Real(-1.5)}};
    set_complex_vector(complex_space, input, *source);
    require(
        complexified.apply(*source, *destination),
        label + " complexified operator apply");
    const auto actual = get_complex_vector(
        complex_space,
        *destination);
    const auto expected = problem.apply(input);
    const Real tolerance =
        std::is_same<Real, float>::value ?
            Real(4.0e-5) :
            Real(4.0e-12);
    for(std::size_t index = 0; index < actual.size(); ++index)
    {
        require(
            close(actual[index], expected[index], tolerance),
            label + " complexified operator value");
    }
    require(
        complexified.operator_calls() == 1 &&
        complexified.component_operator_calls() == 2 &&
        complexified.component_operator_failures() == 0 &&
        real_operator.operator_calls() == 2,
        label + " complexified operator accounting");

    real_operator_type failing_real_operator(real_space, problem);
    failing_real_operator.fail_after(1);
    operator_type failing(
        product_space,
        bridge,
        failing_real_operator);
    const std::vector<std::complex<Real>> sentinel(
        problem.dimension(),
        {Real(7), Real(-9)});
    set_complex_vector(complex_space, sentinel, *destination);
    require(
        !failing.apply(*source, *destination),
        label + " real component failure propagates");
    const auto after_failure = get_complex_vector(
        complex_space,
        *destination);
    for(std::size_t index = 0; index < sentinel.size(); ++index)
    {
        require(
            close(after_failure[index], sentinel[index], tolerance),
            label + " component failure is transactional");
    }
}

template<class Backend>
void test_true_complex_gmres(const std::string& label)
{
    using real_type = double;
    using real_space_type =
        scfd_vector_operations<Backend, real_type>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<real_space_type>;
    using complex_type = backend_complex_type<Backend, real_type>;
    using complex_traits =
        common::scfd_backend_ext::complex_value_traits<complex_type>;
    using complex_space_type =
        scfd_vector_operations<Backend, complex_type>;
    using bridge_type =
        nmfd::operations::scfd_complex_vector_bridge<
            product_space_type,
            complex_space_type>;
    using real_operator_type =
        analytical_dense_operator<real_space_type, real_type>;
    using complexified_operator_type =
        stability::eigensolvers::transformations::
            complexified_real_operator<
                product_space_type,
                complex_space_type,
                bridge_type,
                real_operator_type>;
    using identity_type =
        stability::eigensolvers::transformations::
            identity_operator<complex_space_type>;
    using factor_operator_type =
        stability::eigensolvers::transformations::
            affine_pencil_operator<
                complex_space_type,
                complexified_operator_type,
                identity_type>;
    using preconditioner_type =
        identity_preconditioner<
            complex_space_type,
            factor_operator_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<
            complex_space_type,
            log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            complex_space_type,
            monitor_type,
            log_type,
            factor_operator_type,
            preconditioner_type>;

    analytical_eigenproblem<real_type> identity_problem(
        "one_dimensional_identity",
        1,
        {real_type(1)},
        {});
    auto real_space = std::make_shared<real_space_type>(1);
    product_space_type product_space(*real_space, *real_space);
    auto complex_space = std::make_shared<complex_space_type>(1);
    bridge_type bridge(product_space, *complex_space);
    real_operator_type real_operator(*real_space, identity_problem);
    complexified_operator_type complexified(
        product_space,
        bridge,
        real_operator);
    identity_type identity(*complex_space);
    factor_operator_type multiplication_by_i(
        *complex_space,
        complexified,
        identity,
        complex_traits::make(real_type(0), real_type(1)),
        complex_traits::make(real_type(0), real_type(0)));

    typename solver_type::params parameters;
    parameters.basis_size = 1;
    parameters.batch_size = 1;
    parameters.preconditioner_side = 'L';
    parameters.monitor.rel_tol = 1.0e-13;
    parameters.monitor.abs_tol = 1.0e-14;
    parameters.monitor.max_iters_num = 2;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    auto preconditioner =
        std::make_shared<preconditioner_type>(complex_space);
    solver_type solver(
        complex_space,
        nullptr,
        parameters,
        preconditioner);

    nmfd::detail::vector_wrap<complex_space_type, true, true>
        right_hand_side(*complex_space);
    nmfd::detail::vector_wrap<complex_space_type, true, true>
        solution(*complex_space);
    set_complex_vector(
        *complex_space,
        std::vector<std::complex<real_type>>{{1.0, 0.0}},
        *right_hand_side);
    complex_space->assign_scalar(complex_type{}, *solution);
    require(
        solver.solve(
            multiplication_by_i,
            *right_hand_side,
            *solution),
        label + " true complex GMRES solves i*z=1");
    const auto actual = get_complex_vector(
        *complex_space,
        *solution);
    require(
        close(
            actual[0],
            std::complex<real_type>(0.0, -1.0),
            2.0e-12),
        label + " true complex GMRES coefficient");
    require(
        solver.monitor().iters_performed() == 1,
        label + " true complex GMRES uses one Krylov direction");
}

template<class Backend>
void test_factor_bundle(const std::string& label)
{
    using real_type = double;
    using real_space_type =
        scfd_vector_operations<Backend, real_type>;
    using complex_type = backend_complex_type<Backend, real_type>;
    using complex_traits =
        common::scfd_backend_ext::complex_value_traits<
            complex_type>;
    using complex_space_type =
        scfd_vector_operations<Backend, complex_type>;
    using real_operator_type =
        analytical_dense_operator<real_space_type, real_type>;
    using model_type =
        analytical_real_affine_inverse_model<real_space_type>;
    using provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                real_space_type,
                model_type>;
    using factorization_types =
        stability::eigensolvers::transformations::
            matrix_free_complex_factorization_types<
                real_space_type,
                complex_space_type,
                real_operator_type,
                provider_type>;
    using factor_operator_type =
        typename factorization_types::factor_operator_type;
    using preconditioner_type =
        typename factorization_types::preconditioner_type;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<
            complex_space_type,
            log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            complex_space_type,
            monitor_type,
            log_type,
            factor_operator_type,
            preconditioner_type>;
    using owner_type =
        stability::eigensolvers::transformations::
            matrix_free_complex_factor_solver_bundle<
                factorization_types,
                solver_type>;
    using bundle_type = typename owner_type::bundle_type;

    const auto problem = nonnormal_eigenproblem<real_type>();
    auto real_space =
        std::make_shared<real_space_type>(problem.dimension());
    auto complex_space =
        std::make_shared<complex_space_type>(problem.dimension());
    real_operator_type real_operator(*real_space, problem);

    std::vector<real_type> diagonal(problem.dimension());
    for(std::size_t index = 0; index < diagonal.size(); ++index)
        diagonal[index] = problem.matrix(index, index);
    auto model =
        std::make_shared<model_type>(*real_space, diagonal);
    auto provider =
        std::make_shared<provider_type>(*real_space, *model);

    const std::vector<typename bundle_type::factor_type> factors{
        {{0.20, 0.05}, {1.10, -0.40}},
        {{-0.15, 0.08}, {0.80, 0.30}}};

    typename solver_type::params parameters;
    parameters.basis_size = 3;
    parameters.batch_size = 3;
    parameters.preconditioner_side = 'L';
    parameters.orthogonalization = "mgs";
    parameters.reorthogonalization_policy = "dgks";
    parameters.max_orthogonalization_passes = 2;
    parameters.monitor.rel_tol = 1.0e-12;
    parameters.monitor.abs_tol = 1.0e-13;
    parameters.monitor.max_iters_num = 24;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    owner_type bundle(
        real_space,
        complex_space,
        real_operator,
        provider,
        factors,
        parameters);

    const std::vector<std::complex<real_type>> host_right_hand_side{
        {1.0, -0.25},
        {-0.3, 0.8},
        {2.0, 1.5}};
    const auto expected = expected_factorized_solution(
        problem,
        factors,
        host_right_hand_side);
    nmfd::detail::vector_wrap<complex_space_type, true, true>
        right_hand_side(*complex_space);
    nmfd::detail::vector_wrap<complex_space_type, true, true>
        solution(*complex_space);
    set_complex_vector(
        *complex_space,
        host_right_hand_side,
        *right_hand_side);
    complex_space->assign_scalar(
        complex_traits::make(17.0, -19.0),
        *solution);
    require(
        bundle.solve(*right_hand_side, *solution),
        label + " matrix-free complex factor bundle solve");
    const auto actual = get_complex_vector(
        *complex_space,
        *solution);
    for(std::size_t index = 0; index < actual.size(); ++index)
    {
        require(
            close(actual[index], expected[index], 3.0e-9),
            label + " matrix-free factorized value");
    }
    require(
        bundle.complexified_operator().operator_calls() > 0 &&
        bundle.complexified_operator().component_operator_calls() ==
            2*bundle.complexified_operator().operator_calls() &&
        bundle.complexified_operator().
            component_operator_failures() == 0,
        label + " factor bundle uses two real Jacobian applications");
    require(
        provider->apply_calls() > 0 &&
        provider->failed_applications() == 0 &&
        model->apply_calls() == provider->apply_calls(),
        label + " factor bundle uses only the real affine provider");
    const auto statistics = bundle.statistics();
    require(
        statistics.solve_calls == 1 &&
        statistics.factor_solve_calls == factors.size() &&
        statistics.failed_solves == 0 &&
        statistics.factors.size() == factors.size(),
        label + " complex factor bundle statistics");
    for(std::size_t index = 0; index < factors.size(); ++index)
    {
        require(
            bundle.factor(index).preconditioner().apply_calls() > 0 &&
            bundle.factor(index).preconditioner().
                failed_applications() == 0,
            label + " complexified affine preconditioner statistics");
    }

    owner_type configured_bundle(
        real_space,
        complex_space,
        real_operator,
        provider,
        factors,
        parameters,
        [](
            const typename solver_type::params& base,
            const typename bundle_type::factor_type&,
            std::size_t index)
        {
            auto configured = base;
            configured.basis_size =
                base.basis_size +
                static_cast<decltype(base.basis_size)>(index);
            return configured;
        });
    require(
        configured_bundle.factor(0).
            solver_parameters().basis_size ==
            parameters.basis_size &&
        configured_bundle.factor(1).
            solver_parameters().basis_size ==
            parameters.basis_size + 1,
        label + " per-factor solver parameters");

    auto failing_model =
        std::make_shared<model_type>(*real_space, diagonal);
    failing_model->fail_after(0);
    auto failing_provider =
        std::make_shared<provider_type>(
            *real_space,
            *failing_model);
    preconditioner_type failing_preconditioner(
        bundle.product_space(),
        bundle.bridge(),
        failing_provider,
        factors.front());
    const std::vector<std::complex<real_type>> sentinel(
        problem.dimension(),
        {-13.0, 19.0});
    set_complex_vector(*complex_space, sentinel, *solution);
    require(
        !failing_preconditioner.apply(
            *right_hand_side,
            *solution),
        label + " real affine provider failure propagates");
    const auto after_failure = get_complex_vector(
        *complex_space,
        *solution);
    for(std::size_t index = 0; index < sentinel.size(); ++index)
    {
        require(
            close(after_failure[index], sentinel[index], 2.0e-12),
            label + " affine provider failure is transactional");
    }
}

template<class Backend>
void run_backend(const std::string& label)
{
    test_operator<Backend, float>(label + " float");
    test_operator<Backend, double>(label + " double");
    test_true_complex_gmres<Backend>(label);
    test_factor_bundle<Backend>(label);
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

} // namespace matrix_free_complexification_test
} // namespace tests
} // namespace stability

#endif
