#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_MATRIX_FREE_COMPLEX_FACTOR_SOLVER_BUNDLE_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_MATRIX_FREE_COMPLEX_FACTOR_SOLVER_BUNDLE_H__

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <common/scfd_backend_ext/complex.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/scfd_complex_vector_bridge.h>

#include "affine_pencil_operator.h"
#include "complexified_real_affine_preconditioner.h"
#include "complexified_real_operator.h"
#include "identity_operator.h"
#include "iterative_factor_solver_bundle.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<
    class RealVectorSpace,
    class ComplexVectorSpace,
    class RealOperator,
    class RealAffineInverseProvider>
struct matrix_free_complex_factorization_types
{
    using real_space_type = RealVectorSpace;
    using complex_space_type = ComplexVectorSpace;
    using real_operator_type = RealOperator;
    using provider_type = RealAffineInverseProvider;
    using product_space_type =
        nmfd::operations::two_block_vector_space<real_space_type>;
    using bridge_type =
        nmfd::operations::scfd_complex_vector_bridge<
            product_space_type,
            complex_space_type>;
    using complexified_operator_type =
        complexified_real_operator<
            product_space_type,
            complex_space_type,
            bridge_type,
            real_operator_type>;
    using identity_operator_type =
        identity_operator<complex_space_type>;
    using factor_operator_type =
        affine_pencil_operator<
            complex_space_type,
            complexified_operator_type,
            identity_operator_type>;
    using preconditioner_type =
        complexified_real_affine_preconditioner<
            product_space_type,
            complex_space_type,
            bridge_type,
            factor_operator_type,
            provider_type>;

    template<class LinearSolver>
    using bundle_type =
        iterative_factor_solver_bundle<
            complex_space_type,
            factor_operator_type,
            preconditioner_type,
            LinearSolver>;
};

template<class FactorizationTypes, class LinearSolver>
class matrix_free_complex_factor_solver_bundle
{
public:
    using types = FactorizationTypes;
    using real_space_type = typename types::real_space_type;
    using complex_space_type = typename types::complex_space_type;
    using real_operator_type = typename types::real_operator_type;
    using provider_type = typename types::provider_type;
    using product_space_type = typename types::product_space_type;
    using bridge_type = typename types::bridge_type;
    using complexified_operator_type =
        typename types::complexified_operator_type;
    using identity_operator_type =
        typename types::identity_operator_type;
    using factor_operator_type =
        typename types::factor_operator_type;
    using preconditioner_type =
        typename types::preconditioner_type;
    using solver_type = LinearSolver;
    using bundle_type =
        typename types::template bundle_type<solver_type>;
    using factor_type = typename bundle_type::factor_type;
    using vector_type = typename complex_space_type::vector_type;
    using solver_parameters_type =
        typename bundle_type::solver_parameters_type;
    using log_type = typename bundle_type::log_type;
    using statistics_type = typename bundle_type::statistics_type;

    static_assert(
        std::is_same<
            typename solver_type::vector_operations_type,
            complex_space_type>::value,
        "matrix-free factor solver must use the complex vector space");
    static_assert(
        std::is_same<
            typename solver_type::linear_operator_type,
            factor_operator_type>::value,
        "matrix-free factor solver has the wrong operator type");
    static_assert(
        std::is_same<
            typename solver_type::preconditioner_type,
            preconditioner_type>::value,
        "matrix-free factor solver has the wrong preconditioner type");

    matrix_free_complex_factor_solver_bundle(
        std::shared_ptr<real_space_type> real_space,
        std::shared_ptr<complex_space_type> complex_space,
        const real_operator_type& real_operator,
        std::shared_ptr<const provider_type> provider,
        std::vector<factor_type> factors,
        const solver_parameters_type& solver_parameters,
        log_type* log = nullptr)
        : matrix_free_complex_factor_solver_bundle(
              std::move(real_space),
              std::move(complex_space),
              real_operator,
              std::move(provider),
              std::move(factors),
              solver_parameters,
              typename bundle_type::uniform_solver_parameters{},
              log)
    {
    }

    template<class SolverParametersFactory>
    matrix_free_complex_factor_solver_bundle(
        std::shared_ptr<real_space_type> real_space,
        std::shared_ptr<complex_space_type> complex_space,
        const real_operator_type& real_operator,
        std::shared_ptr<const provider_type> provider,
        std::vector<factor_type> factors,
        const solver_parameters_type& solver_parameters,
        SolverParametersFactory&& solver_parameters_factory,
        log_type* log = nullptr)
        : real_space_(std::move(real_space)),
          complex_space_(std::move(complex_space)),
          provider_(checked_provider(std::move(provider))),
          product_space_(
              checked_real_space(),
              checked_real_space()),
          bridge_(
              product_space_,
              checked_complex_space()),
          complexified_operator_(
              product_space_,
              bridge_,
              real_operator),
          identity_operator_(checked_complex_space()),
          bundle_(
              complex_space_,
              std::move(factors),
              solver_parameters,
              [this](
                  const factor_type& factor,
                  std::size_t)
              {
                  return make_components(factor);
              },
              std::forward<SolverParametersFactory>(
                  solver_parameters_factory),
              log)
    {
    }

    matrix_free_complex_factor_solver_bundle(
        const matrix_free_complex_factor_solver_bundle&) = delete;
    matrix_free_complex_factor_solver_bundle& operator=(
        const matrix_free_complex_factor_solver_bundle&) = delete;
    matrix_free_complex_factor_solver_bundle(
        matrix_free_complex_factor_solver_bundle&&) = delete;
    matrix_free_complex_factor_solver_bundle& operator=(
        matrix_free_complex_factor_solver_bundle&&) = delete;

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        return bundle_.solve(right_hand_side, solution);
    }

    bundle_type& bundle()
    {
        return bundle_;
    }

    const bundle_type& bundle() const
    {
        return bundle_;
    }

    const product_space_type& product_space() const
    {
        return product_space_;
    }

    const real_space_type& real_space() const
    {
        return *real_space_;
    }

    const complex_space_type& complex_space() const
    {
        return *complex_space_;
    }

    const bridge_type& bridge() const
    {
        return bridge_;
    }

    const complexified_operator_type& complexified_operator() const
    {
        return complexified_operator_;
    }

    const provider_type& provider() const
    {
        return *provider_;
    }

    std::size_t factor_count() const
    {
        return bundle_.factor_count();
    }

    const typename bundle_type::factor_state& factor(
        std::size_t index) const
    {
        return bundle_.factor(index);
    }

    typename bundle_type::factor_state& factor(std::size_t index)
    {
        return bundle_.factor(index);
    }

    statistics_type statistics() const
    {
        return bundle_.statistics();
    }

    void reset_statistics() const
    {
        bundle_.reset_statistics();
    }

private:
    static std::shared_ptr<const provider_type> checked_provider(
        std::shared_ptr<const provider_type> provider)
    {
        if(!provider)
            throw std::invalid_argument(
                "matrix-free factor bundle requires a real affine provider");
        return provider;
    }

    real_space_type& checked_real_space() const
    {
        if(!real_space_)
            throw std::invalid_argument(
                "matrix-free factor bundle requires a real vector space");
        return *real_space_;
    }

    complex_space_type& checked_complex_space() const
    {
        if(!complex_space_)
            throw std::invalid_argument(
                "matrix-free factor bundle requires a complex vector space");
        return *complex_space_;
    }

    typename bundle_type::components_type make_components(
        const factor_type& factor)
    {
        using complex_scalar_type =
            typename complex_space_type::scalar_type;
        using complex_traits =
            common::scfd_backend_ext::complex_value_traits<
                complex_scalar_type>;
        return
        {
            std::make_shared<factor_operator_type>(
                checked_complex_space(),
                complexified_operator_,
                identity_operator_,
                complex_traits::make(
                    factor.operator_scale.real(),
                    factor.operator_scale.imag()),
                complex_traits::make(
                    factor.diagonal_shift.real(),
                    factor.diagonal_shift.imag())),
            std::make_shared<preconditioner_type>(
                product_space_,
                bridge_,
                provider_,
                factor)
        };
    }

    std::shared_ptr<real_space_type> real_space_;
    std::shared_ptr<complex_space_type> complex_space_;
    std::shared_ptr<const provider_type> provider_;
    product_space_type product_space_;
    bridge_type bridge_;
    complexified_operator_type complexified_operator_;
    identity_operator_type identity_operator_;
    bundle_type bundle_;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
