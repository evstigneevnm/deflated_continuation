#include <array>
#include <cstdlib>
#include <iostream>

#include <stability/eigensolvers/transformations/affine_inverse_health.h>
#include <stability/model_adapter_contract.h>

namespace
{

using test_vector_type = std::array<double, 2>;

struct vector_operations
{
    using scalar_type = double;
    using vector_type = test_vector_type;
};

struct linearization_provider
{
    void set_linearization_point(const test_vector_type&, double)
    {
    }
};

struct missing_linearization_provider
{
};

struct bool_operator
{
    bool apply(const test_vector_type&, test_vector_type&) const
    {
        return true;
    }
};

struct void_operator
{
    void apply(const test_vector_type&, test_vector_type&) const
    {
    }
};

struct missing_operator
{
};

struct affine_provider
{
    using scalar_type = double;
    using vector_type = test_vector_type;
    using health_type =
        stability::eigensolvers::transformations::
            affine_inverse_health<scalar_type>;

    bool apply(
        scalar_type,
        scalar_type,
        const vector_type&,
        vector_type&) const
    {
        return true;
    }

    health_type health(scalar_type, scalar_type) const
    {
        return {};
    }
};

struct incomplete_affine_provider
{
    using scalar_type = double;
    using vector_type = test_vector_type;
    using health_type =
        stability::eigensolvers::transformations::
            affine_inverse_health<scalar_type>;

    bool apply(
        scalar_type,
        scalar_type,
        const vector_type&,
        vector_type&) const
    {
        return true;
    }
};

struct wrong_health_affine_provider
{
    using scalar_type = double;
    using vector_type = test_vector_type;
    using health_type =
        stability::eigensolvers::transformations::
            affine_inverse_health<scalar_type>;

    bool apply(
        scalar_type,
        scalar_type,
        const vector_type&,
        vector_type&) const
    {
        return true;
    }

    bool health(scalar_type, scalar_type) const
    {
        return true;
    }
};

struct eigensolver_adapter
{
    stability::eigensolvers::eigensolver_result<double>
    execute(const test_vector_type&)
    {
        return {};
    }
};

struct wrong_eigensolver_adapter
{
    bool execute(const test_vector_type&)
    {
        return true;
    }
};

static_assert(
    stability::model_adapter::is_linearization_provider_v<
        linearization_provider,
        test_vector_type,
        double>);
static_assert(
    !stability::model_adapter::is_linearization_provider_v<
        missing_linearization_provider,
        test_vector_type,
        double>);
static_assert(
    stability::model_adapter::is_real_operator_v<
        bool_operator,
        test_vector_type>);
static_assert(
    stability::model_adapter::is_real_operator_v<
        void_operator,
        test_vector_type>);
static_assert(
    !stability::model_adapter::is_real_operator_v<
        missing_operator,
        test_vector_type>);
static_assert(
    stability::model_adapter::is_real_affine_inverse_provider_v<
        affine_provider,
        test_vector_type,
        double>);
static_assert(
    !stability::model_adapter::is_real_affine_inverse_provider_v<
        incomplete_affine_provider,
        test_vector_type,
        double>);
static_assert(
    !stability::model_adapter::is_real_affine_inverse_provider_v<
        wrong_health_affine_provider,
        test_vector_type,
        double>);
static_assert(
    stability::model_adapter::is_eigensolver_adapter_v<
        eigensolver_adapter,
        test_vector_type,
        double>);
static_assert(
    !stability::model_adapter::is_eigensolver_adapter_v<
        wrong_eigensolver_adapter,
        test_vector_type,
        double>);
static_assert(
    stability::model_adapter::evaluator_contract<
        vector_operations,
        linearization_provider,
        eigensolver_adapter>::value);
static_assert(
    stability::model_adapter::matrix_free_contract<
        vector_operations,
        bool_operator,
        affine_provider>::value);

} // namespace

int main()
{
    std::cout << "Stability model adapter contract checks: 12, failures: 0\n";
    return EXIT_SUCCESS;
}
