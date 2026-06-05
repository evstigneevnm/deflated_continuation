#ifndef __SCFD_VECTOR_OPERATIONS_NMFD_INTERFACE_TESTS_H__
#define __SCFD_VECTOR_OPERATIONS_NMFD_INTERFACE_TESTS_H__

#include <cstddef>
#include <string>
#include <vector>

#include <common/tests/vector_operations_template_tests.h>

namespace vector_operations_tests
{

template<class VecOps, class Access>
void run_nmfd_vector_space_interface_tests(
    VecOps& vec_ops,
    Access access,
    std::size_t n,
    const std::string& label,
    test_report& report)
{
    using base_type = typename VecOps::nmfd_parent_type;
    using scalar_type = typename VecOps::scalar_type;
    using vector_type = typename VecOps::vector_type;
    using multivector_type = typename VecOps::multivector_type;

    const base_type& base = vec_ops;

    vector_type x;
    vector_type y;
    vector_type z;
    base.init_vector(x);
    base.init_vector(y);
    base.init_vector(z);
    base.start_use_vector(x);
    base.start_use_vector(y);
    base.start_use_vector(z);

    const auto two = make_scalar<scalar_type>(2.0L, 0.0L);
    const auto three = make_scalar<scalar_type>(3.0L, 0.0L);
    const auto minus_one = make_scalar<scalar_type>(-1.0L, 0.0L);
    const auto half = make_scalar<scalar_type>(0.5L, 0.0L);

    base.assign_scalar(two, x);
    base.assign_lin_comb(three, x, y);
    check_vector_close(
        report,
        label + " NMFD assign_lin_comb one-vector",
        access.read(vec_ops, y, n),
        std::vector<scalar_type>(n, make_scalar<scalar_type>(6.0L, 0.0L)));

    base.assign_lin_comb(two, x, minus_one, y, z);
    check_vector_close(
        report,
        label + " NMFD assign_lin_comb two-vector",
        access.read(vec_ops, z, n),
        std::vector<scalar_type>(n, make_scalar<scalar_type>(-2.0L, 0.0L)));

    base.add_lin_comb(half, x, half, y);
    check_vector_close(
        report,
        label + " NMFD add_lin_comb",
        access.read(vec_ops, y, n),
        std::vector<scalar_type>(n, make_scalar<scalar_type>(4.0L, 0.0L)));

    const auto expected_dot = make_scalar<scalar_type>(8.0L*static_cast<long double>(n), 0.0L);
    check_close(report, label + " NMFD scalar_prod_l2", base.scalar_prod_l2(x, y), expected_dot, 1024.0L);

    multivector_type mv;
    base.init_multivector(mv, 2);
    base.start_use_multivector(mv, 2);
    base.assign(x, mv, 2, 0);
    base.assign(y, mv, 2, 1);
    base.assign(mv, 2, 0, z);
    check_vector_close(
        report,
        label + " NMFD multivector assign out",
        access.read(vec_ops, z, n),
        std::vector<scalar_type>(n, two));
    check_close(report, label + " NMFD multivector scalar_prod", base.scalar_prod(mv, 2, 1, x), expected_dot, 1024.0L);

    base.add_lin_comb(three, mv, 2, 0, minus_one, z);
    check_vector_close(
        report,
        label + " NMFD multivector add_lin_comb",
        access.read(vec_ops, z, n),
        std::vector<scalar_type>(n, make_scalar<scalar_type>(4.0L, 0.0L)));

    base.stop_use_multivector(mv, 2);
    base.free_multivector(mv, 2);

    base.stop_use_vector(x);
    base.stop_use_vector(y);
    base.stop_use_vector(z);
    base.free_vector(x);
    base.free_vector(y);
    base.free_vector(z);
}

} // namespace vector_operations_tests

#endif
