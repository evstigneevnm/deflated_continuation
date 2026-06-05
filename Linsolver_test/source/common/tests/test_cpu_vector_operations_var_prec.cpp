#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <common/cpu_vector_operations_var_prec.h>

namespace
{

struct test_report
{
    std::size_t checks = 0;
    std::size_t failures = 0;

    void require(bool condition, const std::string& label)
    {
        ++checks;
        if(!condition)
        {
            ++failures;
            std::cout << "FAIL " << label << std::endl;
        }
    }
};

template<class T>
void check_equal(test_report& report, const std::string& label, const T& got, const T& expected)
{
    ++report.checks;
    if(!(got == expected))
    {
        ++report.failures;
        std::cout << "FAIL " << label
                  << " got=" << std::setprecision(std::numeric_limits<T>::digits10 + 5) << got
                  << " expected=" << expected << std::endl;
    }
}

template<class T>
void check_vector_equal(test_report& report, const std::string& label, const std::vector<T>& got, const std::vector<T>& expected)
{
    report.require(got.size() == expected.size(), label + " size");
    const auto n = std::min(got.size(), expected.size());
    for(std::size_t i = 0; i < n; ++i)
    {
        check_equal(report, label + "[" + std::to_string(i) + "]", got[i], expected[i]);
    }
}

template<class F>
void expect_throw(test_report& report, const std::string& label, F&& f)
{
    ++report.checks;
    try
    {
        f();
    }
    catch(const std::logic_error&)
    {
        return;
    }
    catch(const std::exception& e)
    {
        ++report.failures;
        std::cout << "FAIL " << label << " threw unexpected exception: " << e.what() << std::endl;
        return;
    }
    ++report.failures;
    std::cout << "FAIL " << label << " did not throw" << std::endl;
}

template<class T>
T pow2(unsigned int exponent)
{
    T value = 1;
    for(unsigned int i = 0; i < exponent; ++i)
    {
        value *= 2;
    }
    return value;
}

void run_var_prec_tests(test_report& report)
{
    using vec_ops_t = cpu_vector_operations_var_prec<80>;
    using default_vec_ops_t = cpu_vector_operations_var_prec<>;
    using T = typename vec_ops_t::scalar_type;
    using vector_type = typename vec_ops_t::vector_type;
    using multivector_type = typename vec_ops_t::multivector_type;
    using nmfd_base_type = typename vec_ops_t::nmfd_parent_type;

    static_assert(vec_ops_t::SignificantBits == 80, "explicit precision must be preserved");
    static_assert(default_vec_ops_t::SignificantBits == 100, "default precision must stay 100 bits");

    vec_ops_t vec_ops(5);
    report.require(vec_ops.get_fp_prec() == 80, "get_fp_prec reports template precision");

    const nmfd_base_type& nmfd_base = vec_ops;
    vector_type bx;
    vector_type by;
    vector_type bz;
    nmfd_base.init_vector(bx);
    nmfd_base.init_vector(by);
    nmfd_base.init_vector(bz);
    nmfd_base.start_use_vector(bx);
    nmfd_base.start_use_vector(by);
    nmfd_base.start_use_vector(bz);
    nmfd_base.assign_scalar(T(2), bx);
    nmfd_base.assign_lin_comb(T(3), bx, by);
    check_vector_equal(report, "NMFD assign_lin_comb one-vector", by, vector_type{T(6), T(6), T(6), T(6), T(6)});
    nmfd_base.assign_lin_comb(T(2), bx, T(-1), by, bz);
    check_vector_equal(report, "NMFD assign_lin_comb two-vector", bz, vector_type{T(-2), T(-2), T(-2), T(-2), T(-2)});
    nmfd_base.add_lin_comb(T(1), bx, T(1), bz);
    check_vector_equal(report, "NMFD add_lin_comb", bz, vector_type{T(0), T(0), T(0), T(0), T(0)});
    check_equal(report, "NMFD scalar_prod_l2", nmfd_base.scalar_prod_l2(bx, by), T(60));

    multivector_type mv;
    nmfd_base.init_multivector(mv, 2);
    nmfd_base.start_use_multivector(mv, 2);
    nmfd_base.assign(bx, mv, 2, 0);
    nmfd_base.assign(by, mv, 2, 1);
    nmfd_base.assign(mv, 2, 0, bz);
    check_vector_equal(report, "NMFD multivector assign", bz, vector_type{T(2), T(2), T(2), T(2), T(2)});
    check_equal(report, "NMFD multivector scalar_prod", nmfd_base.scalar_prod(mv, 2, 1, bx), T(60));
    nmfd_base.add_lin_comb(T(2), mv, 2, 0, T(-1), bz);
    check_vector_equal(report, "NMFD multivector add_lin_comb", bz, vector_type{T(2), T(2), T(2), T(2), T(2)});
    nmfd_base.stop_use_multivector(mv, 2);
    nmfd_base.free_multivector(mv, 2);
    nmfd_base.stop_use_vector(bx);
    nmfd_base.stop_use_vector(by);
    nmfd_base.stop_use_vector(bz);
    nmfd_base.free_vector(bx);
    nmfd_base.free_vector(by);
    nmfd_base.free_vector(bz);

    vec_ops_t one_ops(1);
    vector_type one_x;
    one_ops.init_vector(one_x);
    one_ops.start_use_vector(one_x);
    one_x = {T(-7)};
    check_equal(report, "sum exact size one", one_ops.sum(one_x), T(-7));
    check_equal(report, "asum exact size one", one_ops.asum(one_x), T(7));
    check_equal(report, "scalar_prod exact size one", one_ops.scalar_prod(one_x, one_x), T(49));
    one_ops.stop_use_vector(one_x);
    one_ops.free_vector(one_x);

    vector_type x;
    vector_type y;
    vector_type z;
    vector_type y_long;
    vector_type z_long;
    vector_type short_x;
    vector_type normal_x;
    vec_ops.init_vectors(x, y, z, y_long, z_long, short_x, normal_x);
    vec_ops.start_use_vectors(x, y, z, normal_x);
    vec_ops.start_use_vector(y_long, 6);
    vec_ops.start_use_vector(z_long, 6);
    vec_ops.start_use_vector(short_x, 3);

    x = {T(1), T(-2), T(4), T(-8), T(16)};
    y = {T(2), T(4), T(-8), T(-16), T(32)};
    check_equal(report, "sum exact signed powers", vec_ops.sum(x), T(11));
    check_equal(report, "asum exact signed powers", vec_ops.asum(x), T(31));
    check_equal(report, "scalar_prod exact signed powers", vec_ops.scalar_prod(x, y), T(602));
    check_equal(report, "norm_sq exact signed powers", vec_ops.norm_sq(x), T(341));

    normal_x = {T(1), T(2), T(2), T(0), T(0)};
    check_equal(report, "norm exact 3-4-? vector", vec_ops.norm(normal_x), T(3));

    const T big = pow2<T>(70);
    x = {big, T(1), -big, T(0), T(0)};
    check_equal(report, "sum exact cancellation", vec_ops.sum(x), T(1));
    check_equal(report, "asum exact cancellation", vec_ops.asum(x), T(2)*big + T(1));

    short_x = {T(1), T(-2), T(4)};
    check_equal(report, "sum exact non-default size", vec_ops.sum(short_x), T(3));
    check_equal(report, "asum exact non-default size", vec_ops.asum(short_x), T(7));
    check_equal(report, "scalar_prod exact non-default size", vec_ops.scalar_prod(short_x, short_x), T(21));

    x = {T(8), T(18), T(0), T(0), T(0)};
    y = {T(2), T(3), T(1), T(1), T(1)};
    vec_ops.div_pointwise(x, T(2), y);
    check_vector_equal(report, "div_pointwise in-place exact", x, vector_type{T(2), T(3), T(0), T(0), T(0)});

    x = {T(1), T(2), T(3), T(4), T(5)};
    y = {T(6), T(7), T(8), T(9), T(10)};
    z = {T(0), T(0), T(0), T(0), T(0)};
    y_long = {T(6), T(7), T(8), T(9), T(10), T(11)};
    z_long = {T(0), T(0), T(0), T(0), T(0), T(0)};

    expect_throw(report, "assign_mul rejects mismatched y", [&]()
    {
        vec_ops.assign_mul(T(1), x, T(1), y_long, z);
    });
    expect_throw(report, "assign_mul rejects mismatched z", [&]()
    {
        vec_ops.assign_mul(T(1), x, T(1), y, z_long);
    });
    expect_throw(report, "add_mul rejects mismatched y", [&]()
    {
        vec_ops.add_mul(T(1), x, T(1), y_long, T(0), z);
    });
    expect_throw(report, "mul_pointwise rejects mismatched y", [&]()
    {
        vec_ops.mul_pointwise(T(1), x, T(1), y_long, z);
    });
    expect_throw(report, "div_pointwise rejects mismatched y", [&]()
    {
        vec_ops.div_pointwise(T(1), x, T(1), y_long, z);
    });

    report.require(vec_ops.check_is_valid_number(x), "check_is_valid_number accepts finite values");
    x[2] = std::numeric_limits<T>::quiet_NaN();
    report.require(!vec_ops.check_is_valid_number(x), "check_is_valid_number rejects NaN");
    x[2] = std::numeric_limits<T>::infinity();
    report.require(!vec_ops.check_is_valid_number(x), "check_is_valid_number rejects infinity");

    vec_ops.stop_use_vectors(x, y, z, y_long, z_long, short_x, normal_x);
    vec_ops.free_vectors(x, y, z, y_long, z_long, short_x, normal_x);
}

} // namespace

int main()
{
    test_report report;
    run_var_prec_tests(report);

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if(report.failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
