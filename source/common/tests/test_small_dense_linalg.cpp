#include <cmath>
#include <iostream>
#include <string>

#include <nmfd/operations/linalg/small_dense.h>

namespace
{

int checks = 0;
int failures = 0;

void fail(const std::string& label)
{
    std::cout << "FAIL " << label << std::endl;
    ++failures;
}

template<class T>
void require_near(const std::string& label, T value, T expected, T tol)
{
    ++checks;
    const T err = std::abs(value - expected);
    if(!(err <= tol))
    {
        std::cout << "FAIL " << label << " value=" << value
                  << " expected=" << expected << " err=" << err
                  << " tol=" << tol << std::endl;
        ++failures;
    }
}

void require_status(
    const std::string& label,
    nmfd::operations::linalg::small_solve_status value,
    nmfd::operations::linalg::small_solve_status expected)
{
    ++checks;
    if(value != expected)
    {
        std::cout << "FAIL " << label << " status="
                  << nmfd::operations::linalg::small_solve_status_name(value)
                  << " expected="
                  << nmfd::operations::linalg::small_solve_status_name(expected)
                  << std::endl;
        ++failures;
    }
}

template<class T>
void test_type(const std::string& name, T tol)
{
    using namespace nmfd::operations::linalg;

    {
        small_matrix<T, 3> A{
            {T(0), T(2), T(1)},
            {T(1), T(-2), T(-3)},
            {T(-1), T(1), T(2)}
        };
        small_vector<T, 3> x_ref{T(1), T(-2), T(3)};
        small_vector<T, 3> b(3);
        for(std::size_t i = 0; i < 3; ++i)
        {
            b[i] = T{};
            for(std::size_t j = 0; j < 3; ++j)
                b[i] += A(i, j)*x_ref[j];
        }

        small_vector<T, 3> x;
        const auto info = solve(A, b, x);
        require_status(name + " pivoted solve status", info.status, small_solve_status::success);
        for(std::size_t i = 0; i < 3; ++i)
            require_near(name + " pivoted solve x[" + std::to_string(i) + "]", x[i], x_ref[i], tol);
    }

    {
        small_matrix<T, 2> A{
            {T(4), T(7)},
            {T(2), T(6)}
        };
        small_matrix<T, 2> A_inv;
        const auto info = inverse(A, A_inv);
        require_status(name + " inverse status", info.status, small_solve_status::success);
        require_near(name + " determinant", info.determinant, T(10), tol*T(10));
        require_near(name + " inverse 00", A_inv(0, 0), T(0.6), tol*T(10));
        require_near(name + " inverse 01", A_inv(0, 1), T(-0.7), tol*T(10));
        require_near(name + " inverse 10", A_inv(1, 0), T(-0.2), tol*T(10));
        require_near(name + " inverse 11", A_inv(1, 1), T(0.4), tol*T(10));

        small_matrix<T, 2> I;
        multiply(A, A_inv, I);
        require_near(name + " A invA 00", I(0, 0), T(1), tol*T(50));
        require_near(name + " A invA 01", I(0, 1), T(0), tol*T(50));
        require_near(name + " A invA 10", I(1, 0), T(0), tol*T(50));
        require_near(name + " A invA 11", I(1, 1), T(1), tol*T(50));
    }

    {
        small_matrix<T, 2> A{
            {T(2), T(0)},
            {T(0), T(4)}
        };
        small_matrix<T, 2, 3> B{
            {T(2), T(4), T(6)},
            {T(8), T(12), T(16)}
        };
        small_matrix<T, 2, 3> X;
        const auto info = solve_multiple_rhs(A, B, X);
        require_status(name + " multi-rhs status", info.status, small_solve_status::success);
        require_near(name + " multi-rhs 00", X(0, 0), T(1), tol);
        require_near(name + " multi-rhs 01", X(0, 1), T(2), tol);
        require_near(name + " multi-rhs 02", X(0, 2), T(3), tol);
        require_near(name + " multi-rhs 10", X(1, 0), T(2), tol);
        require_near(name + " multi-rhs 11", X(1, 1), T(3), tol);
        require_near(name + " multi-rhs 12", X(1, 2), T(4), tol);
    }

    {
        small_matrix<T, 2> A{
            {T(1), T(2)},
            {T(2), T(4)}
        };
        small_vector<T, 2> b{T(1), T(2)};
        small_vector<T, 2> x;
        const auto info = solve(A, b, x);
        require_status(name + " singular status", info.status, small_solve_status::singular);
        ++checks;
        if(info.rank != 1)
            fail(name + " singular rank");
    }

    {
        small_matrix<T, 2> A{
            {T(1), T(0)},
            {T(0), T(1e-14)}
        };
        small_vector<T, 2> b{T(1), T(1)};
        small_vector<T, 2> x;
        const auto info = solve(A, b, x, T(1e-20), T(1e-10));
        require_status(name + " ill-conditioned status", info.status, small_solve_status::ill_conditioned);
    }
}

} // namespace

int main()
{
    test_type<float>("float", 1e-4f);
    test_type<double>("double", 1e-11);

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
